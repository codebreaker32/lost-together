"""OpenCV 5 tools as an MCP server (stdio), plus an MCP-backed executor for the agent loop.

Server:
    uv run python -m lost_together.agent.mcp_server [--root DIR ...] [--no-yolox]
Tools: load_frame(path) -> frame_id; detect_candidates(frame_id); look_closer(frame_id, candidate_id, zoom);
compare_frames(prev_frame_id, frame_id, candidate_id); query_map(summary). Frames are referenced by id from a
bounded in-memory store; load_frame only reads images under the allowed roots (default: data/ and results/).
Images (annotated frame, enhanced crop) come back as MCP image content next to a JSON text block.

Client side: `McpExecutor` gives the loop the same interface as `LocalExecutor`, so the agent's tool calls go
through MCP (one tool layer). It holds one MCP session on a background event loop.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from . import tools as T

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [ROOT / "data", ROOT / "results"]


def build_server(toolbox: T.Toolbox | None = None, roots: list[str | Path] | None = None, use_yolox: bool = True):
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError

    tb = toolbox or T.Toolbox(use_yolox=use_yolox, allowed_roots=roots or DEFAULT_ROOTS)
    mcp = MCPServer("lost-together-opencv5", instructions=(
        "OpenCV 5 perception tools for a hiker agent. Load a frame, detect candidates, then refer to candidates "
        "by id (never pixel coordinates). All pixel work is OpenCV 5 (DNN YOLOX, dark-channel defog, CLAHE, "
        "Farneback flow)."))

    def png(img: np.ndarray) -> Image:
        return Image(data=T.encode_image(img, "png", max_side=512), format="png")

    def safe(fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            try:
                return fn(*a, **kw)
            except T.ToolError as e:                      # bad arguments: a clean tool error, not a traceback
                raise McpToolError(str(e)) from None
        return wrapped

    @mcp.tool(structured_output=False)
    @safe
    def load_frame(path: str) -> str:
        """Load an image file (under the server's allowed roots) into the frame store; returns its frame_id."""
        fid = tb.load_frame(path)
        h, w = tb.frame(fid).shape[:2]
        return json.dumps({"frame_id": fid, "width": w, "height": h})

    @mcp.tool(structured_output=False)
    @safe
    def detect_candidates(frame_id: str) -> list:
        """YOLOX (OpenCV 5 DNN) + low-confidence pass + dark-blob pass. Returns candidates and the annotated frame."""
        return [json.dumps(tb.detect(frame_id)), png(tb.annotated(frame_id))]

    @mcp.tool(structured_output=False)
    @safe
    def look_closer(frame_id: str, candidate_id: int, zoom: float = 2.5) -> list:
        """Zoom 2-3x on a candidate with dark-channel defogging and CLAHE, then re-detect. Returns report + crop."""
        report, crop = tb.look_closer(frame_id, candidate_id, zoom)
        return [json.dumps(report), png(crop)]

    @mcp.tool(structured_output=False)
    @safe
    def compare_frames(prev_frame_id: str, frame_id: str, candidate_id: int) -> str:
        """Farneback optical flow inside the candidate box, camera ego-motion removed."""
        return json.dumps(tb.compare(prev_frame_id, frame_id, candidate_id))

    @mcp.tool(structured_output=False)
    @safe
    def query_map(summary: dict) -> str:
        """Nearest exit bearing/distance, recent sightings and predator risk near the hiker from a map summary."""
        return json.dumps(tb.query_map(summary))

    mcp.toolbox = tb                                       # for in-process tests
    return mcp


# --- client-side executor --------------------------------------------------------------------------------------
def _decode(result) -> tuple[dict, np.ndarray | None]:
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        text = " ".join(getattr(c, "text", "") for c in result.content)
        raise T.ToolError(text or "tool error")
    data, img = {}, None
    for c in result.content:
        if c.type == "text":
            try:
                data = json.loads(c.text)
            except json.JSONDecodeError:
                data = {"text": c.text}
        elif c.type == "image":
            img = cv2.imdecode(np.frombuffer(base64.b64decode(c.data), np.uint8), cv2.IMREAD_COLOR)
    return data, img


class McpSession:
    """One MCP client session on a background asyncio loop; `call` is synchronous."""

    def __init__(self, server=None, command: list[str] | None = None, timeout_s: float = 60.0):
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        if server is None:
            cmd = command or [sys.executable, "-m", "lost_together.agent.mcp_server"]
            server = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(ROOT), env=dict(os.environ))
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = Client(server, read_timeout_seconds=timeout_s)
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._main = asyncio.run_coroutine_threadsafe(self._hold(), self._loop)
        if not self._ready.wait(timeout_s) or self._error:
            raise RuntimeError(f"MCP session failed to start: {self._error!r}")

    async def _hold(self):
        """Enter and exit the client in one task (anyio cancel scopes require it)."""
        self._stop = asyncio.Event()
        try:
            async with self._client:
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:
            self._error = e
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout_s + 5)

    def call(self, name: str, args: dict) -> tuple[dict, np.ndarray | None]:
        return _decode(self._run(self._client.call_tool(name, args)))

    def list_tools(self) -> list[str]:
        return [t.name for t in self._run(self._client.list_tools()).tools]

    def close(self) -> None:
        try:
            if self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
            self._main.result(10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _cid(args: dict) -> int:
    v = args.get("candidate_id") if isinstance(args, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise T.ToolError("candidate_id must be an integer id from the candidate list")
    try:
        return int(v)
    except ValueError:
        raise T.ToolError("candidate_id must be an integer id from the candidate list") from None


class McpExecutor:
    """The loop's Executor interface over MCP. Frames go to the server by path (it must be under its roots)."""

    def __init__(self, session: McpSession, frame_path: str | Path, map_summary: dict,
                 prev_frame_path: str | Path | None = None):
        self.s, self.summary = session, map_summary
        self.fid = self.s.call("load_frame", {"path": str(Path(frame_path).resolve())})[0]["frame_id"]
        self.prev_fid = (self.s.call("load_frame", {"path": str(Path(prev_frame_path).resolve())})[0]["frame_id"]
                         if prev_frame_path else None)
        self.has_prev = self.prev_fid is not None
        self._obs, self._ann = None, None

    def observe(self) -> dict:
        if self._obs is None:
            self._obs, self._ann = self.s.call("detect_candidates", {"frame_id": self.fid})
        return self._obs

    def annotated(self) -> np.ndarray:
        self.observe()
        return self._ann

    def call(self, name: str, args: dict) -> tuple[dict, np.ndarray | None]:
        if name == "look_closer":
            return self.s.call("look_closer", {"frame_id": self.fid, "candidate_id": _cid(args)})
        if name == "compare_frames":
            if not self.has_prev:
                raise T.ToolError("no previous frame for this point")
            return self.s.call("compare_frames", {"prev_frame_id": self.prev_fid, "frame_id": self.fid,
                                                  "candidate_id": _cid(args)})
        if name == "query_map":
            return self.s.call("query_map", {"summary": self.summary})
        raise T.ToolError(f"unknown tool {name!r}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="OpenCV 5 MCP server (stdio) for the Lost Together hiker agent.")
    ap.add_argument("--root", action="append", help="allowed frame directory (repeatable); default data/ and results/")
    ap.add_argument("--no-yolox", action="store_true")
    a = ap.parse_args(argv)
    roots = a.root or [r for r in os.environ.get("LT_FRAME_ROOTS", "").split(os.pathsep) if r] or None
    print(f"lost-together-opencv5 MCP server: OpenCV {cv2.__version__}, detector {T.detector_status()}",
          file=sys.stderr)
    build_server(roots=roots, use_yolox=not a.no_yolox).run("stdio")


if __name__ == "__main__":
    main()
