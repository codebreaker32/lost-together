"""Agent layer, fully offline: stub model, a local fake OpenAI server, synthetic frames (dark blob in noise)."""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from lost_together.agent import bench, loop, synth, tools as T
from lost_together.agent.llm import CacheMiss, LLMClient, LLMConfig, request_key
from lost_together.agent.loop import AgentConfig, LocalExecutor, Point, agent_decide, parse_decision, rule_decide
from lost_together.agent.ranger import Autonomy, CLIApprover, Ranger, ScriptedApprover
from lost_together.agent.stub import StubModel
from lost_together.vision.detector import MODEL


def blob_frame(seed=0, shift=0, fog=0.0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 18, (240, 320, 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.ellipse(img, (150 + shift, 130), (28, 18), 0, 0, 360, (25, 25, 30), -1, cv2.LINE_AA)
    if fog:
        img = synth.fog(img, fog)
    return img


SUMMARY = {"t": 100, "cell_m": 5.0, "step_s": 4.0, "hiker": {"id": 0, "pos": [40, 40]},
           "group": [{"id": 1, "pos": [43, 44]}], "exits": [[0, 40], [79, 10]],
           "sightings": [{"pos": [44, 40], "conf": 0.6, "t": 90, "hiker": 1}]}


def make_point(pid="p0", moving=True, truth=None, summary=SUMMARY):
    return Point(pid, blob_frame(shift=8 if moving else 0), summary, prev_frame=blob_frame(shift=0),
                 truth=truth or {"predator_real": moving, "should_escape": moving})


def stub_client(**kw):
    return LLMClient(LLMConfig(mode="stub", cache_dir=None), stub=StubModel(**kw))


# --- tools ---------------------------------------------------------------------------------------------------
def test_tools_find_look_compare_and_query():
    tb = T.Toolbox(use_yolox=False)
    fid = tb.add_frame(blob_frame(shift=8), "cur")
    pid = tb.add_frame(blob_frame(), "prev")
    det = tb.detect(fid)
    assert det["candidates"], "dark blob should become a candidate"
    c = det["candidates"][0]
    x, y, w, h = c["box"]
    assert c["cls"] == "unknown" and c["score"] <= 0.5 and x < 158 < x + w and y < 130 < y + h
    report, crop = tb.look_closer(fid, c["id"])
    assert 2.0 <= report["zoom"] <= 3.0 and crop.shape[0] >= 2 * report["crop_box"][3] - 2
    assert report["after"]["score"] > c["score"]               # enhancement makes the shape clearer
    assert tb.compare(pid, fid, c["id"])["moving"] is True
    assert tb.compare(pid, pid, c["id"])["moving"] is False     # identical frames: no motion
    q = tb.query_map(SUMMARY)
    assert q["nearest_exit"]["distance_m"] == 200.0 and q["nearest_exit"]["bearing_deg"] == 0.0
    assert q["risk_near_hiker"] == pytest.approx(0.6 * 0.5 ** (10 / 60), abs=1e-3)
    with pytest.raises(T.ToolError):
        tb.look_closer(fid, 42)


def test_defog_restores_contrast():
    clear = blob_frame()
    foggy = synth.fog(clear, 0.05)
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std()    # noqa: E731
    assert T.visibility(foggy)["haze"] > T.visibility(clear)["haze"]
    assert g(T.defog(foggy)) > 1.5 * g(foggy)


def test_frame_store_rejects_paths_outside_roots(tmp_path):
    tb = T.Toolbox(use_yolox=False, allowed_roots=[tmp_path / "frames"], max_frames=2)
    (tmp_path / "frames").mkdir()
    cv2.imwrite(str(tmp_path / "frames" / "a.png"), blob_frame())
    cv2.imwrite(str(tmp_path / "b.png"), blob_frame())
    assert tb.load_frame(tmp_path / "frames" / "a.png") == "a"
    with pytest.raises(T.ToolError):
        tb.load_frame(tmp_path / "frames" / ".." / "b.png")
    for k in range(3):
        tb.add_frame(blob_frame(), f"f{k}")
    assert len(tb.frames) == 2                                   # bounded store


@pytest.mark.skipif(not MODEL.exists(), reason="YOLOX model not downloaded")
def test_yolox_low_confidence_pass_runs_on_synthetic_frame():
    det = T.detect_candidates(blob_frame(), use_yolox=True)
    assert det["detector"] == "yolox" and any(c["source"] == "blob" for c in det["candidates"])


# --- validation ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,ok", [
    ('{"decision": "escape", "candidate_id": 0, "why": "moving dark shape"}', True),
    ('```json\n{"decision": "regroup", "candidate_id": null, "why": "unclear"}\n```', True),
    ('{"decision": "run", "candidate_id": 0, "why": "x"}', False),
    ('{"decision": "escape", "candidate_id": 7, "why": "x"}', False),
    ('{"decision": "escape", "candidate_id": 0, "why": ""}', False),
    ('{"decision": "escape", "candidate_id": 0, "why": "x", "coords": [1, 2]}', False),
    ('I think we should escape', False),
    (None, False),
])
def test_parse_decision_is_strict(text, ok):
    parsed, err = parse_decision(text, {0, 1})
    assert (parsed is not None) == ok and (err is None) == ok


# --- loop ----------------------------------------------------------------------------------------------------
def test_agent_loop_end_to_end_with_stub():
    p = make_point()
    tb = T.Toolbox(use_yolox=False)
    d = agent_decide(p, stub_client(), LocalExecutor(tb, p))
    assert d.source == "llm" and d.decision == "escape" and d.candidate_id == 0
    kinds = [s["kind"] for s in d.trace]
    assert kinds[0] == "observe" and kinds[-1] == "validate" and "tool" in kinds
    used = [s["name"] for s in d.trace if s["kind"] == "tool"]
    assert used == ["look_closer", "query_map", "compare_frames"]
    assert d.llm_calls == 3 and d.tool_calls == 3
    assert "look_closer" in d.story() and d.story().endswith("ESCAPE")
    # a static shape (rock) with the same map is not an escape
    rock = make_point("rock", moving=False)
    d2 = agent_decide(rock, stub_client(), LocalExecutor(tb, rock))
    assert d2.decision != "escape"


def test_rule_policy_uses_same_tools_and_reports_evidence():
    p = make_point()
    d = rule_decide(LocalExecutor(T.Toolbox(use_yolox=False), p), p.point_id)
    assert d.source == "rule" and d.decision in loop.DECISIONS and d.evidence is not None
    assert [s["name"] for s in d.trace if s["kind"] == "tool"][0] == "query_map"


@pytest.mark.parametrize("fault,reason", [
    ("invalid_json", "invalid output"),
    ("prose", "invalid output"),
    ("bad_candidate", "invalid output"),
    ("timeout", "model call failed: LLMTimeout"),
    ("error", "model call failed: LLMError"),
])
def test_failures_fall_back_to_rule(fault, reason):
    p = make_point()
    d = agent_decide(p, stub_client(fault=fault), LocalExecutor(T.Toolbox(use_yolox=False), p))
    assert d.source == "fallback" and d.fallback_reason.startswith(reason)
    assert d.decision in loop.DECISIONS and any(s["kind"] == "fallback" for s in d.trace)


def test_unknown_tool_is_reported_to_model_not_raised():
    p = make_point()
    d = agent_decide(p, stub_client(fault="bad_tool"), LocalExecutor(T.Toolbox(use_yolox=False), p))
    bad = [s for s in d.trace if s["kind"] == "tool" and s["name"] == "rm_rf"]
    assert bad and bad[0]["ok"] is False and "unknown tool" in bad[0]["result"]["error"]
    assert d.source == "llm"


def test_tool_budget_is_enforced():
    p = make_point()
    d = agent_decide(p, stub_client(fault="endless_tools"), LocalExecutor(T.Toolbox(use_yolox=False), p),
                     AgentConfig(max_tool_turns=3))
    assert sum(1 for s in d.trace if s["kind"] == "llm" and s["tool_calls"]) == 3
    assert d.source == "llm" and d.llm_calls == 4                # the 4th call had no tools offered


def test_policies_never_see_truth_and_images_follow_provider_rule():
    seen = []

    class Spy(StubModel):
        def __call__(self, messages, tools):
            seen.append(json.loads(json.dumps(messages)))
            return super().__call__(messages, tools)

    p = make_point(truth={"predator_real": True, "should_escape": True, "secret": "TRUTH-MARKER"})
    client = LLMClient(LLMConfig(mode="stub", cache_dir=None, image_in_tool=False), stub=Spy())
    agent_decide(p, client, LocalExecutor(T.Toolbox(use_yolox=False), p))
    blob = json.dumps(seen)
    assert "TRUTH-MARKER" not in blob and "should_escape" not in blob
    last = seen[-1]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert tool_msgs and all(isinstance(m["content"], str) for m in tool_msgs)     # Groq: text-only tool results
    follow = [m for m in last if m["role"] == "user" and isinstance(m["content"], list)
              and any(part.get("type") == "image_url" for part in m["content"]) and "tools above" in m["content"][0]["text"]]
    assert follow, "tool images must arrive in a follow-up user message"
    for i, m in enumerate(last):                                   # tool results directly follow their call
        if m.get("tool_calls"):
            assert all(last[i + 1 + k]["role"] == "tool" for k in range(len(m["tool_calls"])))


# --- llm client: live (fake server), cache, replay, retries -----------------------------------------------------
class FakeOpenAI(BaseHTTPRequestHandler):
    script: list = []
    bodies: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.bodies.append(body)
        status, payload = FakeOpenAI.script.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    FakeOpenAI.script, FakeOpenAI.bodies = [], []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _ok(message, usage=(900, 40)):
    return 200, {"choices": [{"message": message}], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def test_live_mode_records_cache_then_replays_offline(fake_server, tmp_path):
    tool_call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "look_closer", "arguments": '{"candidate_id": 0}'}}]}
    final = {"role": "assistant", "content": '{"decision": "escape", "candidate_id": 0, "why": "dark shape moving"}'}
    FakeOpenAI.script = [(503, {"error": "busy"}), _ok(tool_call), _ok(final)]
    cfg = LLMConfig(mode="live", base_url=fake_server, cache_dir=tmp_path, backoff_s=0.01, retries=2)
    p = make_point()
    d = agent_decide(p, LLMClient(cfg), LocalExecutor(T.Toolbox(use_yolox=False), p))
    assert d.source == "llm" and d.decision == "escape" and d.tokens_in == 1800 and d.cost_usd > 0
    assert [s.get("attempts") for s in d.trace if s["kind"] == "llm"] == [2, 1]       # 503 was retried
    b0 = FakeOpenAI.bodies[0]
    assert b0["model"] == "qwen3-vl:30b-a3b-instruct" and b0["reasoning_effort"] == "none" and b0["tools"]
    assert any(part.get("type") == "image_url" for part in b0["messages"][1]["content"])
    tool_msg = [m for m in FakeOpenAI.bodies[-1]["messages"] if m["role"] == "tool"][0]
    assert isinstance(tool_msg["content"], list)                                      # Ollama: image inside tool msg
    # replay: same requests, no server needed, recorded latency reported
    FakeOpenAI.script = []
    replay = LLMClient(LLMConfig(mode="replay", base_url="http://127.0.0.1:9/v1", cache_dir=tmp_path))
    d2 = agent_decide(p, replay, LocalExecutor(T.Toolbox(use_yolox=False), p))
    assert d2.decision == "escape" and d2.source == "llm" and all(s["cached"] for s in d2.trace if s["kind"] == "llm")
    assert d2.model_latency_s == pytest.approx(d.model_latency_s)
    # replay miss (different point) -> rule fallback, recorded
    other = make_point("p1", summary={**SUMMARY, "t": 101})
    d3 = agent_decide(other, replay, LocalExecutor(T.Toolbox(use_yolox=False), other))
    assert d3.source == "fallback" and "CacheMiss" in d3.fallback_reason


def test_client_errors_and_cache_keys(fake_server, tmp_path):
    FakeOpenAI.script = [(400, {"error": "bad"})]
    c = LLMClient(LLMConfig(mode="live", base_url=fake_server, cache_dir=None, backoff_s=0.0))
    with pytest.raises(Exception) as e:
        c.chat([{"role": "user", "content": "hi"}])
    assert "HTTP 400" in str(e.value) and len(FakeOpenAI.bodies) == 1     # 4xx is not retried
    with pytest.raises(CacheMiss):
        LLMClient(LLMConfig(mode="replay", cache_dir=tmp_path)).chat([{"role": "user", "content": "hi"}])
    assert request_key({"a": 1}, 0) != request_key({"a": 1}, 1)                     # k-repeat samples differ


# --- ranger ---------------------------------------------------------------------------------------------------
def test_ranger_advise_gates_escape_and_records_override(tmp_path):
    p = make_point()
    d = agent_decide(p, stub_client(), LocalExecutor(T.Toolbox(use_yolox=False), p))
    assert d.decision == "escape"
    log = tmp_path / "ranger.jsonl"
    Ranger(Autonomy.ADVISE, ScriptedApprover({p.point_id: "regroup"}), log_path=log).review(d)
    assert d.action == "regroup" and d.approval["approved"] is False
    assert d.trace[-1]["kind"] == "ranger" and "overridden" in d.trace[-1]["outcome"]
    assert json.loads(log.read_text().splitlines()[0])["action"] == "regroup"
    d2 = agent_decide(p, stub_client(), LocalExecutor(T.Toolbox(use_yolox=False), p))
    Ranger(Autonomy.ACT).review(d2)
    assert d2.action == "escape" and d2.approval is None


def test_cli_approver_reads_console():
    p = make_point()
    d = agent_decide(p, stub_client(), LocalExecutor(T.Toolbox(use_yolox=False), p))
    out = io.StringIO()
    Ranger(Autonomy.ADVISE, CLIApprover(stdin=io.StringIO("a\n"), stdout=out)).review(d)
    assert d.action == "escape" and d.approval["approved"] and "RANGER CONSOLE" in out.getvalue()
    d.trace.pop()
    Ranger(Autonomy.ADVISE, CLIApprover(stdin=io.StringIO("keep_going\n"), stdout=io.StringIO())).review(d)
    assert d.action == "keep_going" and not d.approval["approved"]


# --- bench ---------------------------------------------------------------------------------------------------
def test_bench_end_to_end_stub(tmp_path):
    pts_path = synth.make_dataset(tmp_path / "synth", n=6, seed=1)
    pts = loop.load_points(pts_path)
    report = bench.run(pts, LLMConfig(mode="stub", cache_dir=None),
                       bench.BenchConfig(name="t", out_dir=tmp_path / "out", use_yolox=False, ablation="scalar", repeat=2))
    assert report["n"] == 6 and set(report["policies"]) == {"rule", "llm", "llm_scalar"}
    for m in report["policies"].values():
        assert 0 <= m["accuracy"] <= 1 and m["false_alarms"] + m["late_escapes"] <= 6
        assert m["latency_p50_s"] is not None and m["latency_p95_s"] >= m["latency_p50_s"]
    mc = report["mcnemar_rule_vs_llm"]
    assert 0 <= mc["p_exact"] <= 1 and report["repeat"]["all_agree_rate"] == 1.0 and "warning" in report
    out = tmp_path / "out" / "t"
    assert (out / "report.json").exists() and (out / "rule_vs_llm.png").exists()
    rows = [json.loads(x) for x in (out / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 18 and all(r["story"] for r in rows)
    assert len(list((out / "traces").glob("*.json"))) == 18


def test_mcnemar_counts_discordant_pairs():
    mk = lambda pid, dec: loop.Decision(pid, "x", dec, None, "", "rule")             # noqa: E731
    truths = {f"p{i}": {"should_escape": i < 5} for i in range(10)}
    rule = [mk(f"p{i}", "keep_going") for i in range(10)]
    llm = [mk(f"p{i}", "escape" if i < 5 else "keep_going") for i in range(10)]
    r = bench.mcnemar(rule, llm, truths)
    assert r["rule_wrong_llm_right"] == 5 and r["rule_right_llm_wrong"] == 0
    assert r["p_exact"] == pytest.approx(0.0625)


# --- MCP ------------------------------------------------------------------------------------------------------
def test_mcp_server_serves_tools_and_drives_the_agent(tmp_path):
    from lost_together.agent.mcp_server import McpExecutor, McpSession, build_server
    cv2.imwrite(str(tmp_path / "cur.png"), blob_frame(shift=8))
    cv2.imwrite(str(tmp_path / "prev.png"), blob_frame())
    srv = build_server(roots=[tmp_path], use_yolox=False)
    with McpSession(srv, timeout_s=30) as s:
        assert set(s.list_tools()) == {"load_frame", "detect_candidates", "look_closer", "compare_frames", "query_map"}
        ex = McpExecutor(s, tmp_path / "cur.png", SUMMARY, tmp_path / "prev.png")
        assert ex.observe()["candidates"] and ex.annotated().shape == (240, 320, 3)   # image content decoded
        report, crop = ex.call("look_closer", {"candidate_id": 0})
        assert crop is not None and report["after"]["score"] > 0
        with pytest.raises(T.ToolError, match="allowed frame roots"):
            s.call("load_frame", {"path": "/etc/hostname.png"})
        p = Point("mcp", blob_frame(shift=8), SUMMARY, prev_frame=blob_frame())
        d = agent_decide(p, stub_client(), ex)
        assert d.source == "llm" and d.decision == "escape"


# --- review fixes ---------------------------------------------------------------------------------------------
def test_rule_uses_motion_like_the_agent():
    """Fairness (brief §7): the rule spends the same look_closer + compare_frames actions as the agent."""
    tb = T.Toolbox(use_yolox=False)
    p = make_point("mv")
    ex = LocalExecutor(tb, p)
    top = ex.observe()["candidates"][0]
    tb.set_candidates(ex.fid, [{**top, "score": 0.4}])          # inside the relook band
    d = rule_decide(ex, p.point_id)
    assert [s["name"] for s in d.trace if s["kind"] == "tool"] == ["query_map", "look_closer", "compare_frames"]
    assert "moving" in d.why


def test_tool_crash_is_reported_and_rule_failure_still_decides():
    class Crashy(LocalExecutor):
        def call(self, name, args):
            if name == "look_closer":
                raise cv2.error("simulated OpenCV failure")
            return super().call(name, args)

    p = make_point()
    d = agent_decide(p, stub_client(), Crashy(T.Toolbox(use_yolox=False), p))
    bad = [s for s in d.trace if s["kind"] == "tool" and s["name"] == "look_closer"]
    assert bad and bad[0]["ok"] is False and "error" in bad[0]["result"]

    class Dead(LocalExecutor):
        def call(self, name, args):
            raise RuntimeError("tools down")

    d2 = agent_decide(p, stub_client(fault="error"), Dead(T.Toolbox(use_yolox=False), p))
    assert d2.source == "fallback" and d2.decision == "regroup" and "rule failed" in d2.fallback_reason


def test_frame_id_reuse_and_given_candidates_are_safe():
    tb = T.Toolbox(use_yolox=False)
    fid = tb.add_frame(blob_frame(shift=0), "x")
    first = tb.candidates(fid)[0]["box"]
    tb.add_frame(blob_frame(shift=60), "x")                     # same id, different image
    assert tb.candidates(fid)[0]["box"] != first
    tb.set_candidates(fid, [{"id": 0, "cls": "bear", "score": 0.4, "box": [300, 200, 80, 80]}])
    assert tb.candidates(fid)[0]["box"] == [300, 200, 20, 40]   # clipped to the 320x240 frame
    with pytest.raises(T.ToolError):
        tb.set_candidates(fid, [{"id": 0, "box": [500, 500, 10, 10]}])
    with pytest.raises(T.ToolError, match="malformed map summary"):
        tb.query_map({"hiker": {"pos": "nowhere"}})


def test_connection_reset_is_an_llm_error(tmp_path):
    from lost_together.agent.llm import LLMError

    def opener(req, timeout):
        raise ConnectionResetError("peer reset")

    c = LLMClient(LLMConfig(mode="live", cache_dir=None, retries=1, backoff_s=0.0), opener=opener)
    with pytest.raises(LLMError, match="connection failed"):
        c.chat([{"role": "user", "content": "hi"}])
