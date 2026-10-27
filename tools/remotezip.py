"""Extract selected members from a remote zip via HTTP Range requests (stdlib only).

usage: python3 -I remotezip.py URL OUTDIR [--list] [--match REGEX] [--max N]
"""
import sys, os, re, io, zipfile, urllib.request, argparse

UA = {"User-Agent": "remotezip/0.1"}
STATS = {"bytes": 0, "reqs": 0}


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        STATS["bytes"] += len(data)
        STATS["reqs"] += 1
        return data

    def _block(self, i):
        if i not in self.cache:
            s = i * self.block
            e = min(self.size, s + self.block) - 1
            self.cache[i] = self._fetch(s, e)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            b = self._block(i)[o:o + n]
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--match", default=None)
    ap.add_argument("--max", type=int, default=10 ** 9)
    ap.add_argument("--block", type=int, default=1 << 18)
    a = ap.parse_args()
    f = HTTPRangeFile(a.url, block=a.block)
    z = zipfile.ZipFile(f)
    infos = z.infolist()
    print(f"remote size {f.size/1e9:.3f} GB, {len(infos)} members")
    if a.list:
        for zi in infos[:20]:
            print(f"  {zi.filename} {zi.file_size} comp={zi.compress_type}")
        if len(infos) > 20:
            print(f"  ... last: {infos[-1].filename}")
    if a.match:
        rx = re.compile(a.match)
        sel = [zi for zi in infos if rx.search(zi.filename)][: a.max]
        os.makedirs(a.outdir, exist_ok=True)
        for zi in sel:
            name = os.path.basename(zi.filename)  # flatten, never trust paths
            if not name:
                continue
            with open(os.path.join(a.outdir, name), "wb") as o:
                o.write(z.read(zi))
            print("  extracted", zi.filename)
    print(f"transferred {STATS['bytes']/1e6:.2f} MB in {STATS['reqs']} range requests")


if __name__ == "__main__":
    main()
