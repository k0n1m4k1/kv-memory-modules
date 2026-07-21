# Quick compression benchmark over real .kmd modules: does gzip-on-disk pay?
# Measures ratio and throughput for zlib levels 1/6, plus a byte-shuffle
# transform (split f16 stream into high/low byte planes) that typically helps
# float data, and lzma preset 1 on a slice as an upper-bound probe.
import lzma
import sys
import time
import zlib
from pathlib import Path

# Modules to benchmark: pass .kmd paths as CLI args, or default to every
# module under the repo's kmd/ directory (compile some first with mdc.py).
ROOT = Path(__file__).resolve().parents[2]
paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "kmd").glob("*.kmd"))
FILES = [(str(p), p.name) for p in paths]


def shuffle_bytes(b: bytes) -> bytes:
    # Byte-plane split for 16-bit data: all low bytes, then all high bytes.
    return b[0::2] + b[1::2]


def bench(tag, data, comp, decomp):
    t0 = time.perf_counter()
    c = comp(data)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = decomp(c)
    td = time.perf_counter() - t0
    assert len(d) == len(data)
    mb = len(data) / 1e6
    print(f"  {tag:24s} ratio {len(c)/len(data):5.1%}  comp {mb/tc:6.0f} MB/s  decomp {mb/td:6.0f} MB/s")


for path, desc in FILES:
    data = open(path, "rb").read()
    print(f"== {desc}: {len(data)/1e6:.0f} MB")
    bench("zlib-1", data, lambda b: zlib.compress(b, 1), zlib.decompress)
    bench("zlib-6 (gzip default)", data, lambda b: zlib.compress(b, 6), zlib.decompress)
    sh = shuffle_bytes(data)
    bench("shuffle+zlib-1", sh, lambda b: zlib.compress(b, 1), zlib.decompress)
    # lzma is slow: probe a 32 MB slice only, as a "best lossless case" bound.
    sl = data[: 32_000_000]
    bench("lzma-1 (32MB slice)", sl, lambda b: lzma.compress(b, preset=1), lzma.decompress)
