# Compact console summary of every experiment JSON under results/.
#
# Knows the three current result schemas and falls back to listing top-level
# scalars for older/unknown files:
#   - "regimenes" -> e12.py   (setup cost per compute regime)
#   - "mds"       -> bateria7 (per-module recall: joint / linked / nomem)
#   - "condiciones"/anything else -> generic key dump
#
# Usage: python scripts/show-results.py [--tag <tag>]

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def fmt_e12(r):
    # Older result JSONs (pre-rename) use the "kvm_MB" key; accept both.
    mb = r.get("kmd_MB", r.get("kvm_MB"))
    lines = [f"  {r['model']} | module {mb} MB, {r['M']} tokens"]
    for ngl, d in r["regimenes"].items():
        lines.append(
            f"  ngl={ngl:>3}: prefill {d['t_prefill_s']:>7}s ({d['prefill_tps']} t/s)"
            f" {d['prefill']['score']}/{r['n_q']} | restore {d['t_restore_s']}s"
            f" {d['restore']['score']}/{r['n_q']} | x{d['ratio']}"
        )
    return lines


def fmt_bateria7(r):
    lines = [f"  {r['model']} | adversarial prefix {r['P']} tokens"]
    for slug, m in r["mds"].items():
        lines.append(
            f"  {slug}: joint {m['joint']['score']}/{m['n_q']} ({m['t_setup_joint_s']}s)"
            f" | linked {m['linked']['score']}/{m['n_q']} ({m['t_setup_linked_s']}s)"
            f" | nomem {m['nomem']['score']}/{m['n_q']}"
        )
    return lines


def fmt_generic(r):
    scalars = {k: v for k, v in r.items() if isinstance(v, (str, int, float))}
    return ["  " + ", ".join(f"{k}={v}" for k, v in list(scalars.items())[:8])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="only files whose name contains this tag")
    args = ap.parse_args()

    for dirpath, _dirs, files in sorted(os.walk(RESULTS)):
        if os.path.basename(dirpath) == "logs":
            continue
        for fn in sorted(files):
            if not fn.endswith(".json") or (args.tag and args.tag not in fn):
                continue
            path = os.path.join(dirpath, fn)
            try:
                r = json.load(open(path, encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"{fn}: unreadable")
                continue
            print(os.path.relpath(path, RESULTS))
            if isinstance(r, dict) and "regimenes" in r:
                lines = fmt_e12(r)
            elif isinstance(r, dict) and "mds" in r:
                lines = fmt_bateria7(r)
            elif isinstance(r, dict):
                lines = fmt_generic(r)
            else:
                lines = ["  (list payload)"]
            print("\n".join(lines))


if __name__ == "__main__":
    main()
