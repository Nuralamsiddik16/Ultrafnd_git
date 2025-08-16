# scripts/generate_ocr_phrase_features.py
"""
Generate two artifacts:
1) preprocess_ocr/sam/                  (placeholder mask per video id; swap with SAM later)
2) fakesv/preprocess_ocr/ocr_phrase_fea.pkl  (per-video phrase sets & frequencies)

Usage:
python scripts/generate_ocr_phrase_features.py \
  --data_path /Volumes/SR_disk/FakeSV/data_complete.json \
  --out_root /Users/siddikmdnuralam/Ultrafnd5
"""
import os, json, argparse, hashlib, pickle, pathlib, re
from collections import Counter

def clean_tokens(text):
    # Chinese chars + word chars, min length 2
    return [t for t in re.findall(r"[\w\u4e00-\u9fa5]+", text or "") if len(t) >= 2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="JSONL file (data_complete.json)")
    ap.add_argument("--out_root", required=True, help="Repo root (where to create preprocess_ocr/ and fakesv/)")
    args = ap.parse_args()

    out_sam = pathlib.Path(args.out_root) / "preprocess_ocr" / "sam"
    out_pkl_dir = pathlib.Path(args.out_root) / "fakesv" / "preprocess_ocr"
    out_sam.mkdir(parents=True, exist_ok=True)
    out_pkl_dir.mkdir(parents=True, exist_ok=True)

    phrase_sets = {}
    freqs = {}

    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            vid = str(o.get("video_id"))
            ocr = o.get("ocr", "")
            toks = clean_tokens(ocr)
            phrase_sets[vid] = set(toks)
            freqs[vid] = Counter(toks)

            # placeholder "mask": store a stable hash of the token set
            h = hashlib.md5((" ".join(sorted(toks))).encode("utf-8")).hexdigest()
            with open(out_sam / f"{vid}.mask.txt", "w", encoding="utf-8") as mf:
                mf.write(h)

    out_pkl = out_pkl_dir / "ocr_phrase_fea.pkl"
    with open(out_pkl, "wb") as pf:
        pickle.dump({"phrase_sets": phrase_sets, "freqs": {k: dict(v) for k, v in freqs.items()}}, pf)

    print("Wrote:")
    print(" -", out_sam)
    print(" -", out_pkl)

if __name__ == "__main__":
    main()
