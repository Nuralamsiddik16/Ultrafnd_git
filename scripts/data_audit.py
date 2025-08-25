import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from src.data_pipeline.fakesv_dataset import FakeSVRawDataset


def analyze_class_balance(dataset: FakeSVRawDataset) -> Counter:
    """Print and return class distribution of the dataset."""
    counts = Counter(dataset.labels)
    total = sum(counts.values())
    print("Class distribution:")
    for label, count in counts.items():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {label}: {count} ({pct:.2f}%)")
    return counts


def analyze_sources(dataset: FakeSVRawDataset) -> Counter:
    """Analyze distribution of 'source' field if present."""
    sources = [r.get("source") for r in dataset.records if r.get("source")]
    if not sources:
        print("No source information found in records.")
        return Counter()
    counts = Counter(sources)
    print("Top sources:")
    for src, cnt in counts.most_common(10):
        print(f"  {src}: {cnt}")
    return counts


def analyze_text_length(dataset: FakeSVRawDataset) -> Dict[str, float]:
    """Compute average title word count for real vs. fake samples."""
    lengths = {0: [], 1: []}
    for rec, label in zip(dataset.records, dataset.labels):
        wc = len((rec.get("title") or "").split())
        lengths[int(label)].append(wc)
    avg_real = float(np.mean(lengths[0])) if lengths[0] else 0.0
    avg_fake = float(np.mean(lengths[1])) if lengths[1] else 0.0
    print(f"Average title length (real): {avg_real:.2f} words")
    print(f"Average title length (fake): {avg_fake:.2f} words")
    return {"real": avg_real, "fake": avg_fake}


def analyze_images(dataset: FakeSVRawDataset, image_root: Path) -> Tuple[List[Tuple[int, int]], Counter]:
    """Compute average image dimensions and format distribution."""
    from PIL import Image

    dims: List[Tuple[int, int]] = []
    formats: Counter = Counter()
    for rec in dataset.records:
        img_path = None
        for key in ("image", "image_path", "img_path", "img"):
            if key in rec and rec[key]:
                img_path = rec[key]
                break
        if not img_path:
            continue
        path = image_root / img_path
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                dims.append(im.size)  # (width, height)
                if im.format:
                    formats[im.format.lower()] += 1
        except Exception:
            continue
    if dims:
        arr = np.array(dims)
        avg_w, avg_h = arr.mean(axis=0)
        print(f"Average image size: {avg_w:.1f}x{avg_h:.1f}")
    else:
        print("No images found for analysis.")
    if formats:
        print("Image format distribution:")
        for fmt, cnt in formats.items():
            print(f"  {fmt}: {cnt}")
    return dims, formats


def embedding_analysis(dataset: FakeSVRawDataset, out_dir: Path) -> None:
    """Run SBERT + t-SNE embedding visualization if dependencies are available."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"Embedding analysis skipped: {e}")
        return

    titles = [rec.get("title") or "" for rec in dataset.records]
    labels = dataset.labels

    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(titles, show_progress_bar=False)

    tsne = TSNE(n_components=2, init='random', learning_rate='auto')
    coords = tsne.fit_transform(emb)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='coolwarm', alpha=0.7, s=10)
    plt.title('t-SNE of SBERT embeddings')
    plt.savefig(out_dir / 'sbert_tsne.png', dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="FakeSV data auditing utilities")
    parser.add_argument('--data-root', type=str, required=True, help='Path to FakeSV data directory')
    parser.add_argument('--image-root', type=str, default=None, help='Optional root dir for images referenced in records')
    parser.add_argument('--output-dir', type=str, default='audit_outputs', help='Directory to save plots or artifacts')
    parser.add_argument('--embedding-analysis', action='store_true', help='Run SBERT + t-SNE analysis')
    args = parser.parse_args()

    dataset = FakeSVRawDataset(args.data_root)

    analyze_class_balance(dataset)
    analyze_sources(dataset)
    analyze_text_length(dataset)
    if args.image_root:
        analyze_images(dataset, Path(args.image_root))
    if args.embedding_analysis:
        embedding_analysis(dataset, Path(args.output_dir))


if __name__ == '__main__':
    main()
