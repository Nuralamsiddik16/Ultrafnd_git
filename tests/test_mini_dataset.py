import json
from pathlib import Path
import tempfile
import pytest
import importlib.util


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "mini_dataset.py"
    if not script_path.exists():
        pytest.skip("mini_dataset script not available")
    spec = importlib.util.spec_from_file_location("mini_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module, src, dst, num):
    combos = [
        ("--src", "--dest"),
        ("--src", "--out"),
        ("--src", "--output"),
        ("--source", "--dest"),
        ("--source", "--out"),
        ("--root", "--dest"),
        ("--root", "--out"),
        ("--data_root", "--out_dir"),
        ("--data-root", "--out-dir"),
        ("--input", "--output"),
    ]
    for sflag, oflag in combos:
        args = [sflag, str(src), oflag, str(dst), "--num", str(num)]
        try:
            module.main(args)
            return
        except SystemExit as e:
            if e.code == 0:
                return
            continue
        except Exception:
            continue
    pytest.fail("Unable to execute mini_dataset.main with known arguments")


def test_mini_dataset(tmp_path):
    module = _load_module()
    src = tmp_path / "src"
    out_dir = tmp_path / "out"
    comments_dir = src / "video_comment"
    videos_dir = src / "videos"
    comments_dir.mkdir(parents=True)
    videos_dir.mkdir(parents=True)

    records = []
    for i in range(3):
        vid = f"vid{i}"
        records.append({
            "video_id": vid,
            "title": f"title{i}",
            "ocr": f"ocr{i}",
            "comments": [f"c{i}"],
            "annotation": "假",
        })
        (comments_dir / f"{vid}.json").write_text(json.dumps([f"comment{i}"]))
        (videos_dir / f"{vid}.mp4").write_text("video")

    (src / "data_complete.json").write_text(json.dumps(records, ensure_ascii=False))

    _run_main(module, src, out_dir, 2)

    out_json_path = out_dir / "data_complete.json"
    assert out_json_path.exists()
    out_data = json.loads(out_json_path.read_text())
    assert len(out_data) == 2
    source_ids = {r["video_id"] for r in records}
    out_ids = {r.get("video_id") or r.get("id") for r in out_data}
    assert out_ids.issubset(source_ids)
    for vid in out_ids:
        assert (out_dir / "video_comment" / f"{vid}.json").exists()
        assert (out_dir / "videos" / f"{vid}.mp4").exists()
