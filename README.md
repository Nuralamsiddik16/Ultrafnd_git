# Ultrafnd_try
It is a Multimodal fake news detection system project

## Mini dataset extraction

Use `scripts/mini_dataset.py` to create a smaller version of the FakeSV dataset
for quick experiments or testing.

```bash
python scripts/mini_dataset.py --input-dir /path/to/FakeSV --output-dir ./mini_fake --num 50
```

The script copies the first `N` entries from `data_complete.json` and any
matching files in `video_comment/` and `videos/` (if present) into the specified
output directory.
