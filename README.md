# Ultrafnd_try
It is a Multimodal fake news detection system project

## Data Audit

Use the provided script to inspect the FakeSV dataset before training. It reports class balance, source distribution, text lengths, and image metadata statistics.

```bash
python scripts/data_audit.py --data-root /path/to/fakesv \
    --image-root /path/to/fakesv/images \
    --embedding-analysis
```

The optional `--embedding-analysis` flag generates a t-SNE plot of SBERT embeddings.
