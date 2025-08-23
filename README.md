# Ultrafnd_try
It is a Multimodal fake news detection system project

## Training

Use `run_train_eval.py` for end-to-end training and evaluation. To help with
class imbalance you can enable focal loss via the `--focal_gamma` flag, e.g.:

```
python run_train_eval.py --focal_gamma 2.0
```

Setting `--focal_gamma` to a value greater than zero applies focal loss with the
specified gamma; the default `0.0` keeps standard cross-entropy.
