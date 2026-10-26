# NeuroDetect evaluation 20261026T141858648484Z

Dataset: 1311 labelled images. Model: brain_tumor_multiclass.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| glioma | 0.9748 | 0.7733 | 0.8625 | 300 |
| meningioma | 0.7902 | 0.7876 | 0.7889 | 306 |
| notumor | 0.9199 | 0.9926 | 0.9549 | 405 |
| pituitary | 0.8852 | 0.9767 | 0.9287 | 300 |

Macro-F1: 0.8837; weighted-F1: 0.8890; accuracy: 0.8909.

Confusion matrix: rows=true, columns=predicted.

| Actual / predicted | glioma | meningioma | notumor | pituitary |
|---|---:|---:|---:|---:|
| glioma | 232 | 58 | 3 | 7 |
| meningioma | 5 | 241 | 29 | 31 |
| notumor | 0 | 3 | 402 | 0 |
| pituitary | 1 | 3 | 3 | 293 |

These are dataset-specific results, not clinical performance estimates. The 11 unlabelled user images are excluded.

Comparison with previous evaluation:
```json
{
  "previous_run": "20261026T135651977829Z",
  "previous_model": {
    "version": "brain_tumor_multiclass",
    "sha256": "13163374b4ea2401363a95ddaca153fdf863a6ae7936df8d6c1a25d92a385baf",
    "source": "local weights"
  },
  "delta": {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "weighted_f1": 0.0
  },
  "per_class_f1_delta": {
    "glioma": 0.0,
    "meningioma": 0.0,
    "notumor": 0.0,
    "pituitary": 0.0
  }
}
```
