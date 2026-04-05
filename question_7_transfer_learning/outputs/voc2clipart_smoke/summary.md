# TLlib VOC->Clipart Transfer Summary

- Profile: `smoke`
- Dataset root: `question_7_transfer_learning/datasets_smoke`
- Python: `3.11.14`
- Torch: `2.11.0`
- Detectron2: `0.6`

## source-only
- Output dir: `question_7_transfer_learning/logs/source_only_smoke/faster_rcnn_R_101_C4/voc2clipart`
- model_final.pth: `True`
- Checkpoints: `2`
- Metrics: `AP=0.0`, `AP50=0.0`, `AP75=0.0`
- Visualizations: `4`

## phase1
- Output dir: `question_7_transfer_learning/logs/d_adapt_smoke/faster_rcnn_R_101_C4/voc2clipart/phase1`
- model_final.pth: `True`
- Checkpoints: `2`
- Metrics: `AP=0.0`, `AP50=0.0`, `AP75=0.0`
- Visualizations: `4`
- Adaptors: `category=trained`, `bbox=skipped_empty_loader`
