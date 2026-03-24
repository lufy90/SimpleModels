# image_classifier

A lightweight image classification module built with PyTorch.

## Features

- Train an image classifier with transfer learning (`resnet18`).
- Designed for datasets where class labels change gradually (ordinal-like classes).
- Uses an ordinal-aware loss to penalize large-distance misclassification more than near-miss errors.
- Reduces severe mistakes across far-apart classes (for example, highest class predicted as lowest class).
- Run single or batch image prediction from local image paths.
- Export trained `.pth` weights to `.onnx` format.
- Use one shared `config.py` to manage key settings across training, prediction, and export scripts.
- Includes class-imbalance handling in training.

## Scripts

- `train.py`: train and validate the model, then save best and final checkpoints.
- `predict.py`: load a trained checkpoint and print predicted class index for each input image.
- `export.py`: convert a trained PyTorch checkpoint to ONNX.
- `config.py`: central place for common configuration values.
