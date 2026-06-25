import torch


TRAIN_CONFIG = {
    "data_dir": "dataset",
    "batch_size": 64,
    "img_size": 224,
    "num_classes": 5,
    "epochs": 10,
    "learning_rate": 0.0001,
    "seed": 42,
    "distance_loss_weight": 0.35,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "best_model_path": "best.pth",
    "final_model_path": "model_5classes.pth",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


PREDICT_CONFIG = {
    "num_classes": 5,
    "img_size": 224,
    "dropout": 0.5,
    "model_path": "best.pth",
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "quantize_dynamic": True,
    "class_index_offset": 1,
}


EXPORT_CONFIG = {
    "num_classes": 5,
    "input_size": 224,
    "opset_version": 18,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
