import torch
from torchvision import models
import torch.nn as nn
import sys
from config import EXPORT_CONFIG

model_path = sys.argv[1]
output_path = model_path.replace('.pth', '.onnx')

DEVICE = EXPORT_CONFIG["device"]

model = models.resnet18()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, EXPORT_CONFIG["num_classes"]),
)

state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

dummy_input = torch.randn(
    1,
    3,
    EXPORT_CONFIG["input_size"],
    EXPORT_CONFIG["input_size"],
    device=DEVICE,
)

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=EXPORT_CONFIG["opset_version"],
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print(f"Model exported to {output_path}")
