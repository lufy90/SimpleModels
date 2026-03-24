
from PIL import Image

import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import sys
from config import PREDICT_CONFIG

# Recreate model (e.g., resnet18 with 5 classes)
model = models.resnet18()
#model = models.mobilenet_v3_small()
#model.fc = nn.Linear(model.fc.in_features, 5)
model.fc = nn.Sequential(
    nn.Dropout(PREDICT_CONFIG["dropout"]),
    nn.Linear(model.fc.in_features, PREDICT_CONFIG["num_classes"])
)

#model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)

device = PREDICT_CONFIG["device"]

#model.load_state_dict(torch.load('model_5classes.pth', map_location=torch.device(device)))
model.load_state_dict(torch.load(PREDICT_CONFIG["model_path"], map_location=torch.device(device)))
#model.load_state_dict(torch.load('model_5classes_mobilenet_v3_cpu.pth', map_location=torch.device(device)))
model.eval()  # Set to eval mode

if PREDICT_CONFIG["quantize_dynamic"]:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

device = torch.device(device)
model = model.to(device)


mean = PREDICT_CONFIG["mean"]
std = PREDICT_CONFIG["std"]

transform = transforms.Compose([
    transforms.Resize((PREDICT_CONFIG["img_size"], PREDICT_CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load image
for imgf in sys.argv[1:]:
    img = Image.open(imgf)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.inference_mode():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    print(f"Predicted class index for {imgf}: ", predicted.item() + PREDICT_CONFIG["class_index_offset"])

