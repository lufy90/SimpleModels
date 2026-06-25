
import copy
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from config import TRAIN_CONFIG

# -----------------------------
# Configurations
# -----------------------------
DATA_DIR = TRAIN_CONFIG["data_dir"]
BATCH_SIZE = TRAIN_CONFIG["batch_size"]
IMG_SIZE = TRAIN_CONFIG["img_size"]
NUM_CLASSES = TRAIN_CONFIG["num_classes"]
EPOCHS = TRAIN_CONFIG["epochs"]
LR = TRAIN_CONFIG["learning_rate"]
DEVICE = TRAIN_CONFIG["device"]
SEED = TRAIN_CONFIG["seed"]
DISTANCE_LOSS_WEIGHT = TRAIN_CONFIG["distance_loss_weight"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class OrdinalAwareLoss(nn.Module):
    def __init__(self, class_weights, num_classes, distance_weight=0.35):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.distance_weight = distance_weight
        class_ids = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer("class_ids", class_ids)

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        probs = torch.softmax(logits, dim=1)
        class_ids = self.class_ids.to(logits.device)
        distances = torch.abs(class_ids.unsqueeze(0) - labels.float().unsqueeze(1))
        expected_distance = (probs * distances).sum(dim=1).mean()
        return ce_loss + self.distance_weight * expected_distance

# -----------------------------
# Dataset & DataLoaders
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

base_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
train_base_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_base_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

if NUM_CLASSES != len(base_dataset.classes):
    raise ValueError(f"NUM_CLASSES={NUM_CLASSES} does not match dataset classes={len(base_dataset.classes)}")

train_size = int((1 - TRAIN_CONFIG["val_split"]) * len(base_dataset))
val_size = len(base_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

train_dataset = Subset(train_base_dataset, train_subset.indices)
val_dataset = Subset(val_base_dataset, val_subset.indices)

train_targets = torch.tensor([base_dataset.samples[i][1] for i in train_subset.indices], dtype=torch.long)
sample_counts = torch.bincount(train_targets, minlength=NUM_CLASSES).float()
class_weights = (sample_counts.sum() / (sample_counts + 1e-6))
class_weights = class_weights / class_weights.sum()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=6, pin_memory=True, prefetch_factor=3, persistent_workers=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
    num_workers=6, pin_memory=True, prefetch_factor=3, persistent_workers=True
)

print(f"Classes: {base_dataset.classes}")
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
print(f"Train class counts: {sample_counts.tolist()}")

# -----------------------------
# Load Pretrained Model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES),
)

model = model.to(DEVICE)

# -----------------------------
# Loss, Optimizer, Scheduler
# -----------------------------
criterion = OrdinalAwareLoss(
    class_weights=class_weights.to(DEVICE),
    num_classes=NUM_CLASSES,
    distance_weight=DISTANCE_LOSS_WEIGHT,
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=TRAIN_CONFIG["scheduler_factor"],
    patience=TRAIN_CONFIG["scheduler_patience"],
)

# -----------------------------
# Training Loop
# -----------------------------
best_val_acc = 0.0
best_state_dict = None
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0.0
    val_severe_miss = 0
    val_5_to_1 = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_running_loss += loss.item()
            val_severe_miss += (torch.abs(preds - labels) >= 3).sum().item()
            val_5_to_1 += ((labels == NUM_CLASSES - 1) & (preds == 0)).sum().item()

    val_acc = val_correct / val_total
    val_loss = val_running_loss / len(val_loader)
    val_severe_miss_rate = val_severe_miss / val_total

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_state_dict, TRAIN_CONFIG["best_model_path"])

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2%} | "
        f"Severe Miss Rate: {val_severe_miss_rate:.2%} | 5->1 Errors: {val_5_to_1} | "
        f"LR: {current_lr:.6f}"
    )

# -----------------------------
# Save the Trained Model
# -----------------------------
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
torch.save(model.state_dict(), TRAIN_CONFIG["final_model_path"])
print(f"Best Validation Accuracy: {best_val_acc:.2%}")
print(f"Model saved to {TRAIN_CONFIG['final_model_path']}")

