import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATA_DIR = 'data/iris_dataset'  
BATCH_SIZE = 16
EPOCHS = 3 
LR = 1e-4 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use pretrained weights transforms + add augmentation for train
pretrained_weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

train_transform = transforms.Compose([
    transforms.RandomRotation(15),               # rotate +-15 degrees
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    pretrained_weights.transforms()              # includes normalization etc.
])

val_transform = pretrained_weights.transforms()  # only resize + normalize for val

# Load dataset once (no transform)
full_dataset = datasets.ImageFolder(DATA_DIR)

# Split dataset into 80% train, 20% val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply transforms to split datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(full_dataset.classes)

class IrisEncoder(nn.Module):
    def __init__(self, num_classes):
        super(IrisEncoder, self).__init__()
        self.backbone = models.convnext_tiny(weights=pretrained_weights)
        # Replace classifier head
        self.backbone.classifier[2] = nn.Linear(
            self.backbone.classifier[2].in_features, num_classes
        )

    def forward(self, x):
        return self.backbone(x)

model = IrisEncoder(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # verbose removed

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean()

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels).item()

    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += accuracy(outputs, labels).item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    scheduler.step(val_loss)

    # Print current learning rate manually since verbose=True is removed
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/iris_encoder_convnext.pth')
        print(f"Saved best model with val acc: {best_val_acc:.4f}")
