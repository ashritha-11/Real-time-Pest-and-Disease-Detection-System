import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os

# ---------------- Paths ----------------
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_FOLDER, "pest_disease_model.pt")
CLASS_INDICES_FILE = os.path.join(MODEL_FOLDER, "class_indices.json")

TRAIN_FOLDER = "dataset/train"
TEST_FOLDER = "dataset/test"

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- Datasets ----------------
train_dataset = datasets.ImageFolder(TRAIN_FOLDER, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder(TEST_FOLDER, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------- Save class indices ----------------
class_indices = train_dataset.class_to_idx
with open(CLASS_INDICES_FILE, "w") as f:
    json.dump(class_indices, f, indent=4)
print("✅ class_indices.json saved!")

# ---------------- Model ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)

# ---------------- Loss & Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------- Training ----------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader.dataset):.4f}")

# ---------------- Testing ----------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# ---------------- Save Model ----------------
torch.save(model.state_dict(), MODEL_FILE)
print(f"✅ Model saved at {MODEL_FILE}")
