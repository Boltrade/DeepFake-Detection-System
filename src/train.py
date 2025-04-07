import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from deepfake_model import DeepfakeCNN
import os

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DeepfakeCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/deepfake_cnn.pth")
