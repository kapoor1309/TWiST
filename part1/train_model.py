import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from part1.dataset import CustomCholecT50

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()  # Mean Squared Error for count prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels, _, no) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, _, no in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
    return val_loss / len(dataloader)

# if __name__ == "__main__":
#     dataset_dir = "/teamspace/studios/this_studio/CholecT50"
#     train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]
#     test_videos = [92, 96, 103, 110, 111]  # Define test_videos properly

#     # Initialize the CustomCholecT50 dataset
#     cholect = CustomCholecT50(dataset_dir, train_videos, test_videos, normalize=True, n_splits=2)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for fold_index in range(2):
#         # Get the train and validation datasets for the current fold
#         train_dataset, val_dataset = cholect.get_fold(fold_index)

#         # Create DataLoaders for training and validation
#         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

#         # Initialize the ResNet50 model
#         model = models.vgg1(pretrained=True)
#         model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)  # 6 output units for 6 instrument counts
#         model = model.to(device)

#         print(f"Training fold {fold_index + 1}...")
#         train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=0.0001, device=device)

#         # Save the trained model for the current fold
#         torch.save(model.state_dict(), f"trained_model_fold_{fold_index+1}.pth")

#         print(f"Model for fold {fold_index + 1} saved.")