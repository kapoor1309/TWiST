import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from part2.dataset2 import CustomCholecT50_2
from part2.model import MultiBranchModel


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        if self.pos_weight is not None:
            loss = -self.pos_weight * target * torch.log(input + 1e-6) - (1 - target) * torch.log(1 - input + 1e-6)
        else:
            loss = -target * torch.log(input + 1e-6) - (1 - target) * torch.log(1 - input + 1e-6)
        return torch.mean(loss)


def train_model2(model, train_loader, val_loader, device, num_epochs=3, save_path="model_part2_vgg16.pth"):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Define loss functions
    instrument_loss_fn = WeightedBCELoss(pos_weight=torch.tensor(2.0).to(device))
    verb_loss_fn = WeightedBCELoss(pos_weight=torch.tensor(1.5).to(device))
    target_loss_fn = WeightedBCELoss(pos_weight=torch.tensor(1.2).to(device))
    triplet_loss_fn = WeightedBCELoss(pos_weight=torch.tensor(1.8).to(device))

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0
        running_loss = 0
        batch_start_time = time.time()

        for batch_idx, (images, triplet_labels, instrument_labels, verb_labels, target_labels, _, _) in enumerate(train_loader):
            images = images.to(device)
            triplet_labels = triplet_labels.to(device)
            instrument_labels = instrument_labels.to(device)
            verb_labels = verb_labels.to(device)
            target_labels = target_labels.to(device)

            # Forward pass
            instrument_logits, verb_logits, target_logits, triplet_logits = model(images)

            # Calculate sequence-level losses
            instrument_loss = instrument_loss_fn(instrument_logits, instrument_labels)
            verb_loss = verb_loss_fn(verb_logits, verb_labels)
            target_loss = target_loss_fn(target_logits, target_labels)
            triplet_loss = triplet_loss_fn(triplet_logits, triplet_labels)

            loss = instrument_loss + verb_loss + target_loss + triplet_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            # Print running loss every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Running Loss: {running_loss / 10:.4f}, Time/Batch: {time.time() - batch_start_time:.2f}s")
                running_loss = 0
                batch_start_time = time.time()


        # Normalize training loss
        normalized_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, triplet_labels, instrument_labels, verb_labels, target_labels, _, _ in val_loader:
                images = images.to(device)
                triplet_labels = triplet_labels.to(device)
                instrument_labels = instrument_labels.to(device)
                verb_labels = verb_labels.to(device)
                target_labels = target_labels.to(device)

                # Forward pass
                instrument_logits, verb_logits, target_logits, triplet_logits = model(images)

                val_loss += (instrument_loss_fn(instrument_logits, instrument_labels) +
                             verb_loss_fn(verb_logits, verb_labels) +
                             target_loss_fn(target_logits, target_labels) +
                             triplet_loss_fn(triplet_logits, triplet_labels)).item()

        normalized_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {normalized_train_loss:.4f} | "
              f"Val Loss: {normalized_val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")




if __name__ == "__main__":
    dataset_dir = "/teamspace/studios/this_studio/CholecT50"
    train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]
    test_videos = [92, 96, 103, 110, 111]

    batch_size = 8
    num_epochs = 3
    seq_len = 5
    num_triplets = 100
    num_instruments = 6
    num_verbs = 10
    num_targets = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_dataset = CustomCholecT50_2(
        dataset_dir, train_videos, test_videos, seq_len=seq_len, normalize=True, n_splits=5
    )
    train_dataset, val_dataset = custom_dataset.get_fold(0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MultiBranchModel(num_triplets, num_instruments, num_verbs, num_targets, seq_len)
    train_model2(model, train_loader, val_loader, device, num_epochs)

    
   

