import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from part1.dataset import CustomCholecT50
from part2.dataset2 import CustomCholecT50_2
from part1.train_model import train_model, evaluate_model
from part1.cams_train import generate_cams_folder_wise
from pytorch_grad_cam import GradCAM
from part2.train_2 import train_model2, WeightedBCELoss
from part2.model import MultiBranchModel
from part2.tool_probabilities import predict_and_save
from part2.triplet_pred import predict_and_save_triplet
import logging
from merging.processor1 import clean_frame_id, load_triplet_tool_mapping, get_best_triplets_for_tool, process_video, update_video_probabilities, process_all_videos, mapping_data
from merging.processor2 import create_empty_frame_data, process_and_merge_videos
from part1.test_bboxes import generate_test_cams_with_bboxes

print('works')

output_dir = "original_re"
os.makedirs(output_dir, exist_ok = True)

# Loading data
dataset_dir = "/teamspace/studios/this_studio/CholecT50"
train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]
test_videos = [92, 96, 103, 110, 111]  # Define test_videos properly
# Initialize the CustomCholecT50 dataset
cholect = CustomCholecT50(dataset_dir, train_videos, test_videos, normalize=True, n_splits=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------------------------------------

# Train Part 1

for fold_index in range(2):
    # Get the train and validation datasets for the current fold
    train_dataset, val_dataset = cholect.get_fold(fold_index)
    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    # Initialize the ResNet50 model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6)  # 6 output units for 6 instrument counts
    model = model.to(device)
    print(f"Training fold {fold_index + 1}...")
    train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=0.0001, device=device)
    # Save the trained model for the current fold
    torch.save(model.state_dict(), os.path.join(output_dir, f"trained_model1_fold_{fold_index+1}.pth"))
    print(f"Model for fold {fold_index + 1} saved.")

# ------------------------------------------------------------------------------------------------------------

# Get CAMs

train_dataset, val_dataset = cholect.get_fold(0)  # Train and validation split
combined_dataset = ConcatDataset([train_dataset, val_dataset])
data_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
# Load the model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6) # 6 output classes
model.load_state_dict(torch.load(os.path.join(output_dir, "trained_model1_fold_1.pth")))  # Load trained weights
model = model.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
# Initialize Grad-CAM
cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])
# Generate CAMs
output_base_dir = os.path.join(output_dir, 'output/cams_folder_wise')
os.makedirs(output_base_dir, exist_ok=True)
generate_cams_folder_wise(model, data_loader, cam_extractor, output_base_dir, device)

# Get Bounding Boxes

# Initialize the CustomCholecT50 dataset
cholect = CustomCholecT50(dataset_dir, train_videos=train_videos, test_videos=test_videos, normalize=True, n_splits=2)
test_dataset = cholect.build()
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
# Initialize Grad-CAM
cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])
# Generate CAMs and bounding boxes for the test dataset
output_base_dir = os.path.join(output_dir,"output/test_cams_with_bboxes")
generate_test_cams_with_bboxes(model, test_loader, cam_extractor, output_base_dir, device)

# ------------------------------------------------------------------------------------------------------------

# Train Part-2

batch_size = 8
num_epochs = 3
seq_len = 5
num_triplets = 100
num_instruments = 6
num_verbs = 10
num_targets = 15
custom_dataset = CustomCholecT50_2(
    dataset_dir, train_videos, test_videos, seq_len=seq_len, normalize=True, n_splits=5
)
train_dataset, val_dataset = custom_dataset.get_fold(0)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
model = MultiBranchModel(num_triplets, num_instruments, num_verbs, num_targets, seq_len)
model2_path = os.path.join(output_dir, 'trained_model2.pth')
train_model2(model, train_loader, val_loader, device, num_epochs, save_path = model2_path)

# ------------------------------------------------------------------------------------------------------------

# Tool Probabilities

batch_size = 1  # Adjust batch size for test predictions
seq_len = 5
num_triplets = 100
num_instruments = 6
num_verbs = 10
num_targets = 15

print("Initializing dataset...")
# Load the dataset
custom_dataset = CustomCholecT50_2(
    dataset_dir, train_videos=train_videos, test_videos=test_videos, seq_len=seq_len, normalize=True, n_splits=5
)
test_dataset = custom_dataset.build()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print("Loading model...")
# Instantiate the model and load the weights
model = MultiBranchModel(num_triplets, num_instruments, num_verbs, num_targets, seq_len)
model.load_state_dict(torch.load(os.path.join(output_dir, 'trained_model2.pth')))  # Load the saved model weights
print("Model loaded successfully from model.pth")

print("Starting prediction process...")
# Predict and save to JSON
predict_and_save(model, test_loader, device, save_dir=os.path.join(output_dir, "predictions"))

# Triplet predictions

print("Starting prediction process...")
# Predict and save to JSON
predict_and_save_triplet(model, test_loader, device, save_dir=os.path.join(output_dir, "predictions"))

# ------------------------------------------------------------------------------------------------------------

# Merging Module

# Processor 1
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

video_ids = [92, 96, 103, 110, 111]
predictions_path = os.path.join(output_dir, "predictions")
bboxes_path = os.path.join(output_dir, "output/test_cams_with_bboxes")

process_all_videos(video_ids, output_dir, predictions_path, bboxes_path, mapping_data)

# Processor 2
video_configs = {
    '92': 2123,
    '96': 1706,
    '103': 2219,
    '110': 2176,
    '111': 2145
}
process_and_merge_videos(output_dir, video_configs)