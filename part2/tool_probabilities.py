import json
import os
import torch
from torch.utils.data import DataLoader
from part2.dataset2 import CustomCholecT50_2
import time
from part2.model import MultiBranchModel
from datetime import datetime, timedelta

def predict_and_save(model, test_loader, device, save_dir="predictions"):
    model = model.to(device)
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    video_predictions = {}  # Store predictions for all videos
    
    # Setup progress tracking
    total_batches = len(test_loader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(test_loader):
        images, _, _, _, _, frame_ids, video_id = batch  # Unpack batch
        images = images.to(device)
        
        with torch.no_grad():
            # Forward pass (extract all logits)
            instrument_logits, verb_logits, target_logits, triplet_logits = model(images)
            
            # We're interested in instrument_logits now instead of triplet_logits
            instrument_probs = instrument_logits.squeeze(0)  # Ensure proper dimensions
        
        # Ensure instrument_probs has the correct shape and frame_ids match
        if instrument_probs.shape[0] != len(frame_ids):
            raise ValueError(f"Shape mismatch: instrument_probs has shape {instrument_probs.shape} "
                          f"but frame_ids has length {len(frame_ids)}")
        
        # Iterate over frame_ids and corresponding instrument probabilities
        for i, frame_id in enumerate(frame_ids):
            frame_id = frame_id.item()  # Convert tensor to integer
            video_id_str = f"video_{str(video_id[0].item()).zfill(2)}"
            
            if video_id_str not in video_predictions:
                video_predictions[video_id_str] = {}  # Initialize dictionary for the video
            
            video_predictions[video_id_str][frame_id] = {
                "tool_probabilities": instrument_probs[i].tolist()
            }
        
        # Calculate and display progress
        elapsed_time = time.time() - start_time
        progress = (batch_idx + 1) / total_batches
        eta = elapsed_time / (progress) * (1 - progress)
        
        # Format time remaining
        eta_formatted = str(timedelta(seconds=int(eta)))
        
        # Print progress information
        print(f"\rProgress: [{batch_idx + 1}/{total_batches}] "
              f"{progress * 100:.2f}% complete | "
              f"Time elapsed: {str(timedelta(seconds=int(elapsed_time)))} | "
              f"ETA: {eta_formatted}", end="")

    print("\nPrediction process completed!")
    
    # Save predictions for each video
    for video_id_str, predictions in video_predictions.items():
        save_path = os.path.join(save_dir, f"{video_id_str}.json")
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"Predictions saved for {video_id_str} to {save_path}")

if __name__ == "__main__":
    dataset_dir = "/teamspace/studios/this_studio/CholecT50"
    test_videos = [92, 96, 103, 110, 111]
    train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]
    
    batch_size = 1  # Adjust batch size for test predictions
    seq_len = 5
    num_triplets = 100
    num_instruments = 6
    num_verbs = 10
    num_targets = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing dataset...")
    # Load the dataset
    custom_dataset = CustomCholecT50(
        dataset_dir, train_videos=train_videos, test_videos=test_videos, seq_len=seq_len, normalize=True, n_splits=5
    )
    test_dataset = custom_dataset.build()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Loading model...")
    # Instantiate the model and load the weights
    model = MultiBranchModel(num_triplets, num_instruments, num_verbs, num_targets, seq_len)
    model.load_state_dict(torch.load("model_part2_vgg16.pth"))  # Load the saved model weights
    print("Model loaded successfully from model.pth")
    
    print("Starting prediction process...")
    # Predict and save to JSON
    predict_and_save(model, test_loader, device, save_dir="predictions")


