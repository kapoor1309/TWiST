import os
import json
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from utils import save_concatenated_images
from dataset import CustomCholecT50  # Updated dataset module


def generate_cams_folder_wise(model, dataloader, cam_extractor, output_base_dir, device):
    """
    Generate Grad-CAMs and save images concatenated with CAMs folder-wise.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    results = []

    print("Generating CAMs folder-wise...")
    for images, labels, frame_ids, video_ids in tqdm(dataloader, desc="Processing Dataset"):
        images = images.to(device, non_blocking=True)
        labels = labels.cpu().numpy()  # Convert labels to NumPy for indexing and processing

        for i in range(len(frame_ids)):
            frame_id = str(frame_ids[i])
            video_id = f"VID{str(video_ids[i]).zfill(2)}"
            frame_number = frame_id.zfill(6)

            # Prepare the output directory for this video's CAMs
            video_output_dir = os.path.join(output_base_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)

            cam_results = []
            # Normalize image for visualization
            original_image = images[i].cpu().numpy().transpose(1, 2, 0)
            normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

            for tool_id in range(6):  # Assuming 6 tools
                if labels[i][tool_id] > 0:  # Generate CAMs only for tools present in the frame
                    # Generate Grad-CAM heatmap
                    grayscale_cam = cam_extractor(
                        input_tensor=images[i:i + 1],
                        targets=[ClassifierOutputTarget(tool_id)]
                    )
                    cam = grayscale_cam[0]

                    # Create the CAM overlay image
                    cam_image = show_cam_on_image(normalized_image.astype(np.float32), cam, use_rgb=True)

                    # Save the concatenated image
                    cam_path = os.path.join(video_output_dir, f"frame_{frame_number}_tool_{tool_id}_cam.png")
                    save_concatenated_images(normalized_image, cam_image, cam_path)

                    # Append tool-specific results
                    cam_results.append({"tool_id": tool_id, "cam_path": cam_path})

            # Append results for this frame
            results.append({
                "frame_id": frame_id,
                "video_id": video_id,
                "tools": cam_results
            })

    # Save all results to a JSON file
    output_json_path = os.path.join(output_base_dir, "cams_results.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"CAM generation completed. Results saved to {output_json_path}")


if __name__ == "__main__":
    # Directories and paths
    dataset_dir = "/teamspace/studios/this_studio/CholecT50"
    train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]
    test_videos = [92, 96, 103, 110, 111]

    output_base_dir = "output/cams_folder_wise"
    os.makedirs(output_base_dir, exist_ok=True)

    # Initialize the dataset
    cholect = CustomCholecT50(dataset_dir, train_videos, test_videos, normalize=True, n_splits=2)
    train_dataset, val_dataset = cholect.get_fold(0)  # Train and validation split
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    data_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load the model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 output classes
    model.load_state_dict(torch.load("trained_model_fold_1.pth"))  # Load trained weights
    model = model.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # Initialize Grad-CAM
    cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])

    # Generate CAMs
    generate_cams_folder_wise(model, data_loader, cam_extractor, output_base_dir, device)


