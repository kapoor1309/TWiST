import os
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from part1.dataset import CustomCholecT50
from utils import save_concatenated_images


def generate_test_cams_with_bboxes(model, dataloader, cam_extractor, output_base_dir, device):
    """
    Generate Grad-CAMs and bounding boxes for the test dataset.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    video_results = {}

    print("Generating CAMs and bounding boxes for test videos...")
    for images, _, frame_ids, video_ids in tqdm(dataloader, desc="Processing Test Dataset"):
        images = images.to(device, non_blocking=True)

        # Predict instrument counts (6 classes)
        with torch.no_grad():
            predictions = model(images).cpu().numpy()  # Raw outputs, no sigmoid applied

        for i in range(len(frame_ids)):
            frame_id = str(frame_ids[i])
            video_id = f"VID{str(video_ids[i]).zfill(2)}"
            frame_number = frame_id.zfill(6)

            # Prepare the output directory for this video's CAMs
            video_output_dir = os.path.join(output_base_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)

            if video_id not in video_results:
                video_results[video_id] = []

            frame_results = []
            original_image = images[i].cpu().numpy().transpose(1, 2, 0)
            normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

            for tool_id in range(6):  # Iterate over 6 possible tools
                predicted_count = max(0, int(round(predictions[i][tool_id])))  # Model's predicted count for the tool
                if predicted_count > 0:  # Generate CAMs only if tool is predicted
                    # Generate Grad-CAM heatmap
                    grayscale_cam = cam_extractor(
                        input_tensor=images[i:i + 1],
                        targets=[ClassifierOutputTarget(tool_id)]
                    )
                    cam = grayscale_cam[0]

                    # Normalize CAM
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                    threshold = 0.5  # Example threshold
                    binary_mask = (cam >= threshold).astype(np.uint8) * 255

                    # Find contours in the binary mask
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Select the top N contours based on area, where N = predicted_count
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:predicted_count]

                    bounding_boxes = []
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Normalize bounding box coordinates
                        x1, y1, x2, y2 = x / cam.shape[1], y / cam.shape[0], (x + w) / cam.shape[1], (y + h) / cam.shape[0]
                        bounding_boxes.append([x1, y1, x2 - x1, y2 - y1])

                    if bounding_boxes:
                        # Create CAM overlay image
                        cam_image = show_cam_on_image(normalized_image.astype(np.float32), cam, use_rgb=True)
                        cam_path = os.path.join(video_output_dir, f"frame_{frame_number}_tool_{tool_id}_cam.png")
                        save_concatenated_images(normalized_image, cam_image, cam_path)

                        # Append tool-specific bounding boxes
                        for bbox in bounding_boxes:
                            frame_results.append({"tool_id": tool_id, "bbox": bbox})

            # Save frame results if there are any tools detected
            if frame_results:
                video_results[video_id].append({
                    "frame_id": frame_id,
                    "tools": frame_results
                })

    # Save all results to JSON files organized by video_id
    for video_id, frames in video_results.items():
        video_json_path = os.path.join(output_base_dir, f"{video_id}_results.json")
        with open(video_json_path, "w") as f:
            json.dump(frames, f, indent=4)

    print("CAM and bounding box generation completed. Results saved to output directory.")


if __name__ == "__main__":
    # Directories and paths
    dataset_dir = "/teamspace/studios/this_studio/CholecT50"
    test_videos = [92, 96, 103, 110, 111]
    train_videos = [6, 2, 8, 1, 4, 14, 5, 10, 12, 13]

    output_base_dir = "output/test_cams_with_bboxes"
    os.makedirs(output_base_dir, exist_ok=True)

    # Initialize the CustomCholecT50 dataset
    cholect = CustomCholecT50(dataset_dir, train_videos=train_videos, test_videos=test_videos, normalize=True, n_splits=2)
    test_dataset = cholect.build()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load the trained model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 output units for instrument counts
    model.load_state_dict(torch.load("trained_model_fold_1.pth"))  # Load trained weights
    model = model.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # Initialize Grad-CAM
    cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])

    # Generate CAMs and bounding boxes for the test dataset
    generate_test_cams_with_bboxes(model, test_loader, cam_extractor, output_base_dir, device)

