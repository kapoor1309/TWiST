import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import KFold
from torchvision import transforms


class T50(Dataset):
    def __init__(self, video_dir, annotation_file, video_id, seq_len=1, transform=None):
        """
        Loads sequences of frames from a single video folder along with their annotations.
        Ensures matching frame IDs and skips frames with triplet ID -1.
        """
        self.video_dir = video_dir
        self.video_id = video_id
        self.seq_len = seq_len
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)
        self.annotations = annotations["annotations"]
        self.transform = transform

        # Collect valid frame IDs (present in both video folder and JSON, and triplet ID != -1)
        all_frame_ids = sorted(map(int, self.annotations.keys()))
        valid_frame_ids = []
        for frame_id in all_frame_ids:
            triplets = self.annotations[str(frame_id)]
            if triplets and triplets[0][0] != -1.0:  # Check triplet ID
                basename = f"{str(frame_id).zfill(6)}.png"
                img_path = os.path.join(self.video_dir, basename)
                if os.path.exists(img_path):  # Ensure image file exists
                    valid_frame_ids.append(frame_id)

        # Generate valid start indices for sequences
        self.frame_ids = valid_frame_ids
        self.valid_indices = [
            i for i in range(len(valid_frame_ids) - seq_len + 1)
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # Get the start index for the sequence
        start_idx = self.valid_indices[index]

        # Collect a sequence of frames and their annotations
        sequence_images = []
        triplet_labels = []
        tool_labels = []
        verb_labels = []
        target_labels = []
        frame_ids = []

        for offset in range(self.seq_len):
            frame_id = self.frame_ids[start_idx + offset]
            triplets = self.annotations[str(frame_id)]
            basename = f"{str(frame_id).zfill(6)}.png"
            img_path = os.path.join(self.video_dir, basename)

            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Image file not found: {img_path}")

            if self.transform:
                image = self.transform(image)

            # Extract labels for the frame
            triplet_label, tool_label, verb_label, target_label = self.get_binary_labels(triplets)

            sequence_images.append(image)
            triplet_labels.append(triplet_label)
            tool_labels.append(tool_label)
            verb_labels.append(verb_label)
            target_labels.append(target_label)
            frame_ids.append(frame_id)

        # Convert sequence to tensors and stack
        sequence_images = torch.stack(sequence_images)  # Shape: [seq_len, channels, height, width]
        triplet_labels = torch.tensor(triplet_labels)  # Shape: [seq_len, num_triplets]
        tool_labels = torch.tensor(tool_labels)  # Shape: [seq_len, num_tools]
        verb_labels = torch.tensor(verb_labels)  # Shape: [seq_len, num_verbs]
        target_labels = torch.tensor(target_labels)  # Shape: [seq_len, num_targets]

        return sequence_images, triplet_labels, tool_labels, verb_labels, target_labels, frame_ids, self.video_id

    def get_binary_labels(self, labels):
        """
        Extracts binary labels for triplet, tool, verb, and target from annotations.
        """
        tool_label = np.zeros([6])  # Assuming 6 tool classes
        verb_label = np.zeros([10])  # Assuming 10 verb classes
        target_label = np.zeros([15])  # Assuming 15 target classes
        triplet_label = np.zeros([100])  # Assuming 100 triplet classes

        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] = 1  # Mark triplet presence
            tool = label[1:2]
            if tool[0] != -1.0:
                tool_label[tool[0]] = 1  # Mark tool presence
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] = 1  # Mark verb presence
            target = label[8:9]
            if target[0] != -1.0:
                target_label[target[0]] = 1  # Mark target presence

        return triplet_label, tool_label, verb_label, target_label

class CustomCholecT50_2:
    def __init__(self, dataset_dir, train_videos, test_videos, seq_len=1, normalize=True, n_splits=5):
        self.dataset_dir = dataset_dir
        self.train_videos = train_videos
        self.test_videos = test_videos
        self.seq_len = seq_len
        self.normalize = normalize
        self.n_splits = n_splits

        train_transform, test_transform = self.transform()
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.folds = list(self.kfold.split(self.train_videos))

        self.build_test_dataset(self.test_transform)

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.Resize((256, 448))),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            normalize
        ])

        return train_transform, test_transform

    def build_dataset(self, video_indices, transform):
        """
        Builds a dataset folder by folder, processing frames from each video sequentially.
        """
        iterable_dataset = []
        for video in video_indices:
            video_dir = os.path.join(self.dataset_dir, f"videos/VID{str(video).zfill(2)}")
            annotation_file = os.path.join(self.dataset_dir, f"labels/VID{str(video).zfill(2)}.json")

            dataset = T50(video_dir, annotation_file, video, seq_len=self.seq_len, transform=transform)
            iterable_dataset.append(dataset)

        # Combine all video datasets sequentially
        return ConcatDataset(iterable_dataset)

    def build_test_dataset(self, transform):
        """
        Build the test dataset sequentially by video folder.
        """
        self.test_dataset = self.build_dataset(self.test_videos, transform)

    def get_fold(self, fold_index):
        train_indices, val_indices = self.folds[fold_index]

        train_videos = [self.train_videos[i] for i in train_indices]
        val_videos = [self.train_videos[i] for i in val_indices]

        train_dataset = self.build_dataset(train_videos, self.train_transform)
        val_dataset = self.build_dataset(val_videos, self.test_transform)

        return train_dataset, val_dataset

    def build(self):
        return self.test_dataset




# Function to test sequence loading
def test_sequence_loading(dataset_dir, video_id, seq_len):
    video_dir = os.path.join(dataset_dir, f"videos/VID{str(video_id).zfill(2)}")
    annotation_file = os.path.join(dataset_dir, f"labels/VID{str(video_id).zfill(2)}.json")

    transform = transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor()
    ])

    dataset = T50(video_dir, annotation_file, video_id, seq_len, transform)

    # Load a single sequence
    if len(dataset) > 0:
        sequence_images, triplet_labels, tool_labels, verb_labels, target_labels, frame_ids, video_id = dataset[0]
        print(f"Frame IDs in Sequence: {frame_ids}")
        print(f"Sequence Shape: {sequence_images.shape}")  # [seq_len, channels, height, width]
        print(f"Triplet Labels Shape: {triplet_labels.shape}")  # [seq_len, num_triplets]
        print(f"Tool Labels Shape: {tool_labels.shape}")  # [seq_len, num_tools]
        print(f"Verb Labels Shape: {verb_labels.shape}")  # [seq_len, num_verbs]
        print(f"Target Labels Shape: {target_labels.shape}")  # [seq_len, num_targets]")
    else:
        print("No valid sequences found in the dataset.")


# Path to your dataset
dataset_dir = "/teamspace/studios/this_studio/CholecT50"

# Specify the video ID and sequence length to test
video_id = 1  # Replace with the desired video ID
seq_len = 5   # Length of sequences

# Test the sequence loading function
test_sequence_loading(dataset_dir, video_id, seq_len)
