import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import KFold
from torchvision import transforms

class T50(Dataset):
    def __init__(self, video_dir, annotation_file, video_id, transform=None):
        """
        Loads all frames from a single video folder along with their annotations.
        """
        self.video_dir = video_dir
        self.video_id = video_id
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)
        self.annotations = annotations["annotations"]
        self.transform = transform

        # Collect frame IDs sorted numerically
        self.frame_ids = sorted(map(int, self.annotations.keys()))

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        triplets = self.annotations[str(frame_id)]

        basename = f"{str(frame_id).zfill(6)}.png"
        img_path = os.path.join(self.video_dir, basename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        if self.transform:
            image = self.transform(image)

        count_labels = self.get_count_labels(triplets)

        return image, count_labels, frame_id, self.video_id

    def get_count_labels(self, labels):
        tool_counts = np.zeros(6)  # Assuming 6 classes
        for label in labels:
            tool = label[1:2]
            if tool[0] != -1.0:
                tool_counts[tool[0]] += 1
        return tool_counts


class CustomCholecT50:
    def __init__(self, dataset_dir, train_videos, test_videos, normalize=True, n_splits=5):
        self.dataset_dir = dataset_dir
        self.train_videos = train_videos
        self.test_videos = test_videos
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
            transforms.Resize((256, 448)),
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

            dataset = T50(video_dir, annotation_file, video, transform)
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
