import cv2
import torch
from torch.utils.data import Dataset


class AffWildDataset(Dataset):
    def __init__(self, image_paths_list, labels_list, transform=None):
        self.transform = transform
        self.image_paths_list = image_paths_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        chunk_image = self.image_paths_list[idx]
        chunk_label = self.labels_list[idx]
        image_list = []
        # Read images
        for image_path in chunk_image:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            image_list.append(image)

        label_list = [torch.tensor(label).long() for label in chunk_label]

        image_array = torch.stack(image_list, dim=0)
        label_array = torch.stack(label_list, dim=0)

        return image_array, label_array
