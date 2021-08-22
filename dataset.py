import cv2
import torch
from torch.utils.data import Dataset


class AffWildDataset(Dataset):
    def __init__(self, image_paths_list, labels_list, landmarks_list, transform=None):
        self.transform = transform
        self.image_paths_list = image_paths_list
        self.labels_list = labels_list
        self.landmarks_list = landmarks_list

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        chunk_image = self.image_paths_list[idx]
        chunk_label = self.labels_list[idx]
        chunk_landmark = self.landmarks_list[idx]
        transformed_image_list = []
        transformed_landmark_list = []
        # Read images
        for image_path, landmarks in zip(chunk_image, chunk_landmark):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                augmented = self.transform(image=image, keypoints=landmarks)
                image = augmented["image"]
                landmark = augmented["keypoints"]
            transformed_image_list.append(image)
            transformed_landmark_list.append(landmark)

        label_list = [torch.tensor(label).float() for label in chunk_label]
        transformed_landmark_tensor_list = [torch.tensor(landmark).float() for landmark in transformed_landmark_list]

        image_array = torch.stack(transformed_image_list, dim=0)
        label_array = torch.stack(label_list, dim=0)
        landmark_array = torch.stack(transformed_landmark_tensor_list, dim=0)

        return image_array, label_array, landmark_array
