import os

import numpy as np
import pandas as pd


class ChunkCreator:
    """Represent data as consecutive chunks of paths to images, corresponding labels and landmarks."""

    def __init__(self, path_data, path_label, seq_len):
        self.data_path = path_data  # Path to folders with extracted frames
        self.label_path = path_label  # Path to txt files with labels
        self.seq_len = seq_len  # Length of the sequence
        self.result_image_paths = []
        self.result_labels = []
        self.result_landmarks = []

    def _get_paths(self, folder_name, frame_list):
        """Get paths to frames."""
        result_path_list = [os.path.join(self.data_path, folder_name, frame) for frame in frame_list]
        return result_path_list

    def _get_index(self, frame_list):
        """Get indexes from frame list."""
        # Get filename without extension, convert to integer and subtract 1, since numeration for frames starts from 1
        def _convert_to_index(frame_name):
            return int(os.path.splitext(frame_name)[0]) - 1

        return list(map(_convert_to_index, frame_list))

    def _produce_data_chunk(self, frame_path_list):
        for x in range(0, len(frame_path_list), self.seq_len):
            path_chunk = frame_path_list[x : x + self.seq_len]
            if len(path_chunk) == self.seq_len:
                self.result_image_paths.append(path_chunk)

    def _produce_label_chunk(self, label_list, frame_index_list):
        frame_index_array = np.array(frame_index_list)
        label_array = np.array(label_list)
        needed_labels = label_array[frame_index_array]
        # Check that labels don't contain values [-5, 5], since corresponding frames are not annotated
        needed_labels = np.array([x for x in needed_labels if (x[0] != -5 and x[1] != 5)])
        for x in range(0, len(needed_labels), self.seq_len):
            label_chunk = needed_labels[x : x + self.seq_len]
            if len(label_chunk) == self.seq_len:
                self.result_labels.append(label_chunk)

    def _produce_landmark_chunk(self, landmarks_df, index_list, multy):
        landmark_list = []
        landmark_str_array = landmarks_df["landmarks_68"].to_numpy()
        if not multy:
            landmark_str_array = landmark_str_array[index_list]
        for array in landmark_str_array:
            try:
                transformed_list = [np.fromstring(x.lstrip()[1:-1], sep=" ") for x in array[1:-1].split("\n")]
            except TypeError:
                print(type(array))
                transformed_list = [[0, 0] for _ in range(68)]

            landmark_list.append(transformed_list)
            if len(landmark_list[0]) == 0:
                landmark_list = [[0, 0] for _ in range(68)]

        for x in range(0, len(landmark_list), self.seq_len):
            landmark_chunk = landmark_list[x : x + self.seq_len]
            if len(landmark_chunk) == self.seq_len:
                self.result_landmarks.append(landmark_chunk)

    def _chunk_folder(self, folder_name):
        """Produce chunks for one folder."""
        multy = False
        if folder_name.find("left") >= 0 or folder_name.find("right") >= 0:
            multy = True
        # Produce chunk for data
        frame_list = os.listdir(os.path.join(self.data_path, folder_name))
        frame_list.remove("keypoints.csv")
        frame_list.sort()
        frame_paths_list = self._get_paths(folder_name, frame_list)  # Get paths to each frame from the folder
        self._produce_data_chunk(frame_paths_list)

        # Now produce chunk for corresponding labels
        frame_index_list = self._get_index(frame_list)  # Get prepared index list from frame names
        label_file_path = os.path.join(
            self.label_path, folder_name + ".txt"
        )  # Read corresponding txt file with labels
        label_df = pd.read_csv(label_file_path)
        label_list = label_df.values.tolist()  # Get list from labels
        self._produce_label_chunk(label_list, frame_index_list)  # Form chunk from labels

        # Now produce chunk for corresponding landmarks
        keypoint_path = os.path.join(self.data_path, folder_name, "keypoints.csv")
        landmarks_df = pd.read_csv(keypoint_path)
        self._produce_landmark_chunk(landmarks_df, frame_index_list, multy)

    def form_result_list(self):
        """Iterate over all dirs and fill resulting lists with corresponding data and label chunks."""
        for folder in os.listdir(self.data_path):
            print(f"Producing chunks for frames in folder {folder}")
            self._chunk_folder(folder)
            # Check sizes
            print("Images: ", len(self.result_image_paths))
            print("Labels: ", len(self.result_labels))
            print("Landmarks: ", len(self.result_landmarks))
            assert len(self.result_image_paths) == len(self.result_labels) == len(self.result_landmarks)

    def print_size(self):
        print(f"Total number of chunks : {len(self.result_image_paths)}")
        print(f"Total number of processed frames: {len(self.result_image_paths)*self.seq_len}")
