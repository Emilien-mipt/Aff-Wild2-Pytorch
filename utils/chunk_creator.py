import os

import numpy as np
import pandas as pd


class ChunkCreator:
    """Represent data as consecutive chunks of paths to images and corresponding labels."""

    def __init__(self, path_data, path_label, seq_len):
        self.data_path = path_data  # Path to folders with extracted frames
        self.label_path = path_label  # Path to txt files with labels
        self.seq_len = seq_len  # Length of the sequence
        self.result_image_paths = []
        self.result_labels = []

    def _produce_data_chunk(self, frame_path_list):
        return [
            frame_path_list[x : x + self.seq_len]
            for x in range(0, len(frame_path_list), self.seq_len)
            if len(frame_path_list[x : x + self.seq_len]) == self.seq_len
        ]

    def _produce_label_chunk(self, label_list, frame_index_list):
        frame_index_array = np.array(frame_index_list)
        label_array = np.array(label_list)
        needed_labels = label_array[frame_index_array]
        # Check that labels don't contain values [-5, 5], since corresponding frames are not annotated
        needed_labels = np.array([x for x in needed_labels if (x[0] != -5 and x[1] != 5)])
        result_label_chunk = [
            needed_labels[x : x + self.seq_len]
            for x in range(0, len(needed_labels), self.seq_len)
            if len(needed_labels[x : x + self.seq_len]) == self.seq_len
        ]
        return result_label_chunk

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

    def _chunk_folder(self, folder_name):
        """Produce chunks for one folder."""
        # Produce chunk for data
        frame_list = os.listdir(os.path.join(self.data_path, folder_name))
        frame_list.remove("keypoints.csv")
        frame_list.sort()
        frame_paths_list = self._get_paths(folder_name, frame_list)  # Get paths to each frame from the folder
        data_chunk = self._produce_data_chunk(frame_paths_list)
        # Now produce chunk for corresponding labels
        frame_index_list = self._get_index(frame_list)  # Get prepared index list from frame names
        label_file_path = os.path.join(
            self.label_path, folder_name + ".txt"
        )  # Read corresponding txt file with labels
        label_df = pd.read_csv(label_file_path)
        label_list = label_df.values.tolist()  # Get list from labels
        label_chunk = self._produce_label_chunk(label_list, frame_index_list)  # Form chunk from labels
        # Data chunk length should match with label chunk length
        assert len(data_chunk) == len(label_chunk)
        return {"data_chunk": data_chunk, "label_chunk": label_chunk}

    def form_result_list(self):
        """Iterate over all dirs and fill resulting lists with corresponding data and label chunks."""
        for folder in os.listdir(self.data_path):
            print(f"Producing chunks for frames in folder {folder}")
            data_chunks = self._chunk_folder(folder)["data_chunk"]
            label_chunks = self._chunk_folder(folder)["label_chunk"]
            # Append in result lists
            for data_chunk in data_chunks:
                self.result_image_paths.append(data_chunk)
            for label_chunk in label_chunks:
                self.result_labels.append(label_chunk)
            print(f"Number of chunks in folder {folder}: {len(data_chunks)}")
        assert len(self.result_image_paths) == len(self.result_labels)

    def print_size(self):
        print(f"Total number of chunks : {len(self.result_image_paths)}")
        print(f"Total number of processed frames: {len(self.result_image_paths)*self.seq_len}")
