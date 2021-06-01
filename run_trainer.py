from dataset import AffWildDataset
from tranforms import get_transforms
from utils.chunk_creator import ChunkCreator

train_path = "./data/dataset/Train_processed"
label_path = "./data/annotations_VA/Train_Set"
seq_len = 80


def main():
    # Test AffWild dataset class
    data_chunks = ChunkCreator(path_data=train_path, path_label=label_path, seq_len=seq_len)
    data_chunks.form_result_list()
    image_paths = data_chunks.result_image_paths
    labels = data_chunks.result_labels

    train_dataset = AffWildDataset(
        image_paths_list=image_paths, labels_list=labels, transform=get_transforms(data="train")
    )

    print(train_dataset[5])
    print(train_dataset[5][0].shape)
    print(train_dataset[5][1].shape)


if __name__ == "__main__":
    main()
