from dataset.image_dataset import ImageDataset

dataset_base_path = '../spoofing_dataset/training_real/'
image_size = 224
augmentation_list = ["flip", "zoom", "rotate"]

if __name__ == '__main__':

    print("Creating a database based on the specified base path")
    image_dataset = ImageDataset(base_path=dataset_base_path, image_size=(image_size, image_size), augmentation_list=augmentation_list)

    print("Loading the images and applying the data augmentation operations")
    database = image_dataset.load_data()

    __import__("IPython").embed()