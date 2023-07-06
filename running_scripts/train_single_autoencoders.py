import sys

sys.path.insert(0, '..')

from dataset.single_autoencoder_image_dataset import SingleAutoencoderImageDataset
from model_architectures.single_autoencoder import SingleAutoencoder
import yaml

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    dataset_base_path = f'{params["dataset_parameters"]["base_path"]}bonafide/*/train/*/*/*.jpg'
    validation_dataset_base_path = f'{params["dataset_parameters"]["base_path"]}bonafide/*/' \
                                   f'/train/*/*/*.jpg'

    print("Creating a database based on the specified base path and input modality")
    if params["prediction_parameters"]["model_pretrain"]:
        model = SingleAutoencoder(summarize_model=True, input_dimension=params["application_parameters"]["image_size"], pre_trained_path=f'trained_models/{params["application_parameters"]["dataset"]}/best_autoencoder_single_autoencoder.hdf5')
    else:
        model = SingleAutoencoder(summarize_model=True, input_dimension=params["application_parameters"]["image_size"])

    image_dataset = SingleAutoencoderImageDataset(base_path=dataset_base_path,
                                                  image_size=(params["application_parameters"]["image_size"],
                                                              params["application_parameters"]["image_size"]),
                                                  batch_size=params["model_parameters"]["batch_size"],
                                                  augmentation_list=params["input_parameters"]["augmentation_list"])

    validation_image_dataset = SingleAutoencoderImageDataset(base_path=validation_dataset_base_path,
                                                             image_size=(
                                                                 params["application_parameters"]["image_size"],
                                                                 params["application_parameters"]["image_size"]),
                                                             batch_size=params["model_parameters"]["batch_size"],
                                                             augmentation_list=[])

    print(f"Training the single autoencoder for multimodality")
    model.fit_model(input_data=image_dataset, validation_data=image_dataset, number_of_epochs=params["model_parameters"]["number_of_epochs"],
                    dataset=params["application_parameters"]["dataset"])
