import time

import dataset_utils

from collections import OrderedDict

from experiment_ops import train
from experiment_ops import predict


def _prepare_dataset(charset_file, labels_file, desired_image_width, desired_image_height, labels_delimiter=' '):
    image_paths, labels = dataset_utils.read_dataset_list(
        labels_file, delimiter=labels_delimiter)
    max_label_length = len(max(labels, key=len))
    images = dataset_utils.read_images(image_paths=image_paths,
                                       image_extension='png')
    images = dataset_utils.binarize(images)
    images = dataset_utils.resize(images,
                                  desired_height=desired_image_height,
                                  desired_width=desired_image_width)
    images = dataset_utils.invert(images)
    classes = dataset_utils.get_characters_from(charset_file)
    images = dataset_utils.images_as_float32(images)
    labels = dataset_utils.encode(labels, classes)
    num_classes = len(classes) + 1
    labels = dataset_utils.pad(labels, max_label_length)
    return images, labels, num_classes


def train_model(learning_rate, checkpoint_dir, num_epochs=1, save_checkpoint_epochs=1):
    run_params = OrderedDict()
    images, labels, num_classes = _prepare_dataset(charset_file="chars.txt",
                                                   labels_file="train.csv",
                                                   desired_image_width=2048,
                                                   desired_image_height=64)
    run_params["learning_rate"] = learning_rate
    run_params["num_classes"] = num_classes
    train(run_params, images, labels, num_classes, checkpoint_dir, num_epochs=num_epochs,
          save_checkpoint_every_n_epochs=save_checkpoint_epochs)


def predict_with_model(checkpoint_dir):
    run_params = OrderedDict()
    images, labels, num_classes = _prepare_dataset(charset_file="chars.txt",
                                                   labels_file="test.csv",
                                                   desired_image_width=2048,
                                                   desired_image_height=64)
    run_params["num_classes"] = num_classes
    predict(run_params, images, checkpoint_dir)


def main():
    checkpoint_dir = "model-" + time.strftime("%Y%m%d-%H%M%S")
    train_model(learning_rate=0.001, checkpoint_dir=checkpoint_dir, num_epochs=1000,
                save_checkpoint_epochs=100)
    predict_with_model(checkpoint_dir)


if __name__ == '__main__':
    main()
