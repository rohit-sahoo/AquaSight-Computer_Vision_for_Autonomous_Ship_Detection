# scripts/train.py

import os
import sys
import cv2

sys.path.append('..')  # Include the parent directory in the path to import modules from it.

from utils.device_utils import get_device  # Utility to determine and use GPU or CPU.
from ultralytics import YOLO  # Import YOLO model from ultralytics package.


def train_model(config, dataset_path, model_name='yolov8x.pt'):  # Define function to train the model.
    device = get_device()  # Determine the best device (GPU or CPU) for training.
    model = YOLO(model_name).to(device)  # Load the specified YOLO model and move it to the determined device.

    # Convert relative dataset path to absolute path for reliability.
    dataset_abs_path = os.path.abspath(config['dataset_path'])
    # Construct the path to the data.yaml file which contains dataset details.
    data_yaml_path = os.path.join(dataset_abs_path, "data.yaml")

    # Example to show how to read and process an image, not directly related to training.
    image_path = os.path.join(dataset_abs_path, "test/images/02e39612d_jpg.rf.cc5483bb711f080d12b644ff62cf977a.jpg")
    image = cv2.imread(image_path)  # Read the image from the specified path.
    height, width, channels = image.shape  # Extract image dimensions and channel count.
    print(f"The image has dimensions {width}x{height} and {channels} channels.")  # Print image details.

    # Start the training process with parameters specified in the config dictionary.
    model.train(data=data_yaml_path,
                epochs=config['epochs'],  # Number of training epochs.
                imgsz=height,  # Image size for training, taken from the example image (might need adjustment).
                seed=config['seed'],  # Seed for reproducibility.
                batch=config['batch_size'],  # Batch size for training.
                workers=config['workers'])  # Number of workers for data loading.

    print("Training complete.")  # Indicate that the training process has finished.


