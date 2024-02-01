import sys
import os
import yaml

# Importing custom functions from other scripts
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize_detections import visualize_detections

def load_config(config_path='config.yaml'):
    # Function to load YAML configuration files
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Main function to orchestrate the training, evaluation, and visualization processes

    # Set up paths and parameters. Adjust these paths according to your project structure.
    dataset_path = '../ships-aerial-images'  # Path to the dataset directory
    config = load_config('config/config.yaml')  # Loading the configuration settings from a YAML file
    model_name = 'yolov8x.pt'  # Specify the model name or path

    # Paths to the trained model weights and confusion matrix for evaluation
    weights_path = '../runs/detect/train/weights/best.pt'
    confusion_matrix_path = '../runs/detect/train/confusion_matrix.png'
    custom_image_dir = '../ships-aerial-images/test/images'  # Directory containing custom images for evaluation

    # Visualize detections on images from the dataset
    visualize_detections(dataset_path)

    # Train the model with the specified configuration and dataset
    print("Starting model training...")
    train_model(config=config, dataset_path=dataset_path, model_name=model_name)

    # Evaluate the trained model using the specified weights and visualizing the results
    print("Evaluating model...")
    evaluate_model(weights_path, confusion_matrix_path, custom_image_dir)


if __name__ == "__main__":
    main()
