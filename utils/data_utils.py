# utils/data_utils.py

import os
import cv2
import random

def get_image_paths(dataset_path):
    return {
        "train_images": os.path.join(dataset_path, "train/images"),
        "train_labels": os.path.join(dataset_path, "train/labels"),
        "test_images": os.path.join(dataset_path, "test/images"),
        "test_labels": os.path.join(dataset_path, "test/labels"),
        "val_images": os.path.join(dataset_path, "valid/images"),
        "val_labels": os.path.join(dataset_path, "valid/labels"),
    }

def load_image_and_label(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    boxes = []
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                boxes.append((class_id, x_center, y_center, width, height))
    return image, boxes

def select_random_images(image_path, num_images=16):
    image_files = os.listdir(image_path)
    random_images = random.sample(image_files, num_images)
    return random_images
