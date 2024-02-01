# scripts/evaluate.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import os
import sys

sys.path.append('..')

from models.yolov_model import load_model
from utils.device_utils import get_device
import random


def evaluate_model(weights_path, confusion_matrix_path, custom_image_dir):
    device = get_device()  # Get the available device (CPU or GPU)
    model = load_model(weights_path, device=device) # Load the YOLO model with specified weights

    # Evaluate the model on the test data
    # Note: Adjustments might be needed based on the YOLO version and API specifics

    metrics = model.val(conf=0.25, split='test')

    # Visualization of evaluation metrics
    ax = sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])
    ax.set_title('YOLO Evaluation Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for p in ax.patches:
        ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='bottom')
    plt.show()

    # Displaying the confusion matrix
    img = mpimg.imread(confusion_matrix_path)
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Custom ship detection function (consider moving to a separate module for reusability)
    def ship_detect(img_path):
        img = cv2.imread(img_path)
        detect_result = model.detect(img)
        detect_img = detect_result.plot()
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        return detect_img

    # Visualizing detections on custom images
    image_files = os.listdir(custom_image_dir)
    selected_images = random.sample(image_files, 16)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    for i, img_file in enumerate(selected_images):
        img_path = os.path.join(custom_image_dir, img_file)
        detect_img = ship_detect(img_path)
        axes[i // 4, i % 4].imshow(detect_img)
        axes[i // 4, i % 4].axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

