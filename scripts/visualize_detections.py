import cv2
import matplotlib.pyplot as plt
from utils.data_utils import get_image_paths, load_image_and_label, select_random_images
import os

def visualize_detections(dataset_path):
    # Retrieve paths to images in the dataset.
    paths = get_image_paths(dataset_path)
    # Select a random set of 16 images from the training dataset.
    random_images = select_random_images(paths['train_images'], 16)

    # Setup a 4x4 grid for plotting images.
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))

    for i, image_file in enumerate(random_images):
        # Calculate row and column indices for the subplot.
        row = i // 4
        col = i % 4
        # Construct the full path to the image and corresponding label file.
        image_path = os.path.join(paths['train_images'], image_file)
        label_path = os.path.join(paths['train_labels'], os.path.splitext(image_file)[0] + ".txt")
        # Load the image and its labels (annotations).
        image, labels = load_image_and_label(image_path, label_path)

        # Draw bounding boxes on the image based on label information.
        for label in labels:
            # Extract bounding box and class ID from the label.
            class_id, x_center, y_center, width, height = label
            # Convert from normalized to pixel coordinates.
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            # Draw the bounding box on the image.
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        # Display the image with bounding boxes in the subplot.
        axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row, col].axis('off')  # Hide axis for clarity.

    plt.show()  # Render the full grid of images.
