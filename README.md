# AquaSight: Computer Vision for Autonomous Ship Detection

AquaSight is a cutting-edge computer vision application designed to automate the detection and analysis of ships within aerial and satellite imagery. By leveraging the power of the YOLOv8 algorithm, AquaSight aims to provide high-precision insights for maritime surveillance, traffic management, and environmental monitoring.

![Ship Detection output](https://github.com/rohit-sahoo/AquaSight-Computer_Vision_for_Autonomous_Ship_Detection/blob/main/results.png "Ship Detection")

## Problem
Ships aerial images object detection is a challenging task due to the large size of the images and the variability of the objects in the scene. Ships can appear at different scales, orientations, and under various lighting conditions. Manual inspection of these images can be time-consuming, and the detection of ships can be prone to errors due to human error or the sheer volume of images that need to be analyzed. Additionally, there is a need for an accurate and efficient object detection algorithm that can handle the scale and complexity of aerial images.

## Agitate
The current methods for ships aerial images object detection are either too time-consuming, too computationally intensive, or not accurate enough to meet the demands of the task. Traditional object detection methods are not able to handle the scale and complexity of aerial images, while deep learning-based methods such as YOLO have shown promising results but require extensive training to achieve high accuracy. Furthermore, the lack of a publicly available dataset of ships aerial images has limited the development of object detection models for this specific task. This makes it difficult for researchers and practitioners to test and compare their methods, hindering progress in this field.

## Solution
To overcome these challenges, YOLO can be used for ships aerial images object detection. YOLO is a deep learning-based object detection algorithm that has shown promising results in various applications. YOLO can handle the scale and complexity of aerial images while achieving high accuracy. Additionally, by using a publicly available dataset of ships aerial images, researchers and practitioners can compare and test their methods, leading to better models and advancements in the field. The development of YOLO-based models for ships aerial images object detection can lead to more efficient and accurate methods for monitoring ships in various applications, such as maritime surveillance, search and rescue, and environmental monitoring.

## Features

- **Automated Ship Detection**: Quickly identify ships in vast oceanic and coastal areas with high accuracy.
- **Performance Analysis**: Evaluate model performance with metrics such as mAP (mean Average Precision), precision, and recall.
- **Visual Insights**: Visualize detection results with bounding boxes and analyze model performance with various plots.
- **Customizable Training**: Easy-to-use scripts for training the model on custom datasets to improve detection capabilities.

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies


## Quick Start with `main.py`

The `main.py` script is designed to streamline the process of training, evaluating, and visualizing ship detections in a few simple steps.

### Running the Application

1. **Configure Your Environment**: Before running the script, ensure you have all necessary dependencies installed by following the installation instructions provided above.

2. **Prepare Your Dataset**: Place your dataset in the expected directory structure or update the configuration file (`config/config.yaml`) to match your dataset's location.

3. **Execute `main.py`**

This command runs the entire pipeline, from training the model with your dataset to evaluating its performance and visualizing the results.

## Contributing

Contributions to AquaSight are welcome! 

