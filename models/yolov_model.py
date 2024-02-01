# models/yolov_model.py

from ultralytics import YOLO
import torch


def load_model(model_name='yolov8x.pt', device=None):
    """
    Load a YOLO model using the Ultralytics implementation.

    Parameters:
    - model_name (str): The name or path of the YOLO model file. Defaults to 'yolov8x.pt'.
    - device (torch.device): The device to load the model onto (e.g., 'cpu', 'cuda', 'mps').
      If None, the model is loaded onto the default device.

    Returns:
    - model (YOLO): An instance of the YOLO model loaded onto the specified device.
    """
    model = YOLO(model_name)
    if device:
        model.to(device)
    return model

