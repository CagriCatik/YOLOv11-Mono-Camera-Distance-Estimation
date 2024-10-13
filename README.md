# YOLOv11 Mono-Camera Distance Estimation

This a Python-based machine learning project that combines depth estimation and object detection using pre-trained models. Building on Apple's [DepthPro](https://github.com/apple/ml-depth-pro), it offers tools to test depth and object detection on custom inputs efficiently.

---

## Features

- **Depth Estimation**: Predict depth maps from input images using pre-trained models.
- **Object Detection**: Perform object detection on test images or datasets.
- **Custom Input Support**: Easily integrate and test custom images or datasets.
- **Pre-trained Model Support**: Simple script provided to download pre-trained models.
- **Example Scripts**: Provided for testing and evaluating models.

---

## Installation

### Prerequisites

- **Python**: 3.8+
- **PyTorch**: Installation instructions can be found on the [PyTorch website](https://pytorch.org/get-started/locally/)
- Other dependencies listed in `requirements.txt`

### Steps

1. **Clone the Repository**

   Clone this project to your local machine:

   ```bash
   git clone https://github.com/CagriCatik/YOLOv11-Mono-Camera-Distance-Estimation
   cd YOLOv11-Mono-Camera-Distance-Estimation
   ```

2. **Install Dependencies**

   Install all required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**

   Run the provided script to download the necessary pre-trained models:

   ```bash
   chmod +x get_pretrained_models.sh
   ./get_pretrained_models.sh
   ```

   The models will be saved in the `checkpoints/` directory.

---

## Usage

### 1. **Depth Estimation**

   To run depth estimation on your input images, execute the following command:

   ```bash
   python test_depth.py
   ```

### 2. **Object Detection**

   To perform object detection using the pre-trained model, use the following command:

   ```bash
   python test_detection.py
   ```

### 3. **Depth Estimation and Object Detection Combined**

   Run both depth estimation and object detection in one go:

   ```bash
   python test_depth_detection.py
   ```

---

## Input and Output Structure

- **Input**: Place images in the `data_input/` directory for processing.
- **Output**: Results such as depth maps and detection outputs will be saved in the `data_output/` directory
