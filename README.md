# Defect Detection Library and GUI using Gradio

![Defect Detection Demo](assets/gradio_demo_screenshot.png)

## Introduction

Defect Detection GUI is a Gradio app that allows users to perform defect detection on images and videos using various defect detection methods given in the defect_detection library. The app provides a user-friendly graphical interface for easy interaction with the defect detection module. This repository contains the necessary code to run the app.

Defect detection is a crucial task in various industries, especially in manufacturing processes, where identifying defects in products is essential for ensuring quality control and minimizing waste.

## Getting Started

### Requirements

Before running the Defect Detection GUI, ensure that you have the following prerequisites:

- Python installed on your system.
- Required libraries:
  - Gradio
  - NumPy
  - 

### Installation

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Running the App

To run the Defect Detection GUI, follow these steps:

- Clone this repository to your local machine.
- Navigate to the directory containing the repository.
- Run the defect_detection_app.py Python script using the following command:

```bash
python defect_detection_app.py
```
The Gradio app will be launched in your default web browser. Follow the link in the terminal e.g. `http://127.0.0.1:7861` to open the GUI on your web browser.

## How to Use the Defect Detection GU
The Defect Detection GUI provides two tabs for processing **image** and **video** data. The app allows you to choose from different defect detection methods.

### Image Data Tab
- Upload an image that you want to analyze for defects by clicking on "Choose File" under "Input Image."

- Select a method from the dropdown list. The available methods are:
    - Edge Detector: Detect defects based on edges in the image.
    - Background Subtractor: Perform background subtraction to identify defects.
    - Gradient Threshold Detector: Detect defects using gradient thresholding.
    - Otsu Threshold Detector: Use Otsu's method for thresholding to find defects.
    - Kalman Residual Detector: Detect defects using Kalman filter residuals.
    - Object Detector: Detect defects using an object detection model.


- If you choose the method "Background Subtractor," you will need to provide a background image by clicking on "Choose File" under "Background Image." This step is necessary for this specific method.

- Check the "Preprocessing: Blackout non-foil region" checkbox if you want to perform a preprocessing step to blackout non-foil regions in the image.

- Click on "Process Image" to initiate the defect detection process.

- The output will be displayed below the input section, showing the defect image, defect map, and the time taken for the detection process.

### Video Data Tab

- Upload a video that you want to analyze for defects by clicking on "Choose File" under "Input Video."
- Check the "Debug" checkbox if you want to visualize the processed video using OpenCV HighGUI.
- Click on "Process Video" to initiate the defect detection process.
- The output will be displayed below the input section, showing the defect video and the defect mask video.

## Using the defect_detection Module as a Library
The defect_detector library provides classes for defect detection in images and videos. Below are the details of the classes available in the library:

### `ImageDefectDetector` Class

The `ImageDefectDetector` class is used to detect defects in an image using the specified method. The available defect detection methods are:

- `edge_detector`
- `background_subtractor`
- `gradient_threshold_detector`
- `otsu_thresh_detector`
- `kalman_residual_detector`
- `object_detector`

#### How to Use `ImageDefectDetector`

Import the ImageDefectDetector class from the defect_detector library.
```python
from defect_detector import ImageDefectDetector
```
Create an instance of the ImageDefectDetector class, specifying the desired defect detection method.
```python
defect_detector = ImageDefectDetector(method="edge_detector")
```
Use the detect_defects method to detect defects in an image.
```python
# Load the image as a NumPy array
image = cv2.imread("path_to_image.jpg")

# Detect defects in the image
defect_detection_image, defect_mask, timetaken = defect_detector.detect_defects(image)
```