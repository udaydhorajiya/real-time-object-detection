# real-time-object-detection
Real-Time Object Detection System with Custom YOLO Model

This project is a complete, real-time object detection application built with Python. It features a user-friendly graphical interface created with Tkinter and leverages a custom-trained YOLO model to detect specific objects from a live webcam feed. The application is designed to be a powerful tool for both demonstrating the capabilities of custom computer vision and for practical, real-world monitoring tasks.

🟀Features:

Real-Time Webcam Detection: Processes a live video stream from a webcam to perform object detection in real-time.

Custom Model Integration: Automatically loads and uses a custom-trained best.pt model file if it is present in the project directory.

Model Comparison Tool: Allows the user to easily switch between a custom model and officially supported YOLO models (like yolov8n.pt and yolov8m.pt) to visually compare their performance.

Performance Monitoring: An FPS (Frames Per Second) counter is displayed directly on the video feed to provide an immediate measure of the detection speed.

User-Friendly GUI: A simple and intuitive interface built with Tkinter, requiring no command-line interaction to operate.

🟀Technology Stack:

Programming Language: Python

GUI Framework: Tkinter

Computer Vision: OpenCV

Deep Learning Framework: PyTorch

Object Detection Model: Ultralytics YOLO (custom-trained YOLOv10n, with support for other versions like YOLOv8)

Dataset Management: Roboflow

Cloud Training Platform: Kaggle Notebooks with an NVIDIA P100 GPU


Requirements

🟀Hardware:

Processor: Modern multi-core CPU (Intel i5 / AMD Ryzen 5 or better).

Memory: 8 GB RAM minimum (16 GB recommended).

Graphics Card: An NVIDIA GPU (e.g., RTX 3050 or better) is highly recommended for smooth, real-time performance.

Webcam: A standard internal or external webcam.


🟀Software:

Operating System: Windows, macOS, or Linux.

Python: Version 3.8 or newer.

Required Python libraries:

ultralytics

opencv-python

Pillow

Setup and Installation

🟀Clone the Repository:

git clone [https://your-repository-url.git](https://your-repository-url.git)
cd your-project-directory

🟀Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

🟀Install the Required Libraries:

pip install ultralytics opencv-python Pillow

🟀How to Run the Application:

Place Your Custom Model (Optional): If you have a custom-trained model, place the best.pt file in the root directory of the project.

🟀Run the Script: Execute the following command in your terminal from the project's root directory:

		  python object_detector.py


🟀Using the Application:

The main window will appear.

Click on one of the "Load Model" buttons to start the detection process with the selected model.

The live webcam feed will appear with bounding boxes drawn around any detected objects.

To switch models, click "Stop Detection," and then click a different "Load Model" button.

🟀For Developers:- Training a Custom Model:

This application is designed to work with a custom model trained using the ultralytics library. The process used for this project was:

Dataset Management: A large dataset of over 200,000 images was curated, preprocessed (resized to 416x416), and augmented using the Roboflow platform.

Cloud Training: The model was trained on Kaggle to leverage their free high-performance GPUs.

Export Model: The final best.pt model file was downloaded from the Kaggle training output and placed in this project's directory.
