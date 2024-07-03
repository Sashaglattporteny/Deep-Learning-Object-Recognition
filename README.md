# Deep Learning Final Project:  Real-time Object Detection using yolov8

This project was created by: 
BALTASAR CARRASCO IEWDIUKOW,
BENDIX SIBBEL,
GUILLERMO BRUN RUBIO,
IGNACIO FERRO,
MARIA JOSE PEREZ GOMEZ,
SASHA GLATT PORTENY,
VALERIA ABRAHAM LOPEZ

As a final project for: Deep Learning 

Description: This is a repository to use yolov8 to detect cars, buses, and trucks in a video of the streets of Madrid. 

![image](https://github.com/Sashaglattporteny/yolov8/assets/144905213/7872b86f-af24-424b-9353-61c752d2f110)

This code snippet is written in Python and uses several libraries (`cv2`, `pandas`, `ultralytics`, `cvzone`) to perform object detection and tracking on a video file. The object detection is carried out using a pre-trained YOLO (You Only Look Once) model, which is a popular method for real-time object detection. Here's a breakdown of the code:

1. Import Libraries:
   - `cv2`: OpenCV library for computer vision tasks.
   - `pandas`: Data manipulation library, here used to handle data frames.
   - `ultralytics`: A company's package that includes the YOLO model.
   - `cvzone`: Computer vision library for easy OpenCV functions.
   - `tracker`: A module that presumably contains a custom tracking class for tracking objects over frames.

2. Load YOLO Model:
   - The YOLO model is loaded with the weights file `'yolov8s.pt'`.

3. Mouse Position Function:
   - A function `RGB` is defined to print the mouse position whenever the mouse moves within a window named 'RGB'.

4. Setup OpenCV Window and Mouse Callback:
   - A named window 'RGB' is created, and the `RGB` function is set to be called whenever a mouse event occurs in this window.

5. Video Capture:
   - The video file `'tf.mp4'` is opened for processing.

6. Read Class Names:
   - Class names for detected objects are read from `'coco.txt'` and stored in a list called `class_list`.

7. Initialize Variables:
   - Counters for frames and different types of vehicles are initialized.
   - A `Tracker` object is created for tracking the vehicles.
   - Two lines (`cy1`, `cy2`) with an `offset` are defined, which the vehicles will cross.

8. Process Video Frames:
   - The video is processed frame by frame in a loop.
   - Every third frame is processed to reduce computational load.
   - Frames are resized for consistency.
   - The YOLO model predicts objects in the frame.
   - The detections are converted into a pandas DataFrame for easier manipulation.

9. Categorize Detected Objects:
   - Detected objects are categorized as cars, buses, or trucks based on the class names and their bounding boxes are stored.

10. Tracking:
    - The `Tracker` object updates the tracked bounding boxes for cars, buses, and trucks.

11. Draw Lines:
    - Two crossing lines are drawn on the frame to count the vehicles when they cross these lines.

12. Count Vehicles:
    - For each type of vehicle (car, bus, truck), if the center of the bounding box crosses the defined line within the specified offset, the respective counter is incremented.

13. Annotate Frames:
    - Bounding boxes and labels for each vehicle are drawn on the frames.
    - The `cvzone.putTextRect` function is used to put text on the frame.

14. Display Frame:
    - The annotated frame is displayed in the 'RGB' window.

15. Exit Condition:
    - If the 'Esc' key is pressed, the loop is broken and the program proceeds to termination.

16. Print Vehicle Counts:
    - After the video processing is complete, the total counts for each type of vehicle are printed.

17. Cleanup:
    - The video capture is released and all OpenCV windows are destroyed.

This script is essentially for a traffic monitoring application, where it counts the number of cars, buses, and trucks passing a certain line in the video. It demonstrates the use of computer vision and machine learning for real-world applications such as traffic analysis and vehicle tracking.

## How to Run the Script

To run the script and start detecting vehicles in your video, follow these steps:

### Step 1: Install Required Libraries

First, you need to install the required Python libraries. This project includes a `requirements.txt` file that lists all the necessary packages. You can install them using pip by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Video File

Ensure you have a video file named `tf.mp4` in the project's root directory. This video should be of the streets of Madrid or any other urban environment where you want to detect cars, buses, and trucks.

### Step 3: Run the Main Script
With the libraries installed and the video file in place, you can now run the main script. Open your terminal, navigate to the project's root directory, and execute the following command:

```bash
python main.py
```

This command will start the object detection process on the video file tf.mp4. Detected vehicles will be highlighted with bounding boxes, and their types (car, bus, or truck) will be annotated on the video output.

# Project Overview

This project is designed to leverage the power of YOLOv8 for object detection tasks, with a focus on vehicle detection. It is structured to support both evaluation of the model's performance and deployment of the model in a cloud environment. Below is a brief overview of the key components of this project:


## Evaluation Module

The evaluation module is located in the `evaluation/` directory. It contains scripts and configurations for assessing the performance of the YOLOv8 model on a dataset. The key components include:

- `evaluate.py`: This script uses the YOLOv8 model to perform validation on a specified dataset. It leverages the `ultralytics` library for model operations and `roboflow` for dataset management.
- `data.yaml`: Configuration file specifying the dataset paths and classes for evaluation.
- README files (`README.dataset.txt`, `README.roboflow.txt`): Provide additional information about the dataset and its source.

## Cloud Deployment Module

The cloud deployment module enables the model to be deployed as a service in a cloud environment, specifically designed for AWS SageMaker. The module is structured into two main directories:

- `cloud_deploy/model/`: Contains the `inference.py` script which defines the model loading and inference operations for deployment. It also includes a `requirements.txt` file specifying the necessary Python packages.
- `cloud_deploy/sagemaker_deploy/`: Contains the `deploy.py` script which handles the deployment of the model to AWS SageMaker, including setting up the endpoint for model inference.