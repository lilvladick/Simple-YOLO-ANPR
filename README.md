# Simple-YOLO-DeepSORT-ANPR

## Description
This project implements automatic detection and tracking of Russian license plates on video. The system is based on:

- YOLO: A fast and accurate object detection model used to find vehicles and license plates on each frame of a video.
- DeepSORT: An object tracking algorithm that provides stable tracking of detected license plates and cars in a video stream.

## Datasets used
Open datasets from the Roboflow platform were used to train YOLO models:

[Russian Plate dataset](https://universe.roboflow.com/plate-tsusp/russian-plate) — a dataset of Russian license plates

[Vehicles COCO](https://universe.roboflow.com/vehicle-mscoco/vehicles-coco/dataset/1) — a dataset of various vehicles

## Project objectives
- Detection of cars and license plates on each frame using YOLO models trained on the above datasets
- Tracking identified objects in a sequence of frames using DeepSORT, which allows for correct association of objects between frames
- Filtering and validation of detected license plates using regex to ensure accuracy of recognized plates
- Support for video stream processing, which makes the project suitable for working with video files or surveillance cameras

## How to run
- Install dependencies:
  ```python
  pip install -r requitements.txt
  ```
- Unpack the models in the models folder or add your own
- Change file paths in the `cars_and_plates_detector.py` script
- Run the script and wait for the result. The result will be in the file `output.mp4`
