# notes:

- save jpg of frames as their time stamp to make It easier to go back and cross-check

# object_detection

#### Prerequisites

Download all necessary libraries using the requirement.txt

* add clarifai_grpc
* add ultralytics
* 

source detectron_env/bin/activate

```
pip install -r requirments.txt
```

## pip install torch torchvision torchaudiomain.py

Run `main.py` by using a command line argument Inputting the video_path, frame_Interval, frame_location, relevant feature flags, and the output path for the JSON file.

#### Features

* Frame extraction from videos
* Multiple analysis capabilities:
  * Weapon detection
  * Text recognition (OCR)
  * Object detection
  * Emotion recognition
* Configurable frame interval
