# Radicalization Vision Pipeline

**Author:** Kimiya Shahamat  
**Course:** Professor Julia Hirschberg — Radicalization Research Final Report  
**Date:** May 11, 2025  

---

## Project Overview

This project develops a **multi-modal pipeline** for detecting **scene changes** and **visual cues** in videos containing radicalization or recruitment content. The system integrates **YOLOv8** object detection with **audio diarization** to identify visual and linguistic indicators of persuasion and narrative shifts.

---

## Notes

- Save `.jpg` frames using their **timestamp** to simplify manual cross-checking.  
- Referenced from: [earning_call repository](https://github.com/yyw28/earning_call/blob/main/transcript_output_v2.py)

---

## Object Detection

### Prerequisites

'''
source detectron_env/bin/activate
pip install -r requirements.txt
'''
Activate your environment and install dependencies:
python main.py --video_path path/to/video.mp4 \
               --frame_interval 5 \
               --frame_location ./frames \
               --detect_objects \
               --output_path ./results/detections.json
Features

Frame extraction from videos

Configurable frame interval

Multi-feature analysis:

Weapon detection

Text recognition (OCR)

Object detection (YOLOv8)

Emotion recognition

Model & Methodology
Visual Pipeline

Utilized YOLOv8 for high-accuracy object detection.

Outputs include bounding boxes [x1, y1, x2, y2], confidence, and class label.

Detections saved as JSON structures for downstream processing.

Scene Change Detection (KAFR)

Tracked objects across frames and computed centroid velocities.

Frames exceeding a velocity threshold (65) are flagged as scene changes.

Results exported as CSV for later analysis.

Audio Pipeline

Integrated speech segmentation and speaker diarization.

Aligned audio timestamps with visual detections.

Datasets & Benchmarking

Benchmarked on BBC Planet Earth and TSUNAMI datasets (scene-change ground truth).

Achieved ~72–75% accuracy on benchmark datasets.

Audiveris and MUSCIMA datasets used for Optical Music Recognition (OMR) symbol detection experiments.

Audiveris Dataset Overview:

Designed for OMR, converting music score images to symbolic notation.

Includes XML-based .omr project files and supports MusicXML 4.0 export.

Features real-world, multi-page scores with both template matching and neural network components for symbol recognition.

Open-source and cross-platform (Windows, Linux, macOS).


Future Work

Implement adaptive scene-change thresholds using CNN + Transformer fusion.

Add cross-modal attention for joint video–audio scene understanding.

Expand dataset coverage to extremist recruitment material for fine-tuning.

