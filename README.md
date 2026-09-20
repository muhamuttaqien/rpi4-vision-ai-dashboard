# Raspberry Pi Vision AI Dashboard
A distributed Raspberry Pi 4 and GPU-based vision system integrating CLIP, YOLO, SAM 2, and Gemma 3 vision-language models for interactive visual perception quality assessment.

## Overview

This project provides an interactive web dashboard for visual perception using a Raspberry Pi 4 camera system connected to a GPU-enabled PC. The Raspberry Pi handles camera capture, image streaming, image upload, and the web interface, while the main PC performs AI inference using multiple vision models.

## Features

- Live OV5647 camera streaming from Raspberry Pi 4
- Manual camera controls for exposure, gain, gamma, contrast, and brightness
- Image upload and switching between camera and uploaded images
- CLIP-based image recognition and prompt similarity
- YOLO object detection with bounding-box visualization
- SAM 2 interactive point-based segmentation
- Multiple independent SAM segments with colored masks
- Undo and clear-all segmentation controls
- Gemma 3 vision-language model support for visual question answering
- Light and dark dashboard themes

## System Architecture

```text
OV5647 Camera
      ↓
Raspberry Pi 4 (Interface)
      ↓
Web Dashboard
      ↓
GPU PC (Backend)
 ├── CLIP
 ├── YOLO
 ├── SAM 2
 └── Gemma 3 VLM
      ↓
Interactive visual perception results
```
