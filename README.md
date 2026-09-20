# Raspberry Pi Vision AI Dashboard
A distributed Raspberry Pi 4 and GPU-based vision system integrating Real-ESRGAN, AnimeGANv2, CLIP, YOLO, SAM 2, and Gemma 3 for interactive visual perception, image enhancement, anime stylization, and multimodal visual analysis.

## Overview

This project provides an interactive web dashboard for visual perception using a Raspberry Pi 4 camera system connected to a GPU-enabled PC. The Raspberry Pi handles camera capture, image and video upload, media playback, camera controls, and the web interface, while the main PC performs GPU-based image processing and AI inference. The dashboard supports live camera input, uploaded images, and uploaded videos. Camera frames and selected video frames can be frozen and processed using Real-ESRGAN for image enhancement and AnimeGANv2 for anime-style conversion before being analyzed by CLIP, YOLO, SAM 2, and Gemma 3.

## Features

- Live OV5647 camera streaming from Raspberry Pi 4
- Manual camera controls for exposure, analogue gain, RGB gain, gamma, contrast, brightness, and JPEG quality
- Image upload and switching between camera and uploaded images
- Short video upload and browser-based playback
- Pause-frame analysis for uploaded videos
- Real-ESRGAN x2 image enhancement
- AnimeGANv2 anime-style image conversion
- Selectable AnimeGANv2 styles:
  - Paprika
  - Face Paint 512 v2
  - Face Paint 512 v1
  - CelebA Distill
- Optional processing pipeline where AnimeGANv2 receives the Real-ESRGAN-enhanced image when both are enabled
- CLIP-based image recognition and prompt similarity
- YOLO object detection with bounding-box visualization
- YOLO Pose human pose estimation with keypoint and skeleton visualization
- SAM 2 interactive point-based segmentation
- Multiple independent SAM segments with colored masks
- Gemma 3 vision-language model support for visual question answering
- AI analysis of frozen camera frames, uploaded images, and selected video frames
- Independent enable/disable controls for image enhancement and AI models
- Light and dark dashboard themes

## System Architecture

```text
OV5647 Camera / Uploaded Image / Uploaded Video
                    ↓
          Raspberry Pi 4 (Interface)
                    ↓
               Web Dashboard
                    ↓
             Selected Frame
                    ↓
            GPU PC (Backend)
                    ↓
       Real-ESRGAN x2 Enhancement
              [optional]
                    ↓
        AnimeGANv2 Stylization
              [optional]
                    ↓
              AI Inference
              ├── CLIP
              ├── YOLO
              ├── YOLO Pose
              ├── SAM 2
              └── Gemma 3 VLM
                    ↓
      Interactive visual perception results
```
