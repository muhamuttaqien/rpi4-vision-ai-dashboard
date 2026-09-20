#!/usr/bin/env python3

import subprocess
import threading
import time
import json
import cv2
import numpy as np

from flask import Flask, Response, request, jsonify


# ============================================================
# Configuration
# ============================================================

WIDTH = 640
HEIGHT = 480

FRAME_SIZE = WIDTH * HEIGHT * 2

VIDEO_DEVICE = "/dev/video0"
SENSOR_DEVICE = "/dev/v4l-subdev0"

PC_CLIP_URL = "http://192.168.0.16:5000"

SKIP_INITIAL_FRAMES = 3


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# Camera settings
# ============================================================

DEFAULT_SETTINGS = {
    "exposure": 500,
    "analogue_gain": 600,
    "gamma": 0.50,
    "blue_gain": 1.00,
    "green_gain": 1.00,
    "red_gain": 1.00,
    "contrast": 1.00,
    "brightness": 0,
    "jpeg_quality": 80
}

settings = DEFAULT_SETTINGS.copy()

settings_lock = threading.Lock()


# ============================================================
# Camera state
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

uploaded_image = None
uploaded_image_lock = threading.Lock()

running = True


# ============================================================
# V4L2 sensor controls
# ============================================================

def set_sensor_control(name, value):

    try:

        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                SENSOR_DEVICE,
                "--set-ctrl",
                f"{name}={value}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

    except Exception as e:

        print(
            f"Failed setting {name}: {e}"
        )


def apply_sensor_settings():

    with settings_lock:

        exposure = int(
            settings["exposure"]
        )

        analogue_gain = int(
            settings["analogue_gain"]
        )

    set_sensor_control(
        "exposure",
        exposure
    )

    set_sensor_control(
        "analogue_gain",
        analogue_gain
    )


# ============================================================
# Camera image processing
# ============================================================

def apply_gamma(image, gamma):

    gamma = max(
        float(gamma),
        0.01
    )

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in np.arange(256)
        ]
    ).astype("uint8")

    return cv2.LUT(
        image,
        table
    )


def process_raw_frame(raw):

    if len(raw) != FRAME_SIZE:
        return None

    # OV5647 GB10:
    # 16-bit little-endian storage containing 10-bit Bayer values.
    bayer10 = np.frombuffer(
        raw,
        dtype="<u2"
    ).reshape(
        HEIGHT,
        WIDTH
    )

    # Convert 10-bit 0...1023 to 8-bit 0...255.
    bayer8 = (
        bayer10.astype(np.uint32)
        * 255
        // 1023
    ).astype(np.uint8)

    # GBRG Bayer -> BGR.
    frame = cv2.cvtColor(
        bayer8,
        cv2.COLOR_BayerGB2BGR
    )

    with settings_lock:

        blue_gain = float(
            settings["blue_gain"]
        )

        green_gain = float(
            settings["green_gain"]
        )

        red_gain = float(
            settings["red_gain"]
        )

        gamma = float(
            settings["gamma"]
        )

        contrast = float(
            settings["contrast"]
        )

        brightness = int(
            settings["brightness"]
        )

    # Manual BGR gains.
    frame_float = frame.astype(
        np.float32
    )

    frame_float[:, :, 0] *= blue_gain
    frame_float[:, :, 1] *= green_gain
    frame_float[:, :, 2] *= red_gain

    frame = np.clip(
        frame_float,
        0,
        255
    ).astype(np.uint8)

    # Gamma.
    frame = apply_gamma(
        frame,
        gamma
    )

    # Contrast + brightness.
    frame = cv2.convertScaleAbs(
        frame,
        alpha=contrast,
        beta=brightness
    )

    # Mild denoise.
    blurred = cv2.GaussianBlur(
        frame,
        (3, 3),
        0
    )

    # Mild sharpening.
    frame = cv2.addWeighted(
        frame,
        1.4,
        blurred,
        -0.4,
        0
    )

    return frame


# ============================================================
# Camera capture thread
# ============================================================

def camera_loop():

    global latest_frame
    global running

    apply_sensor_settings()

    command = [
        "v4l2-ctl",
        "-d",
        VIDEO_DEVICE,
        f"--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=GB10",
        "--stream-mmap=3",
        "--stream-count=0",
        "--stream-to=-"
    ]

    while running:

        process = None

        try:

            print(
                "Starting OV5647 camera..."
            )

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=FRAME_SIZE * 4
            )

            skipped = 0

            while running:

                raw = process.stdout.read(
                    FRAME_SIZE
                )

                if not raw or len(raw) != FRAME_SIZE:

                    raise RuntimeError(
                        "Camera stream stopped."
                    )

                if skipped < SKIP_INITIAL_FRAMES:

                    skipped += 1
                    continue

                frame = process_raw_frame(
                    raw
                )

                if frame is None:
                    continue

                with frame_lock:

                    latest_frame = frame

        except Exception as e:

            print(
                f"Camera error: {e}"
            )

            time.sleep(1)

        finally:

            if process is not None:

                try:
                    process.terminate()
                except Exception:
                    pass

                try:
                    process.wait(
                        timeout=2
                    )
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass


# ============================================================
# MJPEG generator
# ============================================================

def generate_mjpeg():

    while running:

        with frame_lock:

            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)
            continue

        with settings_lock:

            quality = int(
                settings["jpeg_quality"]
            )

        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality
            ]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        time.sleep(0.03)


# ============================================================
# Main page
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Raspberry Pi 4 - OV5647</title>

<style>

:root {
    --bg: #f4f4f4;
    --panel: #ffffff;
    --text: #222222;
    --secondary: #666666;
    --border: #dddddd;
    --input: #ffffff;
    --button: #eeeeee;
}

body.dark {
    --bg: #171717;
    --panel: #252525;
    --text: #eeeeee;
    --secondary: #bbbbbb;
    --border: #444444;
    --input: #303030;
    --button: #383838;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 25px;
    background: var(--bg);
    color: var(--text);
    font-family: Arial, sans-serif;
    transition: background 0.2s, color 0.2s;
}

.container {
    max-width: 900px;
    margin: auto;
}

h1 {
    text-align: center;
    margin-top: 0;
    margin-bottom: 25px;
}

.camera {
    position: relative;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 20px;
    text-align: center;
}

#yoloOverlay {
    position: absolute;
    pointer-events: none;
    z-index: 10;
}

#samOverlay {
    position: absolute;
    pointer-events: none;
    z-index: 9;
}

.camera img {
    width: 100%;
    max-width: 800px;
    display: block;
    margin: auto;
}

.panel,
.clip-panel,
.clip-test-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 18px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.section-header .section {
    margin-bottom: 0;
}

.toggle-switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
}

.toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.toggle-slider {
    position: absolute;
    inset: 0;
    cursor: pointer;
    background: #aeb4b0;
    border-radius: 22px;
    transition: 0.22s ease;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
}

.toggle-slider::before {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    left: 3px;
    top: 3px;
    background: #ffffff;
    border-radius: 50%;
    transition: 0.22s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.28);
}

.toggle-switch input:checked + .toggle-slider {
    background: #2e9b55;
}

.toggle-switch input:checked + .toggle-slider::before {
    transform: translateX(18px);
}

.toggle-switch input:focus-visible + .toggle-slider {
    outline: 2px solid rgba(79,127,98,0.28);
    outline-offset: 2px;
}

.section {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 16px;
}

.control {
    display: grid;
    grid-template-columns: 150px 1fr 70px;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.control:last-child {
    margin-bottom: 0;
}

.control label {
    font-size: 14px;
}

.control input[type="range"] {
    width: 100%;
}

.value {
    text-align: right;
    font-family: monospace;
}

.buttons {
    margin-top: 18px;
    display: flex;
    gap: 10px;
}

button {
    border: 1px solid var(--border);
    background: var(--button);
    color: var(--text);
    padding: 9px 15px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
}

button:hover {
    filter: brightness(0.95);
}

#clip_best {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 12px;
}

.clip-result {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
}

.clip-result:last-child {
    border-bottom: none;
}

.clip-status {
    color: var(--secondary);
    font-size: 13px;
    text-align: center;
    margin-top: 8px;
}

.prompt-row {
    display: flex;
    gap: 10px;
}

#promptInput {
    flex: 1;
    border: 1px solid var(--border);
    background: var(--input);
    color: var(--text);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 15px;
}

#promptResult {
    margin-top: 16px;
    text-align: center;
}

.prompt-name {
    font-size: 15px;
    color: var(--secondary);
    margin-bottom: 6px;
}

.prompt-score {
    font-size: 28px;
    font-weight: bold;
}

.prompt-note {
    margin-top: 8px;
    color: var(--secondary);
    font-size: 12px;
}

#gemmaPrompt {
    width: 100%;
    min-height: 82px;
    resize: vertical;
    border: 1px solid var(--border);
    background: var(--input);
    color: var(--text);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 15px;
    font-family: inherit;
    line-height: 1.45;
}

#gemmaPrompt:focus {
    outline: 2px solid rgba(79,127,98,0.22);
    outline-offset: 1px;
}

.gemma-answer {
    margin-top: 14px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    background: var(--input);
    border-radius: 6px;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.gemma-answer.empty {
    color: var(--secondary);
}

.gemma-actions {
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.gemma-status {
    color: var(--secondary);
    font-size: 12px;
}



.sam-object-list {
    margin-top: 12px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
}

.sam-object-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 5px 0;
    font-size: 13px;
}

.sam-object-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.sam-color-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.sam-object-score {
    color: var(--secondary);
    font-family: monospace;
}

.source-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 14px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.source-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.source-status {
    margin-left: auto;
    color: var(--secondary);
    font-size: 13px;
}

#imageUpload {
    display: none;
}

.theme-button {
    position: fixed;
    right: 20px;
    bottom: 20px;
    border-radius: 50px;
    padding: 10px 15px;
    z-index: 1000;
}

@media (max-width: 650px) {

    body {
        padding: 12px;
    }

    .control {
        grid-template-columns: 110px 1fr 55px;
        gap: 8px;
    }

    .prompt-row {
        flex-direction: column;
    }

}

</style>
</head>


<body>

<div class="container">

<h1>Raspberry Pi 4 - OV5647</h1>


<div class="camera">
    <img id="cameraStream" src="/video">
    <canvas id="samOverlay"></canvas>
    <canvas id="yoloOverlay"></canvas>
</div>

<div class="source-panel">
<div class="source-row">
    <input
        id="imageUpload"
        type="file"
        accept="image/*"
        onchange="uploadDisplayImage(this)">

    <button onclick="document.getElementById('imageUpload').click()">
        Upload Image
    </button>

    <button
        id="backToCameraButton"
        onclick="backToCamera()"
        disabled>
        Back to Camera
    </button>

    <span
        id="sourceStatus"
        class="source-status">
        Camera stream
    </span>
</div>
</div>


<!-- ===================================================== -->
<!-- CLIP Recognition                                      -->
<!-- ===================================================== -->

<div class="clip-panel">

<div class="section-header">
<div class="section">
CLIP Recognition
</div>
<label class="toggle-switch" title="Enable or disable CLIP">
    <input
        id="clipToggle"
        type="checkbox"
        checked
        onchange="toggleModel('clip')">
    <span class="toggle-slider"></span>
</label>
</div>

<div id="clip_best">
Waiting for CLIP...
</div>

<div id="clip_scores">
</div>

<div
    id="clip_status"
    class="clip-status">
</div>

</div>


<!-- ===================================================== -->
<!-- CLIP Prompt Test                                      -->
<!-- ===================================================== -->

<div class="clip-test-panel">

<div class="section">
CLIP Prompt Test
</div>

<div class="prompt-row">

<input
    id="promptInput"
    type="text"
    placeholder="Enter a prompt, e.g. a book">

<button
    id="promptButton"
    onclick="testClipPrompt()">
Test Prompt
</button>

</div>

<div id="promptResult">

<div class="prompt-name">
Enter a prompt and compare it with the current camera image.
</div>

</div>

</div>


<!-- ===================================================== -->
<!-- YOLO Detection                                        -->
<!-- ===================================================== -->

<div class="clip-panel">

<div class="section-header">
<div class="section">
YOLO Detection
</div>
<label class="toggle-switch" title="Enable or disable YOLO">
    <input
        id="yoloToggle"
        type="checkbox"
        checked
        onchange="toggleModel('yolo')">
    <span class="toggle-slider"></span>
</label>
</div>

<div id="yolo_status">
Waiting for YOLO...
</div>

<div id="yolo_detections">
</div>

</div>


<!-- ===================================================== -->
<!-- SAM 2 Segmentation                                    -->
<!-- ===================================================== -->

<div class="clip-panel">

<div class="section-header">
<div class="section">
SAM 2 Segmentation
</div>
<label class="toggle-switch" title="Enable or disable SAM 2">
    <input
        id="samToggle"
        type="checkbox"
        checked
        onchange="toggleModel('sam')">
    <span class="toggle-slider"></span>
</label>
</div>

<div id="sam_status" class="clip-status">
Click objects in the camera image to segment them.
</div>

<div id="sam_result" class="prompt-note"></div>

<div
    id="sam_object_list"
    class="sam-object-list"
    style="display:none;">
</div>

<div class="buttons">
<button onclick="undoLastSamMask()">
Undo Last
</button>
<button onclick="clearSamMask()">
Clear All Masks
</button>
</div>

</div>


<!-- ===================================================== -->
<!-- Gemma 3 Vision                                       -->
<!-- ===================================================== -->

<div class="clip-panel">

<div class="section-header">
<div class="section">
Gemma 3 Vision
</div>
<label class="toggle-switch" title="Enable or disable Gemma 3 Vision">
    <input
        id="gemmaToggle"
        type="checkbox"
        checked
        onchange="toggleModel('gemma')">
    <span class="toggle-slider"></span>
</label>
</div>

<textarea
    id="gemmaPrompt"
    placeholder="Ask about the current image, e.g. What do you see in the image?"></textarea>

<div class="gemma-actions">
<button
    id="gemmaAskButton"
    onclick="askGemma()">
Ask
</button>

<button
    id="gemmaClearButton"
    onclick="clearGemmaResult()">
Clear
</button>

<span
    id="gemma_status"
    class="gemma-status">
Ready
</span>
</div>

<div
    id="gemma_answer"
    class="gemma-answer empty">
Ask Gemma 3 a question about the current displayed image.
</div>

</div>


<div class="panel">

<div class="section">
Camera Settings
</div>


<div class="control">
<label>Exposure</label>
<input
    id="exposure"
    type="range"
    min="4"
    max="500"
    step="1"
    oninput="updateSetting(this)">
<span
    class="value"
    id="exposure_value">
</span>
</div>


<div class="control">
<label>Analogue Gain</label>
<input
    id="analogue_gain"
    type="range"
    min="16"
    max="1023"
    step="1"
    oninput="updateSetting(this)">
<span
    class="value"
    id="analogue_gain_value">
</span>
</div>


<div class="control">
<label>Gamma</label>
<input
    id="gamma"
    type="range"
    min="0.10"
    max="2.00"
    step="0.05"
    oninput="updateSetting(this)">
<span
    class="value"
    id="gamma_value">
</span>
</div>


<div class="control">
<label>Blue Gain</label>
<input
    id="blue_gain"
    type="range"
    min="0.10"
    max="3.00"
    step="0.05"
    oninput="updateSetting(this)">
<span
    class="value"
    id="blue_gain_value">
</span>
</div>


<div class="control">
<label>Green Gain</label>
<input
    id="green_gain"
    type="range"
    min="0.10"
    max="3.00"
    step="0.05"
    oninput="updateSetting(this)">
<span
    class="value"
    id="green_gain_value">
</span>
</div>


<div class="control">
<label>Red Gain</label>
<input
    id="red_gain"
    type="range"
    min="0.10"
    max="3.00"
    step="0.05"
    oninput="updateSetting(this)">
<span
    class="value"
    id="red_gain_value">
</span>
</div>


<div class="control">
<label>Contrast</label>
<input
    id="contrast"
    type="range"
    min="0.50"
    max="2.00"
    step="0.05"
    oninput="updateSetting(this)">
<span
    class="value"
    id="contrast_value">
</span>
</div>


<div class="control">
<label>Brightness</label>
<input
    id="brightness"
    type="range"
    min="-100"
    max="100"
    step="1"
    oninput="updateSetting(this)">
<span
    class="value"
    id="brightness_value">
</span>
</div>


<div class="control">
<label>JPEG Quality</label>
<input
    id="jpeg_quality"
    type="range"
    min="30"
    max="100"
    step="1"
    oninput="updateSetting(this)">
<span
    class="value"
    id="jpeg_quality_value">
</span>
</div>


<div class="buttons">

<button onclick="resetSettings()">
Reset
</button>

</div>

</div>


</div>


<button
    class="theme-button"
    onclick="toggleTheme()"
    id="themeButton">
🌙
</button>


<script>

const PC_CLIP_URL =
    "http://192.168.0.16:5000";


let clipEnabled = true;
let yoloEnabled = true;
let samEnabled = true;
let gemmaEnabled = true;
let displayMode = "camera";
let sourceWidth = 640;
let sourceHeight = 480;

let samSegments = [];

const samColors = [
    [45, 125, 220],
    [255, 99, 71],
    [46, 160, 67],
    [180, 80, 210],
    [255, 165, 0],
    [0, 180, 180],
    [220, 70, 140],
    [140, 110, 60]
];

function updateModelButtons() {

    const clipButton =
        document.getElementById(
            "clipToggle"
        );

    const yoloButton =
        document.getElementById(
            "yoloToggle"
        );

    const samButton =
        document.getElementById(
            "samToggle"
        );

    const gemmaButton =
        document.getElementById(
            "gemmaToggle"
        );

    clipButton.checked = clipEnabled;
    yoloButton.checked = yoloEnabled;
    samButton.checked = samEnabled;
    gemmaButton.checked = gemmaEnabled;

    document.getElementById(
        "cameraStream"
    ).style.cursor =
        samEnabled
        ? "crosshair"
        : "default";

    const promptInput =
        document.getElementById(
            "promptInput"
        );

    const promptButton =
        document.getElementById(
            "promptButton"
        );

    promptInput.disabled =
        !clipEnabled;

    promptButton.disabled =
        !clipEnabled;

    const gemmaPrompt =
        document.getElementById(
            "gemmaPrompt"
        );

    const gemmaAskButton =
        document.getElementById(
            "gemmaAskButton"
        );

    gemmaPrompt.disabled =
        !gemmaEnabled;

    gemmaAskButton.disabled =
        !gemmaEnabled;
}


async function loadModelState() {

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/control"
            );

        const data =
            await response.json();

        clipEnabled =
            Boolean(data.clip_enabled);

        yoloEnabled =
            Boolean(data.yolo_enabled);

        samEnabled =
            Boolean(data.sam_enabled);

        gemmaEnabled =
            Boolean(data.gemma_enabled);

        updateModelButtons();

    } catch (error) {

        console.error(
            "Could not load model state:",
            error
        );

    }
}


async function toggleModel(modelName) {

    const nextClip =
        modelName === "clip"
        ? !clipEnabled
        : clipEnabled;

    const nextYolo =
        modelName === "yolo"
        ? !yoloEnabled
        : yoloEnabled;

    const nextSam =
        modelName === "sam"
        ? !samEnabled
        : samEnabled;

    const nextGemma =
        modelName === "gemma"
        ? !gemmaEnabled
        : gemmaEnabled;

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/control",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        clip_enabled: nextClip,
                        yolo_enabled: nextYolo,
                        sam_enabled: nextSam,
                        gemma_enabled: nextGemma
                    })
                }
            );

        const data =
            await response.json();

        clipEnabled =
            Boolean(data.clip_enabled);

        yoloEnabled =
            Boolean(data.yolo_enabled);

        samEnabled =
            Boolean(data.sam_enabled);

        gemmaEnabled =
            Boolean(data.gemma_enabled);

        updateModelButtons();

        if (!clipEnabled) {

            document.getElementById(
                "clip_best"
            ).textContent =
                "CLIP disabled";

            document.getElementById(
                "clip_scores"
            ).innerHTML = "";

            document.getElementById(
                "clip_status"
            ).textContent = "";

            document.getElementById(
                "promptResult"
            ).innerHTML =
                '<div class="prompt-name">CLIP is disabled.</div>';
        }

        if (!yoloEnabled) {

            document.getElementById(
                "yolo_status"
            ).textContent =
                "YOLO disabled";

            document.getElementById(
                "yolo_detections"
            ).innerHTML = "";

            drawYoloBoxes([]);
        }

        if (!samEnabled) {

            document.getElementById(
                "sam_status"
            ).textContent =
                "SAM disabled";

            document.getElementById(
                "sam_result"
            ).textContent = "";

            clearSamMask(false);

        } else if (modelName === "sam") {

            document.getElementById(
                "sam_status"
            ).textContent =
                displayMode === "upload"
                ? "Click objects in the uploaded image to segment them."
                : "Click objects in the camera image to segment them.";
        }

        if (!gemmaEnabled) {

            document.getElementById(
                "gemma_status"
            ).textContent =
                "Gemma 3 disabled";

        } else if (modelName === "gemma") {

            document.getElementById(
                "gemma_status"
            ).textContent =
                "Ready";
        }

    } catch (error) {

        console.error(
            "Could not change model state:",
            error
        );

    }
}



/* ========================================================
   Display source: camera stream / uploaded image
   ======================================================== */

async function uploadDisplayImage(input) {

    if (!input.files || input.files.length === 0) {
        return;
    }

    const file = input.files[0];

    const status =
        document.getElementById(
            "sourceStatus"
        );

    status.textContent =
        "Uploading and processing...";

    try {

        const piForm =
            new FormData();

        piForm.append(
            "image",
            file
        );

        const piResponse =
            await fetch(
                "/upload_image",
                {
                    method: "POST",
                    body: piForm
                }
            );

        const piData =
            await piResponse.json();

        if (!piResponse.ok) {

            throw new Error(
                piData.error
                || "Raspberry Pi upload failed."
            );
        }

        const pcForm =
            new FormData();

        pcForm.append(
            "image",
            file
        );

        const pcResponse =
            await fetch(
                PC_CLIP_URL
                + "/source/upload",
                {
                    method: "POST",
                    body: pcForm
                }
            );

        const pcData =
            await pcResponse.json();

        if (!pcResponse.ok) {

            throw new Error(
                pcData.error
                || "PC image processing failed."
            );
        }

        const img =
            document.getElementById(
                "cameraStream"
            );

        displayMode =
            "upload";

        sourceWidth =
            Number(pcData.width)
            || Number(piData.width)
            || 640;

        sourceHeight =
            Number(pcData.height)
            || Number(piData.height)
            || 480;

        img.src =
            "/uploaded_image?t="
            + Date.now();

        document.getElementById(
            "backToCameraButton"
        ).disabled =
            false;

        status.textContent =
            "Uploaded image: "
            + file.name;

        clearSamMask(false);
        drawYoloBoxes([]);
        clearGemmaResult(false);

        document.getElementById(
            "sam_status"
        ).textContent =
            samEnabled
            ? "Click objects in the uploaded image to segment them."
            : "SAM disabled";

        updateModelButtons();

        updateClip();
        updateYolo();
        updateYoloOverlay();

    } catch (error) {

        status.textContent =
            "Upload failed: "
            + error.message;
    }

    input.value = "";
}


async function backToCamera() {

    const status =
        document.getElementById(
            "sourceStatus"
        );

    status.textContent =
        "Switching to camera...";

    try {

        const response =
            await fetch(
                PC_CLIP_URL
                + "/source/camera",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error
                || "Could not switch PC to camera."
            );
        }

        const img =
            document.getElementById(
                "cameraStream"
            );

        displayMode =
            "camera";

        sourceWidth =
            Number(data.width)
            || 640;

        sourceHeight =
            Number(data.height)
            || 480;

        img.src =
            "/video?t="
            + Date.now();

        document.getElementById(
            "backToCameraButton"
        ).disabled =
            true;

        status.textContent =
            "Camera stream";

        clearSamMask(false);
        drawYoloBoxes([]);
        clearGemmaResult(false);

        document.getElementById(
            "sam_status"
        ).textContent =
            samEnabled
            ? "Click objects in the camera image to segment them."
            : "SAM disabled";

        updateModelButtons();

        updateClip();
        updateYolo();
        updateYoloOverlay();

    } catch (error) {

        status.textContent =
            "Could not return to camera: "
            + error.message;
    }
}


/* ========================================================
   Camera settings
   ======================================================== */

async function loadSettings() {

    try {

        const response =
            await fetch("/settings");

        const data =
            await response.json();

        for (const [key, value]
             of Object.entries(data)) {

            const input =
                document.getElementById(key);

            const label =
                document.getElementById(
                    key + "_value"
                );

            if (input) {

                input.value =
                    value;

            }

            if (label) {

                label.textContent =
                    value;

            }

        }

    } catch (error) {

        console.error(
            "Could not load settings:",
            error
        );

    }

}


let settingsTimer = null;


function updateSetting(element) {

    const key =
        element.id;

    const value =
        element.value;

    const label =
        document.getElementById(
            key + "_value"
        );

    if (label) {

        label.textContent =
            value;

    }

    clearTimeout(
        settingsTimer
    );

    settingsTimer =
        setTimeout(
            async () => {

                try {

                    await fetch(
                        "/settings",
                        {
                            method: "POST",
                            headers: {
                                "Content-Type":
                                    "application/json"
                            },
                            body: JSON.stringify({
                                [key]: value
                            })
                        }
                    );

                } catch (error) {

                    console.error(
                        error
                    );

                }

            },
            100
        );

}


async function resetSettings() {

    try {

        await fetch(
            "/reset",
            {
                method: "POST"
            }
        );

        await loadSettings();

    } catch (error) {

        console.error(
            error
        );

    }

}


/* ========================================================
   CLIP Recognition
   ======================================================== */

async function updateClip() {

    if (!clipEnabled) {
        return;
    }

    const best =
        document.getElementById(
            "clip_best"
        );

    const scoresContainer =
        document.getElementById(
            "clip_scores"
        );

    const status =
        document.getElementById(
            "clip_status"
        );

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/clip"
            );

        if (!response.ok) {

            throw new Error(
                "HTTP " + response.status
            );

        }

        const data =
            await response.json();

        clipEnabled =
            data.enabled !== false;

        updateModelButtons();

        if (!clipEnabled) {

            best.textContent =
                "CLIP disabled";

            scoresContainer.innerHTML =
                "";

            return;
        }


        if (!data.best) {

            best.textContent =
                "Waiting for camera...";

            scoresContainer.innerHTML =
                "";

            return;

        }


        best.textContent =
            "Best: " + data.best;


        scoresContainer.innerHTML =
            "";


        data.scores.forEach(
            item => {

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "clip-result";

                const label =
                    document.createElement(
                        "span"
                    );

                const score =
                    document.createElement(
                        "span"
                    );

                label.textContent =
                    item.label;

                score.textContent =
                    (
                        item.score * 100
                    ).toFixed(1)
                    + "%";

                row.appendChild(
                    label
                );

                row.appendChild(
                    score
                );

                scoresContainer.appendChild(
                    row
                );

            }
        );


        status.textContent =
            "";

    } catch (error) {

        best.textContent =
            "CLIP PC unavailable";

        scoresContainer.innerHTML =
            "";

        status.textContent =
            "Could not reach "
            + PC_CLIP_URL;

    }

}


/* ========================================================
   CLIP Prompt Test
   ======================================================== */

async function testClipPrompt() {

    if (!clipEnabled) {
        return;
    }

    const input =
        document.getElementById(
            "promptInput"
        );

    const result =
        document.getElementById(
            "promptResult"
        );

    const button =
        document.getElementById(
            "promptButton"
        );

    const prompt =
        input.value.trim();


    if (!prompt) {

        result.innerHTML =
            '<div class="prompt-name">'
            + 'Please enter a prompt.'
            + '</div>';

        return;

    }


    button.disabled =
        true;

    button.textContent =
        "Testing...";


    result.innerHTML =
        '<div class="prompt-name">'
        + 'Computing CLIP similarity...'
        + '</div>';


    try {

        const response =
            await fetch(
                PC_CLIP_URL
                + "/clip_prompt?prompt="
                + encodeURIComponent(prompt)
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error
                || "Prompt test failed."
            );

        }


        const similarity =
            Number(
                data.similarity
            );


        result.innerHTML =
            '<div class="prompt-name">'
            + escapeHtml(data.prompt)
            + '</div>'
            + '<div class="prompt-score">'
            + similarity.toFixed(4)
            + '</div>'
            + '<div class="prompt-note">'
            + 'Cosine similarity between the text prompt '
            + 'and the current displayed image'
            + '</div>';


    } catch (error) {

        result.innerHTML =
            '<div class="prompt-name">'
            + 'CLIP prompt test unavailable'
            + '</div>'
            + '<div class="prompt-note">'
            + escapeHtml(error.message)
            + '</div>';

    } finally {

        button.disabled =
            false;

        button.textContent =
            "Test Prompt";

    }

}


document
    .getElementById(
        "promptInput"
    )
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
            ) {

                testClipPrompt();

            }

        }
    );


function escapeHtml(text) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        text;

    return div.innerHTML;

}


/* ========================================================
   Gemma 3 Vision
   ======================================================== */

function clearGemmaResult(updateStatus = true) {

    const answer =
        document.getElementById(
            "gemma_answer"
        );

    answer.textContent =
        "Ask Gemma 3 a question about the current displayed image.";

    answer.classList.add(
        "empty"
    );

    if (updateStatus) {

        document.getElementById(
            "gemma_status"
        ).textContent =
            gemmaEnabled
            ? "Ready"
            : "Gemma 3 disabled";
    }
}


async function askGemma() {

    if (!gemmaEnabled) {
        return;
    }

    const input =
        document.getElementById(
            "gemmaPrompt"
        );

    const button =
        document.getElementById(
            "gemmaAskButton"
        );

    const status =
        document.getElementById(
            "gemma_status"
        );

    const answer =
        document.getElementById(
            "gemma_answer"
        );

    const prompt =
        input.value.trim();

    if (!prompt) {

        status.textContent =
            "Please enter a question.";

        return;
    }

    button.disabled =
        true;

    button.textContent =
        "Thinking...";

    status.textContent =
        "Analyzing current image...";

    answer.textContent =
        "Generating response...";

    answer.classList.add(
        "empty"
    );

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/gemma",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        prompt: prompt
                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error
                || "Gemma request failed."
            );
        }

        answer.textContent =
            data.answer
            || "No response generated.";

        answer.classList.remove(
            "empty"
        );

        status.textContent =
            data.source === "upload"
            ? "Answered from uploaded image"
            : "Answered from camera frame";

    } catch (error) {

        answer.textContent =
            "Gemma 3 Vision unavailable";

        answer.classList.add(
            "empty"
        );

        status.textContent =
            error.message;

    } finally {

        button.disabled =
            !gemmaEnabled;

        button.textContent =
            "Ask";
    }
}


document
    .getElementById(
        "gemmaPrompt"
    )
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
                && (event.ctrlKey || event.metaKey)
            ) {

                event.preventDefault();
                askGemma();
            }
        }
    );


/* ========================================================
   YOLO Detection
   ======================================================== */

async function updateYolo() {

    if (!yoloEnabled) {
        return;
    }

    const status =
        document.getElementById(
            "yolo_status"
        );

    const container =
        document.getElementById(
            "yolo_detections"
        );

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/yolo"
            );

        if (!response.ok) {

            throw new Error(
                "HTTP " + response.status
            );

        }

        const data =
            await response.json();

        yoloEnabled =
            data.enabled !== false;

        updateModelButtons();

        if (!yoloEnabled) {

            status.textContent =
                "YOLO disabled";

            container.innerHTML =
                "";

            drawYoloBoxes([]);

            return;
        }

        const detections =
            data.detections || [];

        container.innerHTML =
            "";

        if (detections.length === 0) {

            status.textContent =
                "No objects detected";

            return;

        }

        status.textContent =
            "";

        detections.forEach(
            item => {

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "clip-result";

                const label =
                    document.createElement(
                        "span"
                    );

                const score =
                    document.createElement(
                        "span"
                    );

                label.textContent =
                    item.label;

                score.textContent =
                    (
                        item.confidence * 100
                    ).toFixed(1)
                    + "%";

                row.appendChild(
                    label
                );

                row.appendChild(
                    score
                );

                container.appendChild(
                    row
                );

            }
        );

    } catch (error) {

        status.textContent =
            "YOLO PC unavailable";

        container.innerHTML =
            "";

    }

}



/* ========================================================
   YOLO Bounding Box Overlay
   ======================================================== */

function drawYoloBoxes(
    detections,
    inputWidth = sourceWidth,
    inputHeight = sourceHeight
) {

    const img =
        document.getElementById(
            "cameraStream"
        );

    const canvas =
        document.getElementById(
            "yoloOverlay"
        );

    if (!img || !canvas) {
        return;
    }

    const ctx =
        canvas.getContext(
            "2d"
        );

    const displayWidth =
        img.clientWidth;

    const displayHeight =
        img.clientHeight;

    if (
        displayWidth === 0
        || displayHeight === 0
    ) {
        return;
    }

    /*
       YOLO coordinates use the active source dimensions.
       Camera mode is normally 640 x 480, while uploaded
       images may have any resolution.
    */

    const safeSourceWidth =
        inputWidth > 0
        ? inputWidth
        : 640;

    const safeSourceHeight =
        inputHeight > 0
        ? inputHeight
        : 480;

    const scaleX =
        displayWidth / safeSourceWidth;

    const scaleY =
        displayHeight / safeSourceHeight;


    /*
       Position the canvas exactly over the
       displayed camera image.
    */

    canvas.style.left =
        img.offsetLeft + "px";

    canvas.style.top =
        img.offsetTop + "px";

    canvas.style.width =
        displayWidth + "px";

    canvas.style.height =
        displayHeight + "px";


    /*
       Canvas drawing coordinates use the
       actual displayed image dimensions.
    */

    canvas.width =
        displayWidth;

    canvas.height =
        displayHeight;


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    detections.forEach(
        detection => {

            if (
                !detection.bbox
                || detection.bbox.length !== 4
            ) {
                return;
            }


            const [
                x1,
                y1,
                x2,
                y2
            ] = detection.bbox;


            const x =
                x1 * scaleX;

            const y =
                y1 * scaleY;

            const width =
                (x2 - x1) * scaleX;

            const height =
                (y2 - y1) * scaleY;


            const label =
                detection.label;

            const confidence =
                (
                    detection.confidence
                    * 100
                ).toFixed(1)
                + "%";


            /*
               Bounding box
            */

            ctx.strokeStyle =
                "#00ff66";

            ctx.lineWidth =
                3;

            ctx.strokeRect(
                x,
                y,
                width,
                height
            );


            /*
               Label
            */

            const text =
                label
                + " "
                + confidence;

            ctx.font =
                "16px Arial";

            const textWidth =
                ctx.measureText(
                    text
                ).width;

            const labelHeight =
                24;


            let labelY =
                y - labelHeight;

            if (labelY < 0) {
                labelY = y;
            }


            ctx.fillStyle =
                "rgba(0, 0, 0, 0.75)";

            ctx.fillRect(
                x,
                labelY,
                textWidth + 12,
                labelHeight
            );


            ctx.fillStyle =
                "#00ff66";

            ctx.fillText(
                text,
                x + 6,
                labelY + 17
            );

        }
    );
}


async function updateYoloOverlay() {

    if (!yoloEnabled) {
        drawYoloBoxes([]);
        return;
    }

    try {

        const response =
            await fetch(
                PC_CLIP_URL
                + "/yolo"
            );

        if (!response.ok) {
            throw new Error(
                "HTTP "
                + response.status
            );
        }

        const data =
            await response.json();

        sourceWidth =
            Number(data.width)
            || sourceWidth;

        sourceHeight =
            Number(data.height)
            || sourceHeight;

        drawYoloBoxes(
            data.detections || [],
            sourceWidth,
            sourceHeight
        );

    } catch (error) {

        drawYoloBoxes([]);

    }

}


/* ========================================================
   SAM 2 Segmentation
   ======================================================== */

function prepareSamCanvas() {

    const img =
        document.getElementById(
            "cameraStream"
        );

    const canvas =
        document.getElementById(
            "samOverlay"
        );

    const width = img.clientWidth;
    const height = img.clientHeight;

    if (width === 0 || height === 0) {
        return null;
    }

    canvas.style.left =
        img.offsetLeft + "px";

    canvas.style.top =
        img.offsetTop + "px";

    canvas.style.width =
        width + "px";

    canvas.style.height =
        height + "px";

    canvas.width = width;
    canvas.height = height;

    return canvas;
}



function updateSamObjectList() {

    const container =
        document.getElementById(
            "sam_object_list"
        );

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (samSegments.length === 0) {

        container.style.display =
            "none";

        return;
    }

    container.style.display =
        "block";

    samSegments.forEach(
        (segment, index) => {

            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "sam-object-row";

            const left =
                document.createElement(
                    "div"
                );

            left.className =
                "sam-object-left";

            const dot =
                document.createElement(
                    "span"
                );

            dot.className =
                "sam-color-dot";

            const color =
                samColors[
                    index % samColors.length
                ];

            dot.style.background =
                `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

            const name =
                document.createElement(
                    "span"
                );

            name.textContent =
                "Object "
                + (index + 1);

            left.appendChild(
                dot
            );

            left.appendChild(
                name
            );

            const score =
                document.createElement(
                    "span"
                );

            score.className =
                "sam-object-score";

            score.textContent =
                "Score "
                + Number(
                    segment.score
                ).toFixed(4);

            row.appendChild(
                left
            );

            row.appendChild(
                score
            );

            container.appendChild(
                row
            );
        }
    );
}


function clearSamMask(updateText = true) {

    samSegments = [];

    updateSamObjectList();

    const canvas = prepareSamCanvas();

    if (canvas) {

        const ctx =
            canvas.getContext("2d");

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );
    }

    document.getElementById(
        "sam_result"
    ).textContent = "";

    if (updateText && samEnabled) {

        document.getElementById(
            "sam_status"
        ).textContent =
            displayMode === "upload"
            ? "Click objects in the uploaded image to segment them."
            : "Click objects in the camera image to segment them.";
    }
}


function loadSamMaskImage(maskBase64) {

    return new Promise(
        (resolve, reject) => {

            const image =
                new Image();

            image.onload =
                () => resolve(image);

            image.onerror =
                reject;

            image.src =
                "data:image/png;base64,"
                + maskBase64;
        }
    );
}


async function redrawSamSegments() {

    const canvas =
        prepareSamCanvas();

    if (!canvas) {
        return;
    }

    const ctx =
        canvas.getContext("2d");

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    for (
        let index = 0;
        index < samSegments.length;
        index++
    ) {

        const segment =
            samSegments[index];

        let maskImage;

        try {

            maskImage =
                await loadSamMaskImage(
                    segment.mask_base64
                );

        } catch (error) {

            console.error(
                "Could not load SAM mask:",
                error
            );

            continue;
        }

        const temp =
            document.createElement(
                "canvas"
            );

        temp.width =
            canvas.width;

        temp.height =
            canvas.height;

        const tempCtx =
            temp.getContext("2d");

        tempCtx.drawImage(
            maskImage,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const imageData =
            tempCtx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
            );

        const pixels =
            imageData.data;

        const color =
            samColors[
                index % samColors.length
            ];

        for (
            let i = 0;
            i < pixels.length;
            i += 4
        ) {

            const inside =
                pixels[i] > 127;

            if (inside) {

                pixels[i] =
                    color[0];

                pixels[i + 1] =
                    color[1];

                pixels[i + 2] =
                    color[2];

                pixels[i + 3] =
                    95;

            } else {

                pixels[i + 3] =
                    0;
            }
        }

        tempCtx.putImageData(
            imageData,
            0,
            0
        );

        ctx.drawImage(
            temp,
            0,
            0
        );

        const px =
            segment.point[0]
            * canvas.width
            / segment.width;

        const py =
            segment.point[1]
            * canvas.height
            / segment.height;

        ctx.beginPath();

        ctx.arc(
            px,
            py,
            5,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        ctx.fill();

        ctx.lineWidth =
            2;

        ctx.strokeStyle =
            "#ffffff";

        ctx.stroke();
    }
}


function drawSamMask(data) {

    samSegments.push(
        data
    );

    redrawSamSegments();
    updateSamObjectList();

    document.getElementById(
        "sam_result"
    ).textContent =
        samSegments.length
        + (
            samSegments.length === 1
            ? " segmented object"
            : " segmented objects"
        );
}


function undoLastSamMask() {

    if (samSegments.length === 0) {
        return;
    }

    samSegments.pop();

    redrawSamSegments();
    updateSamObjectList();

    if (samSegments.length === 0) {

        document.getElementById(
            "sam_result"
        ).textContent = "";

    } else {

        document.getElementById(
            "sam_result"
        ).textContent =
            samSegments.length
            + (
                samSegments.length === 1
                ? " segmented object"
                : " segmented objects"
            );
    }
}


async function runSamAtPoint(event) {

    if (!samEnabled) {
        return;
    }

    const img =
        document.getElementById(
            "cameraStream"
        );

    const rect =
        img.getBoundingClientRect();

    const x =
        (event.clientX - rect.left)
        * sourceWidth
        / rect.width;

    const y =
        (event.clientY - rect.top)
        * sourceHeight
        / rect.height;

    const status =
        document.getElementById(
            "sam_status"
        );

    status.textContent =
        "Segmenting...";

    try {

        const response =
            await fetch(
                PC_CLIP_URL + "/sam",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        x: x,
                        y: y
                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.error
                || "SAM segmentation failed."
            );
        }

        drawSamMask(data);

        status.textContent =
            "Segment added. Click another object to add more.";

    } catch (error) {

        status.textContent =
            "SAM unavailable";

        document.getElementById(
            "sam_result"
        ).textContent =
            error.message;
    }
}


document.getElementById(
    "cameraStream"
).addEventListener(
    "click",
    runSamAtPoint
);


/* ========================================================
   Theme
   ======================================================== */

function applyTheme(theme) {

    const button =
        document.getElementById(
            "themeButton"
        );

    if (theme === "dark") {

        document.body.classList.add(
            "dark"
        );

        button.textContent =
            "☀️";

    } else {

        document.body.classList.remove(
            "dark"
        );

        button.textContent =
            "🌙";

    }

}


function toggleTheme() {

    const dark =
        document.body.classList.contains(
            "dark"
        );

    const theme =
        dark
        ? "light"
        : "dark";

    localStorage.setItem(
        "theme",
        theme
    );

    applyTheme(
        theme
    );

}


const savedTheme =
    localStorage.getItem(
        "theme"
    ) || "light";

applyTheme(
    savedTheme
);


/* ========================================================
   Startup
   ======================================================== */

loadSettings();

loadModelState();

updateClip();

updateYoloOverlay();

setInterval(
    updateYoloOverlay,
    500
);
updateYolo();

setInterval(
    updateClip,
    500
);

setInterval(
    updateYolo,
    500
);

</script>

</body>
</html>
"""


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():

    return HTML


@app.route("/video")
def video():

    return Response(
        generate_mjpeg(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/upload_image", methods=["POST"])
def upload_image():

    global uploaded_image

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({
            "error": "No image file provided."
        }), 400

    data = file.read()

    if not data:
        return jsonify({
            "error": "Uploaded image is empty."
        }), 400

    image_array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "error": "Unsupported or invalid image."
        }), 400

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:
        return jsonify({
            "error": "Could not encode uploaded image."
        }), 500

    with uploaded_image_lock:
        uploaded_image = encoded.tobytes()

    return jsonify({
        "status": "ok",
        "width": int(image.shape[1]),
        "height": int(image.shape[0])
    })


@app.route("/uploaded_image")
def uploaded_image_route():

    with uploaded_image_lock:
        image_data = uploaded_image

    if image_data is None:
        return jsonify({
            "error": "No uploaded image available."
        }), 404

    return Response(
        image_data,
        mimetype="image/jpeg"
    )


@app.route(
    "/settings",
    methods=[
        "GET",
        "POST"
    ]
)
def camera_settings():

    if request.method == "GET":

        with settings_lock:
            return jsonify(
                settings.copy()
            )


    data = request.get_json(
        silent=True
    ) or {}


    with settings_lock:

        if "exposure" in data:

            settings["exposure"] = int(
                float(
                    data["exposure"]
                )
            )


        if "analogue_gain" in data:

            settings["analogue_gain"] = int(
                float(
                    data["analogue_gain"]
                )
            )


        if "gamma" in data:

            settings["gamma"] = float(
                data["gamma"]
            )


        if "blue_gain" in data:

            settings["blue_gain"] = float(
                data["blue_gain"]
            )


        if "green_gain" in data:

            settings["green_gain"] = float(
                data["green_gain"]
            )


        if "red_gain" in data:

            settings["red_gain"] = float(
                data["red_gain"]
            )


        if "contrast" in data:

            settings["contrast"] = float(
                data["contrast"]
            )


        if "brightness" in data:

            settings["brightness"] = int(
                float(
                    data["brightness"]
                )
            )


        if "jpeg_quality" in data:

            settings["jpeg_quality"] = int(
                float(
                    data["jpeg_quality"]
                )
            )


        new_settings = settings.copy()


    if "exposure" in data:

        set_sensor_control(
            "exposure",
            new_settings["exposure"]
        )


    if "analogue_gain" in data:

        set_sensor_control(
            "analogue_gain",
            new_settings[
                "analogue_gain"
            ]
        )


    return jsonify(
        new_settings
    )


@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    with settings_lock:

        settings.clear()

        settings.update(
            DEFAULT_SETTINGS
        )


    apply_sensor_settings()


    return jsonify({
        "status": "ok"
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()


    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True
        )

    finally:

        running = False
