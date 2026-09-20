import time
import threading
import urllib.request
import os
import base64

import cv2
import numpy as np
import torch
import open_clip
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS


# ============================================================
# Configuration
# ============================================================

CAMERA_URL = "http://192.168.0.26:5000/video"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

TOP_K = 5

YOLO_MODEL = "third_party/yolov8n.pt"
YOLO_CONFIDENCE = 0.25

GEMMA_MODEL = "google/gemma-3-4b-it"
GEMMA_MAX_NEW_TOKENS = 64

GEMMA_STYLE_PROMPT = (
    "Answer naturally and directly in one or two brief sentences. "
    "Do not use headings, bullet points, markdown formatting, "
    "or offer additional help."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = os.path.join(
    BASE_DIR,
    "third_party",
    "sam2",
    "checkpoints",
    "sam2.1_hiera_base_plus.pt"
)

PROMPTS = [
    "a person",
    "a cat",
    "a dog",
    "a chair",
    "a table",
    "a desk",
    "a sofa",
    "an armchair",
    "a stool",
    "a bed",
    "a pillow",
    "a blanket",
    "a mattress",
    "a curtain",
    "a rug",
    "a carpet",
    "a lamp",
    "a light bulb",
    "a ceiling light",
    "a fan",
    "an air conditioner",
    "a television",
    "a remote control",
    "a monitor",
    "a laptop",
    "a keyboard",
    "a mouse",
    "a smartphone",
    "a tablet",
    "a charger",
    "a power strip",
    "a cable",
    "a speaker",
    "a clock",
    "a wall clock",
    "a book",
    "a notebook",
    "a magazine",
    "a newspaper",
    "a pen",
    "a pencil",
    "an eraser",
    "a ruler",
    "a backpack",
    "a handbag",
    "a wallet",
    "a key",
    "a pair of glasses",
    "a hat",
    "a shirt",
    "a jacket",
    "a pair of pants",
    "a pair of shoes",
    "a pair of socks",
    "a bottle",
    "a water bottle",
    "a cup",
    "a mug",
    "a drinking glass",
    "a bowl",
    "a plate",
    "a spoon",
    "a fork",
    "a kitchen knife",
    "a pair of chopsticks",
    "a cooking pot",
    "a frying pan",
    "a cutting board",
    "a kettle",
    "a rice cooker",
    "a microwave",
    "an oven",
    "a refrigerator",
    "a toaster",
    "a blender",
    "a sink",
    "a faucet",
    "a dish rack",
    "a trash can",
    "a tissue box",
    "a paper towel roll",
    "a vase",
    "a flower",
    "a houseplant",
    "a plant pot",
    "a mirror",
    "a towel",
    "a bath towel",
    "a toothbrush",
    "toothpaste",
    "a soap bottle",
    "a shampoo bottle",
    "a laundry basket",
    "a clothes hanger",
    "an iron",
    "an ironing board",
    "a vacuum cleaner",
    "a broom",
    "a mop",
    "a storage box",
    "a cardboard box",
    "a plastic container",
    "a basket",
    "an umbrella"
]


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load CLIP
# ============================================================

print("Loading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(device)
model.eval()

print("CLIP model loaded.")


# ============================================================
# Load YOLO
# ============================================================

print("Loading YOLO model...")

yolo_model = YOLO(YOLO_MODEL)

print("YOLO model loaded.")


# ============================================================
# Load SAM 2.1
# ============================================================

print("Loading SAM 2.1 base+ model...")

sam2_model = build_sam2(
    SAM2_CONFIG,
    SAM2_CHECKPOINT,
    device=device
)

sam_predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM 2.1 model loaded.")


# ============================================================
# Load Gemma 3 Vision
# ============================================================

print("Loading Gemma 3 4B Vision...")

gemma_processor = AutoProcessor.from_pretrained(
    GEMMA_MODEL
)

gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
    GEMMA_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

gemma_model.eval()

print("Gemma 3 4B Vision loaded.")


# ============================================================
# Encode fixed prompt vocabulary once
# ============================================================

print(f"Encoding {len(PROMPTS)} prompts...")

with torch.no_grad():

    text_tokens = tokenizer(PROMPTS).to(device)

    text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(
        dim=-1,
        keepdim=True
    )

print("Prompt encoding complete.")


# ============================================================
# Shared state
# ============================================================

state_lock = threading.Lock()

latest_image_feature = None

latest_result = {
    "best": None,
    "scores": []
}

latest_yolo_result = {
    "detections": []
}

latest_camera_frame = None
active_frame = None
source_mode = "camera"

clip_enabled = True
yolo_enabled = True
sam_enabled = True
gemma_enabled = True

sam_lock = threading.Lock()
gemma_lock = threading.Lock()


# ============================================================
# CLIP inference
# ============================================================

def process_frame(frame):

    global latest_image_feature
    global latest_result

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():

        image_feature = model.encode_image(image_tensor)

        image_feature /= image_feature.norm(
            dim=-1,
            keepdim=True
        )

        # CLIP-style classification over the fixed 100 prompts
        logits = 100.0 * image_feature @ text_features.T

        probabilities = logits.softmax(dim=-1)[0]

        values, indices = probabilities.topk(
            min(TOP_K, len(PROMPTS))
        )

    scores = []

    for value, index in zip(values, indices):

        scores.append({
            "label": PROMPTS[index.item()],
            "score": float(value.item())
        })

    result = {
        "best": scores[0]["label"] if scores else None,
        "scores": scores
    }

    # Keep current normalized image embedding so prompt tests
    # do not require another image inference.
    with state_lock:

        latest_image_feature = (
            image_feature.detach().clone()
        )

        latest_result = result


# ============================================================
# YOLO inference
# ============================================================

def process_yolo_frame(frame):

    global latest_yolo_result

    results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        device=0 if device == "cuda" else "cpu",
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [
                float(v)
                for v in box.xyxy[0].tolist()
            ]

            detections.append({
                "label": result.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    with state_lock:

        latest_yolo_result = {
            "detections": detections
        }


# ============================================================
# SAM 2.1 point-prompt segmentation
# ============================================================

def process_sam_point(frame, x, y):

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    point_coords = np.array(
        [[float(x), float(y)]],
        dtype=np.float32
    )

    point_labels = np.array(
        [1],
        dtype=np.int32
    )

    with sam_lock:

        with torch.inference_mode():

            sam_predictor.set_image(
                image_rgb
            )

            masks, scores, _ = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

    best_index = int(
        np.argmax(scores)
    )

    mask = masks[best_index].astype(np.uint8) * 255

    ok, encoded = cv2.imencode(
        ".png",
        mask
    )

    if not ok:
        raise RuntimeError(
            "Could not encode SAM mask."
        )

    mask_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return {
        "score": float(scores[best_index]),
        "point": [float(x), float(y)],
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "mask_base64": mask_base64
    }


# ============================================================
# Gemma 3 Vision inference
# ============================================================

def process_gemma_prompt(frame, prompt):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        rgb
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": (
			GEMMA_STYLE_PROMPT
        		+ "\n\nUser question: "
        		+ prompt
		    )
                }
            ]
        }
    ]

    with gemma_lock:

        inputs = gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(
            gemma_model.device
        )

        with torch.inference_mode():

            output = gemma_model.generate(
                **inputs,
                max_new_tokens=GEMMA_MAX_NEW_TOKENS,
                do_sample=False
            )

        generated = output[
            0,
            inputs["input_ids"].shape[-1]:
        ]

        answer = gemma_processor.decode(
            generated,
            skip_special_tokens=True
        ).strip()

    return answer


# ============================================================
# MJPEG camera loop
# ============================================================

def clip_loop():

    global latest_camera_frame
    global active_frame

    print(f"Connecting to camera: {CAMERA_URL}")

    while True:

        stream = None

        try:

            stream = urllib.request.urlopen(
                CAMERA_URL,
                timeout=10
            )

            print("Connected to Raspberry Pi camera.")

            buffer = b""

            while True:

                chunk = stream.read(4096)

                if not chunk:
                    raise ConnectionError(
                        "Camera stream disconnected."
                    )

                buffer += chunk

                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9")

                if start != -1 and end != -1 and end > start:

                    jpg = buffer[start:end + 2]

                    buffer = buffer[end + 2:]

                    image_array = np.frombuffer(
                        jpg,
                        dtype=np.uint8
                    )

                    frame = cv2.imdecode(
                        image_array,
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None:

                        with state_lock:

                            latest_camera_frame = frame.copy()

                            use_camera = (
                                source_mode == "camera"
                            )

                            if use_camera:
                                active_frame = frame.copy()

                            run_clip = (
                                use_camera
                                and clip_enabled
                            )

                            run_yolo = (
                                use_camera
                                and yolo_enabled
                            )

                        if run_clip:
                            process_frame(frame)

                        if run_yolo:
                            process_yolo_frame(frame)

        except Exception as e:

            print(f"Camera connection error: {e}")
            print("Retrying in 2 seconds...")

            time.sleep(2)

        finally:

            if stream is not None:

                try:
                    stream.close()
                except Exception:
                    pass


# ============================================================
# Flask server
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# Existing CLIP Recognition API
# ============================================================

@app.route("/clip")
def clip():

    with state_lock:

        result = {
            "enabled": clip_enabled,
            "source": source_mode,
            "best": latest_result["best"],
            "scores": [
                dict(item)
                for item in latest_result["scores"]
            ]
        }

    return jsonify(result)


# ============================================================
# YOLO Detection API
# ============================================================

@app.route("/yolo")
def yolo():

    with state_lock:

        if active_frame is None:
            source_width = 0
            source_height = 0
        else:
            source_height, source_width = (
                active_frame.shape[:2]
            )

        result = {
            "enabled": yolo_enabled,
            "source": source_mode,
            "width": int(source_width),
            "height": int(source_height),
            "detections": [
                dict(item)
                for item in latest_yolo_result["detections"]
            ]
        }

    return jsonify(result)


# ============================================================
# SAM 2.1 Segmentation API
# ============================================================

@app.route("/sam", methods=["GET", "POST"])
def sam():

    if request.method == "GET":

        with state_lock:
            return jsonify({
                "enabled": sam_enabled
            })

    data = request.get_json(
        silent=True
    ) or {}

    try:
        x = float(data["x"])
        y = float(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Valid x and y coordinates are required."
        }), 400

    with state_lock:

        if not sam_enabled:
            return jsonify({
                "error": "SAM is disabled."
            }), 503

        if active_frame is None:
            return jsonify({
                "error": "No active image available yet."
            }), 503

        frame = active_frame.copy()

    h, w = frame.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        return jsonify({
            "error": "SAM point is outside the camera image."
        }), 400

    try:
        result = process_sam_point(
            frame,
            x,
            y
        )
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    result["enabled"] = True

    return jsonify(result)


# ============================================================
# Gemma 3 Vision API
# ============================================================

@app.route("/gemma", methods=["GET", "POST"])
def gemma():

    if request.method == "GET":

        with state_lock:
            return jsonify({
                "enabled": gemma_enabled,
                "model": GEMMA_MODEL,
                "source": source_mode
            })

    data = request.get_json(
        silent=True
    ) or {}

    prompt = str(
        data.get("prompt", "")
    ).strip()

    if not prompt:
        return jsonify({
            "error": "No prompt provided."
        }), 400

    with state_lock:

        if not gemma_enabled:
            return jsonify({
                "error": "Gemma 3 Vision is disabled."
            }), 503

        if active_frame is None:
            return jsonify({
                "error": "No active image available yet."
            }), 503

        frame = active_frame.copy()
        current_source = source_mode

    try:

        answer = process_gemma_prompt(
            frame,
            prompt
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    return jsonify({
        "prompt": prompt,
        "answer": answer,
        "source": current_source,
        "model": GEMMA_MODEL
    })


# ============================================================
# Uploaded video frame API
# ============================================================

@app.route("/source/video_frame", methods=["POST"])
def source_video_frame():

    global active_frame
    global source_mode

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({
            "error": "No video frame provided."
        }), 400

    data = file.read()

    if not data:
        return jsonify({
            "error": "Video frame is empty."
        }), 400

    image_array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Could not decode video frame."
        }), 400

    with state_lock:

        source_mode = "video"
        active_frame = frame.copy()

        run_clip = clip_enabled
        run_yolo = yolo_enabled

    if run_clip:
        process_frame(frame)

    if run_yolo:
        process_yolo_frame(frame)

    h, w = frame.shape[:2]

    return jsonify({
        "status": "ok",
        "source": source_mode,
        "width": int(w),
        "height": int(h)
    })


@app.route("/source/video_clear", methods=["POST"])
def source_video_clear():

    global active_frame
    global source_mode
    global latest_image_feature
    global latest_result
    global latest_yolo_result

    with state_lock:

        source_mode = "video"
        active_frame = None

        latest_image_feature = None

        latest_result = {
            "best": None,
            "scores": []
        }

        latest_yolo_result = {
            "detections": []
        }

    return jsonify({
        "status": "ok",
        "source": source_mode
    })


# ============================================================
# Active image source API
# ============================================================

@app.route("/source/upload", methods=["POST"])
def source_upload():

    global active_frame
    global source_mode

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

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Unsupported or invalid image."
        }), 400

    with state_lock:

        source_mode = "upload"
        active_frame = frame.copy()

        run_clip = clip_enabled
        run_yolo = yolo_enabled

    if run_clip:
        process_frame(frame)

    if run_yolo:
        process_yolo_frame(frame)

    h, w = frame.shape[:2]

    return jsonify({
        "status": "ok",
        "source": source_mode,
        "width": int(w),
        "height": int(h)
    })


@app.route("/source/camera", methods=["POST"])
def source_camera():

    global active_frame
    global source_mode

    with state_lock:

        source_mode = "camera"

        if latest_camera_frame is None:
            frame = None
        else:
            frame = latest_camera_frame.copy()
            active_frame = frame.copy()

        run_clip = (
            frame is not None
            and clip_enabled
        )

        run_yolo = (
            frame is not None
            and yolo_enabled
        )

    if run_clip:
        process_frame(frame)

    if run_yolo:
        process_yolo_frame(frame)

    if frame is None:
        return jsonify({
            "status": "waiting",
            "source": source_mode,
            "width": 0,
            "height": 0
        })

    h, w = frame.shape[:2]

    return jsonify({
        "status": "ok",
        "source": source_mode,
        "width": int(w),
        "height": int(h)
    })


@app.route("/source")
def source_status():

    with state_lock:

        if active_frame is None:
            source_width = 0
            source_height = 0
        else:
            source_height, source_width = (
                active_frame.shape[:2]
            )

        return jsonify({
            "source": source_mode,
            "width": int(source_width),
            "height": int(source_height)
        })


# ============================================================
# CLIP / YOLO / SAM Control API
# ============================================================

@app.route("/control", methods=["GET", "POST"])
def control():

    global clip_enabled
    global yolo_enabled
    global sam_enabled
    global gemma_enabled
    global latest_image_feature
    global latest_result
    global latest_yolo_result

    refresh_frame = None
    refresh_clip = False
    refresh_yolo = False

    if request.method == "POST":

        data = request.get_json(
            silent=True
        ) or {}

        with state_lock:

            if "clip_enabled" in data:
                clip_enabled = bool(
                    data["clip_enabled"]
                )

                if not clip_enabled:
                    latest_image_feature = None
                    latest_result = {
                        "best": None,
                        "scores": []
                    }

            if "yolo_enabled" in data:
                yolo_enabled = bool(
                    data["yolo_enabled"]
                )

                if not yolo_enabled:
                    latest_yolo_result = {
                        "detections": []
                    }

            if "sam_enabled" in data:
                sam_enabled = bool(
                    data["sam_enabled"]
                )

            if "gemma_enabled" in data:
                gemma_enabled = bool(
                    data["gemma_enabled"]
                )

            if (
                source_mode in ("upload", "video")
                and active_frame is not None
            ):
                refresh_frame = (
                    active_frame.copy()
                )
                refresh_clip = clip_enabled
                refresh_yolo = yolo_enabled

        if refresh_frame is not None:

            if refresh_clip:
                process_frame(
                    refresh_frame
                )

            if refresh_yolo:
                process_yolo_frame(
                    refresh_frame
                )

    with state_lock:
        return jsonify({
            "clip_enabled": clip_enabled,
            "yolo_enabled": yolo_enabled,
            "sam_enabled": sam_enabled,
            "gemma_enabled": gemma_enabled,
            "source": source_mode
        })


# ============================================================
# CLIP Prompt Test API
# ============================================================

@app.route("/clip_prompt", methods=["GET", "POST"])
def clip_prompt():

    if request.method == "POST":

        data = request.get_json(
            silent=True
        ) or {}

        prompt = str(
            data.get("prompt", "")
        ).strip()

    else:

        prompt = request.args.get(
            "prompt",
            ""
        ).strip()


    if not prompt:

        return jsonify({
            "error": "No prompt provided."
        }), 400


    with state_lock:

        if not clip_enabled:

            return jsonify({
                "error": "CLIP is disabled."
            }), 503


        if latest_image_feature is None:

            return jsonify({
                "error": "No active image available yet."
            }), 503

        image_feature = (
            latest_image_feature
            .detach()
            .clone()
        )


    # Encode arbitrary user prompt
    with torch.no_grad():

        tokens = tokenizer(
            [prompt]
        ).to(device)

        prompt_feature = model.encode_text(
            tokens
        )

        prompt_feature /= prompt_feature.norm(
            dim=-1,
            keepdim=True
        )

        # Cosine similarity:
        #
        # image_feature and prompt_feature are normalized,
        # therefore dot product == cosine similarity.
        similarity = (
            image_feature @ prompt_feature.T
        ).item()


    return jsonify({
        "prompt": prompt,
        "similarity": float(similarity)
    })


# ============================================================
# Status
# ============================================================

@app.route("/")
def index():

    with state_lock:
        camera_ready = latest_image_feature is not None

    return jsonify({
        "status": "running",
        "device": device,
        "model": MODEL_NAME,
        "camera_ready": camera_ready,
        "clip_enabled": clip_enabled,
        "yolo_enabled": yolo_enabled,
        "sam_enabled": sam_enabled,
        "gemma_enabled": gemma_enabled,
        "gemma_model": GEMMA_MODEL,
        "source": source_mode,
        "num_prompts": len(PROMPTS),
        "endpoints": {
            "recognition": "/clip",
            "prompt_test": "/clip_prompt?prompt=a book",
            "yolo": "/yolo",
            "sam": "/sam",
            "gemma": "/gemma",
            "control": "/control",
            "source": "/source",
            "source_upload": "/source/upload",
            "source_camera": "/source/camera",
            "source_video_frame": "/source/video_frame",
            "source_video_clear": "/source/video_clear"
        }
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    worker = threading.Thread(
        target=clip_loop,
        daemon=True
    )

    worker.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
