import time
import threading
import urllib.request
import os
import sys
import base64
import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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

YOLO_MODEL = "third_party/yolo/yolov8n.pt"
YOLO_CONFIDENCE = 0.25

POSE_MODEL = "third_party/yolo/yolov8n-pose.pt"
POSE_CONFIDENCE = 0.25

REALESRGAN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "third_party",
    "Real-ESRGAN"
)

REALESRGAN_MODEL = os.path.join(
    REALESRGAN_ROOT,
    "weights",
    "RealESRGAN_x2plus.pth"
)

ANIMEGAN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "third_party",
    "AnimeGANv2-PyTorch"
)

ANIMEGAN_STYLES = (
    "paprika",
    "face_paint_512_v2",
    "face_paint_512_v1",
    "celeba_distill"
)

anime_style = "face_paint_512_v2"

GEMMA_MODEL = "google/gemma-3-4b-it"
GEMMA_MAX_NEW_TOKENS = 64

GEMMA_STYLE_PROMPT = (
    "Answer naturally and directly in one or two brief sentences. "
    "Do not use headings, bullet points, markdown formatting, "
    "or offer additional help."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if REALESRGAN_ROOT not in sys.path:
    sys.path.insert(
        0,
        REALESRGAN_ROOT
    )

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

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
# Models are loaded lazily according to dashboard toggles
# ============================================================

realesrgan_upsampler = None

animegan_model = None
animegan_loaded_style = None

model = None
preprocess = None
tokenizer = None
text_features = None

yolo_model = None
pose_model = None

sam2_model = None
sam_predictor = None

gemma_processor = None
gemma_model = None


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

latest_pose_result = {
    "persons": []
}

latest_camera_frame = None
active_frame = None
source_original_frame = None
source_mode = "camera"

enhancement_enabled = True
anime_enabled = True

clip_enabled = True
yolo_enabled = False
pose_enabled = False
sam_enabled = False
gemma_enabled = False

enhancement_lock = threading.RLock()
anime_lock = threading.RLock()
clip_lock = threading.RLock()
yolo_lock = threading.RLock()
pose_lock = threading.RLock()
sam_lock = threading.RLock()
gemma_lock = threading.RLock()


# ============================================================
# Lazy model loading / VRAM lifecycle
# ============================================================

def release_cuda_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_realesrgan_model():

    global realesrgan_upsampler

    with enhancement_lock:

        if realesrgan_upsampler is not None:
            return

        print("Loading Real-ESRGAN x2 model...")

        if not os.path.exists(
            REALESRGAN_MODEL
        ):
            raise FileNotFoundError(
                "Real-ESRGAN checkpoint not found: "
                + REALESRGAN_MODEL
            )

        network = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )

        realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path=REALESRGAN_MODEL,
            model=network,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda")
        )

        print("Real-ESRGAN x2 model loaded.")


def unload_realesrgan_model():

    global realesrgan_upsampler

    with enhancement_lock:

        if realesrgan_upsampler is None:
            return

        realesrgan_upsampler = None

    release_cuda_memory()
    print("Real-ESRGAN x2 model unloaded.")


def load_animegan_model(
    style_name=None
):

    global animegan_model
    global animegan_loaded_style

    if style_name is None:
        style_name = anime_style

    if style_name not in ANIMEGAN_STYLES:
        raise ValueError(
            "Unsupported AnimeGANv2 style: "
            + str(style_name)
        )

    with anime_lock:

        if (
            animegan_model is not None
            and animegan_loaded_style
            == style_name
        ):
            return

        if not os.path.isdir(
            ANIMEGAN_ROOT
        ):
            raise FileNotFoundError(
                "AnimeGANv2-PyTorch repository not found: "
                + ANIMEGAN_ROOT
            )

        print(
            "Loading AnimeGANv2 style: "
            + style_name
            + "..."
        )

        new_model = torch.hub.load(
            ANIMEGAN_ROOT,
            "generator",
            source="local",
            pretrained=style_name,
            device=device
        ).eval()

        old_model = animegan_model

        animegan_model = new_model
        animegan_loaded_style = style_name

        if old_model is not None:
            del old_model

    release_cuda_memory()

    print(
        "AnimeGANv2 model loaded. Style: "
        + style_name
    )


def unload_animegan_model():

    global animegan_model
    global animegan_loaded_style

    with anime_lock:

        if animegan_model is None:
            animegan_loaded_style = None
            return

        animegan_model = None
        animegan_loaded_style = None

    release_cuda_memory()
    print("AnimeGANv2 model unloaded.")


def load_clip_model():

    global model
    global preprocess
    global tokenizer
    global text_features

    with clip_lock:

        if model is not None:
            return

        print("Loading CLIP model...")

        (
            loaded_model,
            _,
            loaded_preprocess
        ) = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED
        )

        loaded_tokenizer = (
            open_clip.get_tokenizer(
                MODEL_NAME
            )
        )

        loaded_model = (
            loaded_model
            .to(device)
            .eval()
        )

        print(
            f"Encoding {len(PROMPTS)} prompts..."
        )

        with torch.inference_mode():

            tokens = (
                loaded_tokenizer(
                    PROMPTS
                ).to(device)
            )

            encoded_text = (
                loaded_model
                .encode_text(tokens)
            )

            encoded_text /= (
                encoded_text.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        model = loaded_model
        preprocess = loaded_preprocess
        tokenizer = loaded_tokenizer
        text_features = encoded_text

        print("CLIP model loaded.")


def unload_clip_model():

    global model
    global preprocess
    global tokenizer
    global text_features
    global latest_image_feature

    with clip_lock:

        model = None
        preprocess = None
        tokenizer = None
        text_features = None
        latest_image_feature = None

    release_cuda_memory()
    print("CLIP model unloaded.")


def load_yolo_model():

    global yolo_model

    with yolo_lock:

        if yolo_model is not None:
            return

        print("Loading YOLOv8 model...")

        yolo_model = YOLO(
            YOLO_MODEL
        )

        print("YOLOv8 model loaded.")


def unload_yolo_model():

    global yolo_model

    with yolo_lock:

        yolo_model = None

    release_cuda_memory()
    print("YOLOv8 model unloaded.")


def load_pose_model():

    global pose_model

    with pose_lock:

        if pose_model is not None:
            return

        print(
            "Loading YOLOv8 pose model..."
        )

        pose_model = YOLO(
            POSE_MODEL
        )

        print(
            "YOLOv8 pose model loaded."
        )


def unload_pose_model():

    global pose_model

    with pose_lock:

        pose_model = None

    release_cuda_memory()
    print("YOLOv8 pose model unloaded.")


def load_sam_model():

    global sam2_model
    global sam_predictor

    with sam_lock:

        if sam_predictor is not None:
            return

        print(
            "Loading SAM 2.1 base+ model..."
        )

        sam2_model = build_sam2(
            SAM2_CONFIG,
            SAM2_CHECKPOINT,
            device=device
        )

        sam_predictor = (
            SAM2ImagePredictor(
                sam2_model
            )
        )

        print(
            "SAM 2.1 model loaded."
        )


def unload_sam_model():

    global sam2_model
    global sam_predictor

    with sam_lock:

        sam_predictor = None
        sam2_model = None

    release_cuda_memory()
    print("SAM 2.1 model unloaded.")


def load_gemma_model():

    global gemma_processor
    global gemma_model

    with gemma_lock:

        if gemma_model is not None:
            return

        print(
            "Loading Gemma 3 4B Vision..."
        )

        gemma_processor = (
            AutoProcessor.from_pretrained(
                GEMMA_MODEL
            )
        )

        gemma_model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                GEMMA_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        )

        gemma_model.eval()

        print(
            "Gemma 3 4B Vision loaded."
        )


def unload_gemma_model():

    global gemma_processor
    global gemma_model

    with gemma_lock:

        gemma_model = None
        gemma_processor = None

    release_cuda_memory()
    print("Gemma 3 4B Vision unloaded.")


def initialize_enabled_models():

    print(
        "Initializing models enabled "
        "by default..."
    )

    if enhancement_enabled:
        load_realesrgan_model()

    if anime_enabled:
        load_animegan_model(
            anime_style
        )

    if clip_enabled:
        load_clip_model()

    if yolo_enabled:
        load_yolo_model()

    if pose_enabled:
        load_pose_model()

    if sam_enabled:
        load_sam_model()

    if gemma_enabled:
        load_gemma_model()

    print(
        "Default enabled models ready."
    )


initialize_enabled_models()


# ============================================================
# Image enhancement / AnimeGANv2
# ============================================================

def process_realesrgan_frame(frame):

    with enhancement_lock:

        if realesrgan_upsampler is None:
            load_realesrgan_model()

        output, _ = (
            realesrgan_upsampler.enhance(
                frame,
                outscale=2
            )
        )

    return output


def process_animegan_frame(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        rgb
    ).to(
        device=device,
        dtype=torch.float32
    )

    tensor = (
        tensor.permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        / 127.5
        - 1.0
    )

    h = int(tensor.shape[-2])
    w = int(tensor.shape[-1])

    pad_h = (
        32 - (h % 32)
    ) % 32

    pad_w = (
        32 - (w % 32)
    ) % 32

    if pad_h or pad_w:

        pad_mode = (
            "reflect"
            if h > pad_h
            and w > pad_w
            else "replicate"
        )

        tensor = F.pad(
            tensor,
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            mode=pad_mode
        )

    with anime_lock:

        current_style = anime_style

        if (
            animegan_model is None
            or animegan_loaded_style
            != current_style
        ):
            load_animegan_model(
                current_style
            )

        with torch.inference_mode():

            output = animegan_model(
                tensor
            )

    output = output[
        :,
        :,
        :h,
        :w
    ]

    output = (
        output[0]
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    return cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )


def encode_frame_base64(frame):

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:
        raise RuntimeError(
            "Could not encode processed image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")


def run_enabled_perception_models(frame):

    with state_lock:

        run_clip = clip_enabled
        run_yolo = yolo_enabled
        run_pose = pose_enabled

    if run_clip:
        process_frame(frame)

    if run_yolo:
        process_yolo_frame(frame)

    if run_pose:
        process_pose_frame(frame)


def process_static_pipeline(original):

    processed = original.copy()

    with state_lock:

        do_enhance = (
            enhancement_enabled
        )

        do_anime = anime_enabled

    if do_enhance:

        processed = (
            process_realesrgan_frame(
                processed
            )
        )

    if do_anime:

        processed = (
            process_animegan_frame(
                processed
            )
        )

    return (
        processed,
        do_enhance,
        do_anime
    )


def activate_static_frame(
    frame,
    new_source
):

    global active_frame
    global source_original_frame
    global source_mode

    original = frame.copy()

    (
        processed,
        did_enhance,
        did_anime
    ) = process_static_pipeline(
        original
    )

    with state_lock:

        source_mode = new_source

        source_original_frame = (
            original.copy()
        )

        active_frame = (
            processed.copy()
        )

    run_enabled_perception_models(
        processed
    )

    return (
        processed,
        did_enhance,
        did_anime
    )


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

    with clip_lock:

        if model is None:
            load_clip_model()

        image_tensor = (
            preprocess(image)
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():

            image_feature = (
                model.encode_image(
                    image_tensor
                )
            )

            image_feature /= (
                image_feature.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            # CLIP-style classification over the fixed prompts.
            logits = (
                100.0
                * image_feature
                @ text_features.T
            )

            probabilities = (
                logits.softmax(
                    dim=-1
                )[0]
            )

            values, indices = (
                probabilities.topk(
                    min(
                        TOP_K,
                        len(PROMPTS)
                    )
                )
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

    with yolo_lock:

        if yolo_model is None:
            load_yolo_model()

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
# Pose estimation
# ============================================================

def process_pose_frame(frame):

    global latest_pose_result

    with pose_lock:

        if pose_model is None:
            load_pose_model()

        results = pose_model.predict(
            source=frame,
            conf=POSE_CONFIDENCE,
            device=0 if device == "cuda" else "cpu",
            verbose=False
        )

    persons = []

    for result in results:

        if result.keypoints is None:
            continue

        xy = result.keypoints.xy

        if xy is None:
            continue

        xy = xy.detach().cpu().numpy()

        kp_conf = result.keypoints.conf

        if kp_conf is not None:
            kp_conf = kp_conf.detach().cpu().numpy()

        box_conf = None

        if (
            result.boxes is not None
            and result.boxes.conf is not None
        ):
            box_conf = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

        for person_index in range(xy.shape[0]):

            keypoints = []

            for keypoint_index in range(
                xy.shape[1]
            ):

                x = float(
                    xy[
                        person_index,
                        keypoint_index,
                        0
                    ]
                )

                y = float(
                    xy[
                        person_index,
                        keypoint_index,
                        1
                    ]
                )

                confidence = None

                if kp_conf is not None:
                    confidence = float(
                        kp_conf[
                            person_index,
                            keypoint_index
                        ]
                    )

                keypoints.append({
                    "x": x,
                    "y": y,
                    "confidence": confidence
                })

            person_confidence = None

            if (
                box_conf is not None
                and person_index < len(box_conf)
            ):
                person_confidence = float(
                    box_conf[person_index]
                )

            persons.append({
                "confidence": person_confidence,
                "keypoints": keypoints
            })

    with state_lock:

        latest_pose_result = {
            "persons": persons
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

        if sam_predictor is None:
            load_sam_model()

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

        if gemma_model is None:
            load_gemma_model()

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

                            run_pose = (
                                use_camera
                                and pose_enabled
                            )

                        if run_clip:
                            process_frame(frame)

                        if run_yolo:
                            process_yolo_frame(frame)

                        if run_pose:
                            process_pose_frame(frame)

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
# Pose Estimation API
# ============================================================

@app.route("/pose")
def pose():

    with state_lock:

        if active_frame is None:
            source_width = 0
            source_height = 0
        else:
            source_height, source_width = (
                active_frame.shape[:2]
            )

        result = {
            "enabled": pose_enabled,
            "source": source_mode,
            "width": int(source_width),
            "height": int(source_height),
            "persons": [
                {
                    "confidence": person["confidence"],
                    "keypoints": [
                        dict(keypoint)
                        for keypoint
                        in person["keypoints"]
                    ]
                }
                for person in latest_pose_result["persons"]
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
# Frozen camera frame API
# ============================================================

@app.route("/source/camera_frame", methods=["POST"])
def source_camera_frame():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({
            "error": "No camera frame provided."
        }), 400

    data = file.read()

    if not data:
        return jsonify({
            "error": "Camera frame is empty."
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
            "error": "Could not decode camera frame."
        }), 400

    try:

        processed, enhanced, anime_applied = (
            activate_static_frame(
                frame,
                "camera_frozen"
            )
        )

    except Exception as e:

        return jsonify({
            "error": (
                "Static image processing failed: "
                + str(e)
            )
        }), 500

    h, w = processed.shape[:2]

    response = {
        "status": "ok",
        "source": "camera_frozen",
        "width": int(w),
        "height": int(h),
        "enhancement_enabled": enhancement_enabled,
        "enhanced": enhanced,
        "anime_enabled": anime_enabled,
        "anime_applied": anime_applied
    }

    if enhanced or anime_applied:
        response["image_base64"] = (
            encode_frame_base64(
                processed
            )
        )

    return jsonify(response)


# ============================================================
# Uploaded video frame API
# ============================================================

@app.route("/source/video_frame", methods=["POST"])
def source_video_frame():

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

    try:

        processed, enhanced, anime_applied = (
            activate_static_frame(
                frame,
                "video"
            )
        )

    except Exception as e:

        return jsonify({
            "error": (
                "Static image processing failed: "
                + str(e)
            )
        }), 500

    h, w = processed.shape[:2]

    response = {
        "status": "ok",
        "source": "video",
        "width": int(w),
        "height": int(h),
        "enhancement_enabled": enhancement_enabled,
        "enhanced": enhanced,
        "anime_enabled": anime_enabled,
        "anime_applied": anime_applied
    }

    if enhanced or anime_applied:
        response["image_base64"] = (
            encode_frame_base64(
                processed
            )
        )

    return jsonify(response)


@app.route("/source/video_clear", methods=["POST"])
def source_video_clear():

    global active_frame
    global source_mode
    global latest_image_feature
    global latest_result
    global latest_yolo_result
    global latest_pose_result
    global source_original_frame

    with state_lock:

        source_mode = "video"
        active_frame = None
        source_original_frame = None

        latest_image_feature = None

        latest_result = {
            "best": None,
            "scores": []
        }

        latest_yolo_result = {
            "detections": []
        }

        latest_pose_result = {
            "persons": []
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

    try:

        processed, enhanced, anime_applied = (
            activate_static_frame(
                frame,
                "upload"
            )
        )

    except Exception as e:

        return jsonify({
            "error": (
                "Static image processing failed: "
                + str(e)
            )
        }), 500

    h, w = processed.shape[:2]

    response = {
        "status": "ok",
        "source": "upload",
        "width": int(w),
        "height": int(h),
        "enhancement_enabled": enhancement_enabled,
        "enhanced": enhanced,
        "anime_enabled": anime_enabled,
        "anime_applied": anime_applied
    }

    if enhanced or anime_applied:
        response["image_base64"] = (
            encode_frame_base64(
                processed
            )
        )

    return jsonify(response)


@app.route("/source/camera", methods=["POST"])
def source_camera():

    global active_frame
    global source_original_frame
    global source_mode

    with state_lock:

        source_mode = "camera"
        source_original_frame = None

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

        run_pose = (
            frame is not None
            and pose_enabled
        )

    if run_clip:
        process_frame(frame)

    if run_yolo:
        process_yolo_frame(frame)

    if run_pose:
        process_pose_frame(frame)

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
# Real-ESRGAN Enhancement API
# ============================================================

def reprocess_current_static_source():

    global active_frame

    with state_lock:

        current_source = source_mode

        if (
            source_original_frame
            is not None
            and current_source
            in (
                "upload",
                "video",
                "camera_frozen"
            )
        ):
            original = (
                source_original_frame
                .copy()
            )
        else:
            original = None

    if original is None:

        with state_lock:

            if active_frame is None:
                width = 0
                height = 0
            else:
                height, width = (
                    active_frame.shape[:2]
                )

        return (
            None,
            False,
            False,
            int(width),
            int(height),
            current_source
        )

    (
        processed,
        did_enhance,
        did_anime
    ) = process_static_pipeline(
        original
    )

    with state_lock:

        active_frame = (
            processed.copy()
        )

    run_enabled_perception_models(
        processed
    )

    height, width = (
        processed.shape[:2]
    )

    return (
        processed,
        did_enhance,
        did_anime,
        int(width),
        int(height),
        current_source
    )


@app.route(
    "/enhancement",
    methods=["GET", "POST"]
)
def enhancement():

    global enhancement_enabled

    if request.method == "GET":

        with state_lock:

            return jsonify({
                "enabled": enhancement_enabled,
                "loaded": (
                    realesrgan_upsampler
                    is not None
                ),
                "model": "RealESRGAN_x2plus",
                "scale": 2,
                "source": source_mode,
                "applies_to_live_stream": False
            })

    data = request.get_json(
        silent=True
    ) or {}

    requested_enabled = bool(
        data.get(
            "enabled",
            enhancement_enabled
        )
    )

    try:

        if requested_enabled:
            load_realesrgan_model()

        with state_lock:

            enhancement_enabled = (
                requested_enabled
            )

        if not requested_enabled:
            unload_realesrgan_model()

        (
            processed,
            enhanced,
            anime_applied,
            width,
            height,
            current_source
        ) = reprocess_current_static_source()

    except Exception as e:

        with state_lock:
            enhancement_enabled = False

        unload_realesrgan_model()

        return jsonify({
            "error": (
                "Image processing failed: "
                + str(e)
            )
        }), 500

    response = {
        "enabled": enhancement_enabled,
        "loaded": (
            realesrgan_upsampler
            is not None
        ),
        "enhanced": enhanced,
        "anime_enabled": anime_enabled,
        "anime_applied": anime_applied,
        "source": current_source,
        "width": width,
        "height": height,
        "scale": 2,
        "applies_to_live_stream": False
    }

    if (
        processed is not None
        and (
            enhanced
            or anime_applied
        )
    ):
        response["image_base64"] = (
            encode_frame_base64(
                processed
            )
        )

    return jsonify(response)


# ============================================================
# AnimeGANv2 API
# ============================================================

@app.route(
    "/anime",
    methods=["GET", "POST"]
)
def anime():

    global anime_enabled
    global anime_style

    if request.method == "GET":

        with state_lock:

            return jsonify({
                "enabled": anime_enabled,
                "loaded": (
                    animegan_model
                    is not None
                ),
                "loaded_style": (
                    animegan_loaded_style
                ),
                "model": "AnimeGANv2",
                "style": anime_style,
                "styles": list(
                    ANIMEGAN_STYLES
                ),
                "source": source_mode,
                "applies_to_live_stream": False
            })

    data = request.get_json(
        silent=True
    ) or {}

    requested_enabled = bool(
        data.get(
            "enabled",
            anime_enabled
        )
    )

    requested_style = str(
        data.get(
            "style",
            anime_style
        )
    ).strip()

    if (
        requested_style
        not in ANIMEGAN_STYLES
    ):
        return jsonify({
            "error": (
                "Unsupported AnimeGANv2 style: "
                + requested_style
            ),
            "styles": list(
                ANIMEGAN_STYLES
            )
        }), 400

    previous_style = anime_style

    try:

        if requested_enabled:
            load_animegan_model(
                requested_style
            )

        with state_lock:

            anime_enabled = (
                requested_enabled
            )

            anime_style = (
                requested_style
            )

        if not requested_enabled:
            unload_animegan_model()

        (
            processed,
            enhanced,
            anime_applied,
            width,
            height,
            current_source
        ) = reprocess_current_static_source()

    except Exception as e:

        with state_lock:

            anime_enabled = False
            anime_style = previous_style

        unload_animegan_model()

        return jsonify({
            "error": (
                "AnimeGANv2 processing failed: "
                + str(e)
            )
        }), 500

    response = {
        "enabled": anime_enabled,
        "loaded": (
            animegan_model
            is not None
        ),
        "loaded_style": (
            animegan_loaded_style
        ),
        "enhancement_enabled": enhancement_enabled,
        "enhanced": enhanced,
        "anime_applied": anime_applied,
        "source": current_source,
        "width": width,
        "height": height,
        "model": "AnimeGANv2",
        "style": anime_style,
        "styles": list(
            ANIMEGAN_STYLES
        ),
        "applies_to_live_stream": False
    }

    if (
        processed is not None
        and (
            enhanced
            or anime_applied
        )
    ):
        response["image_base64"] = (
            encode_frame_base64(
                processed
            )
        )

    return jsonify(response)


# ============================================================
# CLIP / YOLO / SAM Control API
# ============================================================

@app.route("/control", methods=["GET", "POST"])
def control():

    global clip_enabled
    global yolo_enabled
    global pose_enabled
    global sam_enabled
    global gemma_enabled
    global latest_image_feature
    global latest_result
    global latest_yolo_result
    global latest_pose_result

    refresh_frame = None
    refresh_clip = False
    refresh_yolo = False
    refresh_pose = False

    if request.method == "POST":

        data = request.get_json(
            silent=True
        ) or {}

        requested = {}

        with state_lock:

            requested["clip"] = bool(
                data.get(
                    "clip_enabled",
                    clip_enabled
                )
            )

            requested["yolo"] = bool(
                data.get(
                    "yolo_enabled",
                    yolo_enabled
                )
            )

            requested["pose"] = bool(
                data.get(
                    "pose_enabled",
                    pose_enabled
                )
            )

            requested["sam"] = bool(
                data.get(
                    "sam_enabled",
                    sam_enabled
                )
            )

            requested["gemma"] = bool(
                data.get(
                    "gemma_enabled",
                    gemma_enabled
                )
            )

        try:

            if requested["clip"]:
                load_clip_model()
            else:
                unload_clip_model()

            if requested["yolo"]:
                load_yolo_model()
            else:
                unload_yolo_model()

            if requested["pose"]:
                load_pose_model()
            else:
                unload_pose_model()

            if requested["sam"]:
                load_sam_model()
            else:
                unload_sam_model()

            if requested["gemma"]:
                load_gemma_model()
            else:
                unload_gemma_model()

        except Exception as e:

            return jsonify({
                "error": (
                    "Model load/unload failed: "
                    + str(e)
                )
            }), 500

        with state_lock:

            clip_enabled = (
                requested["clip"]
            )

            yolo_enabled = (
                requested["yolo"]
            )

            pose_enabled = (
                requested["pose"]
            )

            sam_enabled = (
                requested["sam"]
            )

            gemma_enabled = (
                requested["gemma"]
            )

            if not clip_enabled:

                latest_image_feature = None

                latest_result = {
                    "best": None,
                    "scores": []
                }

            if not yolo_enabled:

                latest_yolo_result = {
                    "detections": []
                }

            if not pose_enabled:

                latest_pose_result = {
                    "persons": []
                }

            if (
                source_mode
                in (
                    "upload",
                    "video",
                    "camera_frozen"
                )
                and active_frame is not None
            ):
                refresh_frame = (
                    active_frame.copy()
                )

                refresh_clip = (
                    clip_enabled
                )

                refresh_yolo = (
                    yolo_enabled
                )

                refresh_pose = (
                    pose_enabled
                )

        if refresh_frame is not None:

            if refresh_clip:
                process_frame(
                    refresh_frame
                )

            if refresh_yolo:
                process_yolo_frame(
                    refresh_frame
                )

            if refresh_pose:
                process_pose_frame(
                    refresh_frame
                )

    with state_lock:

        return jsonify({
            "clip_enabled": clip_enabled,
            "clip_loaded": (
                model is not None
            ),
            "yolo_enabled": yolo_enabled,
            "yolo_loaded": (
                yolo_model is not None
            ),
            "pose_enabled": pose_enabled,
            "pose_loaded": (
                pose_model is not None
            ),
            "sam_enabled": sam_enabled,
            "sam_loaded": (
                sam_predictor is not None
            ),
            "gemma_enabled": gemma_enabled,
            "gemma_loaded": (
                gemma_model is not None
            ),
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
        "enhancement_enabled": enhancement_enabled,
        "enhancement_loaded": (
            realesrgan_upsampler
            is not None
        ),
        "enhancement_model": "RealESRGAN_x2plus",
        "anime_enabled": anime_enabled,
        "anime_loaded": (
            animegan_model
            is not None
        ),
        "anime_loaded_style": (
            animegan_loaded_style
        ),
        "anime_model": "AnimeGANv2",
        "anime_style": anime_style,
        "clip_enabled": clip_enabled,
        "yolo_enabled": yolo_enabled,
        "pose_enabled": pose_enabled,
        "sam_enabled": sam_enabled,
        "gemma_enabled": gemma_enabled,
        "gemma_model": GEMMA_MODEL,
        "source": source_mode,
        "num_prompts": len(PROMPTS),
        "endpoints": {
            "recognition": "/clip",
            "prompt_test": "/clip_prompt?prompt=a book",
            "yolo": "/yolo",
            "pose": "/pose",
            "sam": "/sam",
            "gemma": "/gemma",
            "control": "/control",
            "enhancement": "/enhancement",
            "anime": "/anime",
            "source": "/source",
            "source_upload": "/source/upload",
            "source_camera": "/source/camera",
            "source_camera_frame": "/source/camera_frame",
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
