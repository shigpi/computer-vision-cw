import os
import threading

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

#  Page config 
st.set_page_config(
    page_title="Guitar Classifier",
    page_icon="🎸",
    layout="centered",
)

#  Constants 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "finetuned_cnn", "resnet18_deploy.pth")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "finetuned_cnn",
    "resnet18_deploy.pth"
)

# Create folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download model only if it doesn't already exist
from huggingface_hub import hf_hub_download
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Hugging Face...")

    downloaded_path = hf_hub_download(
        repo_id="kkarhm/guitar-type-detector-resnet18",
        filename="resnet18_deploy.pth",
        local_dir=os.path.dirname(MODEL_PATH),
        local_dir_use_symlinks=False
    )

    # Rename/move to your exact MODEL_PATH if needed
    if downloaded_path != MODEL_PATH:
        os.replace(downloaded_path, MODEL_PATH)

    print(f"Model downloaded to: {MODEL_PATH}")
else:
    print("Model already exists locally.")

CLASS_ICONS = {
    "acoustic": "🎸",
    "electric": "⚡",
    "bass":     "🎵",
}

CLASS_COLORS = {          # BGR for OpenCV overlay
    "acoustic": (50, 200, 50),
    "electric": (50, 150, 255),
    "bass":     (200, 80, 200),
}

CLASS_DESCRIPTIONS = {
    "acoustic": "Hollow body: produces sound naturally without amplification.",
    "electric": "Solid body: requires an amplifier to project sound.",
    "bass":     "Typically 4 strings: provides the low-end foundation.",
}

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


#  Model loader (cached so it only loads once) 
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    class_names = checkpoint["class_names"]
    image_size  = checkpoint["image_size"]
    mean        = checkpoint["imagenet_mean"]
    std         = checkpoint["imagenet_std"]
    num_classes = checkpoint["num_classes"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, transform, class_names


#  Inference helper
def predict(model, transform, class_names, image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()
    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top_class = max(scores, key=scores.get)
    return top_class, scores


#  Video frame processor 
class GuitarVideoProcessor:
    """
    Processes each webcam frame:
      - Runs inference every N frames to keep it smooth
      - Draws an overlay with the predicted class + confidence
    """

    def __init__(self, model, transform, class_names):
        self.model       = model
        self.transform   = transform
        self.class_names = class_names

        self._lock        = threading.Lock()
        self._label       = "Detecting…"
        self._confidence  = 0.0
        self._color       = (200, 200, 200)
        self._frame_count = 0
        self._run_every   = 8   # classify every N frames

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        self._frame_count += 1
        if self._frame_count % self._run_every == 0:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            top_class, scores = predict(
                self.model, self.transform, self.class_names, pil_img
            )
            confidence = scores[top_class] * 100
            color      = CLASS_COLORS.get(top_class, (200, 200, 200))
            label      = f"{top_class.upper()}  {confidence:.0f}%"

            with self._lock:
                self._label      = label
                self._confidence = confidence
                self._color      = color

        # Draw overlay on every frame using the last prediction
        with self._lock:
            label = self._label
            color = self._color

        h, w = img_bgr.shape[:2]

        # Semi-transparent banner at the bottom
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
        img_bgr = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)

        # Label text
        cv2.putText(
            img_bgr, label,
            (14, h - 18),
            cv2.FONT_HERSHEY_DUPLEX, 0.95,
            color, 2, cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


#  App header ─
st.title("🎸 Guitar Type Classifier")
st.caption(
    "Classify guitars as **acoustic**, **electric**, or **bass**: "
    "from a photo or live via your webcam."
)
st.divider()

#  Load model ─
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Make sure `resnet18_deploy.pth` is inside `finetuned_cnn/`."
    )
    st.stop()

model, transform, class_names = load_model(MODEL_PATH)

#  Tabs ─
tab_upload, tab_live = st.tabs(["📁 Upload Image", "📷 Live Camera"])

#  Tab 1: Upload 
with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Classifying…"):
                top_class, scores = predict(model, transform, class_names, image)

            icon       = CLASS_ICONS.get(top_class, "🎸")
            confidence = scores[top_class] * 100

            st.subheader(f"{icon} {top_class.capitalize()}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.caption(CLASS_DESCRIPTIONS.get(top_class, ""))

            st.divider()
            st.write("**All class probabilities**")
            for cls in sorted(scores, key=scores.get, reverse=True):
                pct = scores[cls] * 100
                st.progress(
                    pct / 100,
                    text=f"{CLASS_ICONS.get(cls, '')} {cls.capitalize()} :  {pct:.1f}%",
                )
    else:
        st.info("Upload a guitar image above to get a prediction.")


#  Tab 2: Live Camera
with tab_live:
    st.write("Point your webcam at a guitar: the model classifies it in real time.")
    st.caption("Prediction updates every few frames and is overlaid directly on the video.")

    webrtc_streamer(
        key="guitar-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: GuitarVideoProcessor(
            model, transform, class_names
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
