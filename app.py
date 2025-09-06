import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageOps

# =====================================
# Load model
# =====================================
MODEL_PATH = r"experiments/checkpoints/unet_best.keras"
model = load_model(MODEL_PATH, compile=False)

# Automatically match model input size
IMG_SIZE = model.input_shape[1:3]  # (224, 224)

# =====================================
# Helper functions
# =====================================
def preprocess_image(img: Image.Image):
    img = ImageOps.grayscale(img)
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (h, w, 1)
    arr = np.expand_dims(arr, axis=0)    # (1, h, w, 1)
    return arr

def postprocess_image(arr):
    arr = np.squeeze(arr) * 255.0
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr)

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="Signature Cleaner", layout="wide")
st.title("🖊️ Signature Denoising with U-Net")

option = st.radio("Choose input method:", ["Draw on Canvas", "Upload Image"])

col1, col2 = st.columns(2)

if option == "Draw on Canvas":
    with col1:
        st.subheader("Draw your signature:")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=256,
            height=128,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Submit Signature", key="canvas_submit"):
            if canvas_result.image_data is not None:
                img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"))
                inp = preprocess_image(img)
                pred = model.predict(inp)[0]
                out_img = postprocess_image(pred)

                col2.subheader("Result:")
                st.image([img, out_img], caption=["Input", "Model Output"], width=250)

elif option == "Upload Image":
    with col1:
        st.subheader("Upload an image of a signature")
        file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if file is not None:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", width=250)

            if st.button("Submit Image", key="upload_submit"):
                inp = preprocess_image(img)
                pred = model.predict(inp)[0]
                out_img = postprocess_image(pred)

                col2.subheader("Result:")
                st.image([img, out_img], caption=["Input", "Model Output"], width=250)
