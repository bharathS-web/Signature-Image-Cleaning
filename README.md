# 🖊️ Signature Image Cleaning with U-Net  

This project implements a **U-Net based deep learning model** to clean noisy or degraded handwritten signatures. The system can **denoise signatures**, **remove background clutter**, and **produce a cleaner binary representation**.  

---

## 📂 Project Structure  

Signature-Image-Cleaning/
│── experiments/
│ └── checkpoints/   # Saved model weights
│── augmentations/   # Noise + augmentation functions
│── app.py           # Streamlit web app
│── visualization.py # Visualization functions
│── model.py         # U-Net model definition
│── train.ipynb      # Training script
│── requirements.txt # Dependencies
│── README.md        # Project description

---

## 🗃️ Dataset Creation  

Since no large-scale signature denoising dataset exists, we **generated data** using augmentations:  

1. **Base Data**  
   - Collected clean handwritten signatures (PNG/JPG).  
   - Converted to grayscale and resized to **224×224**.  

2. **Noise Augmentations** (applied to simulate real-world degradations):  
   - **Gaussian Noise**  
   - **Salt & Pepper Noise**  
   - **Speckle Noise**  
   - **Random Lines/Text Overlay** (to mimic stamps, scribbles, or scratches)  
   - **Brightness/Contrast Variations**  

3. **Training Pairs**  
   - **Input**: Noisy signature  
   - **Label**: Original clean signature  

This makes the task a **supervised image-to-image translation problem**.  

---

## 🧠 Model Description  

We use a **lightweight U-Net**:  

- **Encoder**:  
  - 4 levels of convolutional blocks with **Conv2D → BatchNorm → ReLU → MaxPool**  
  - Filters: 32 → 64 → 128 → 256  

- **Bottleneck**:  
  - 512 filters (Conv2D + BatchNorm)  

- **Decoder**:  
  - 4 levels of **UpSampling → Concatenate(skip connection) → Conv2D**  
  - Filters: 256 → 128 → 64 → 32  

- **Output**:  
  - `Conv2D(1, kernel_size=1, activation="sigmoid")` → produces binary mask of cleaned signature  

👉 Optimized for **minimal computation** while keeping **segmentation accuracy high**.  

---

## 🏗️ Model Architecture

Below is the U-Net architecture used in this project:

![U-Net Architecture](unet_architecture.png)


---

## ⚙️ Training  

- **Loss**: `Binary Crossentropy`  
- **Optimizer**: Adam (`lr=1e-3`)  
- **Metrics**: Accuracy, IoU (Intersection-over-Union)  
- **Checkpointing**:  
  - Best weights stored at `experiments/checkpoints/unet_weights.weights.h5`  

---

## 🖥️ Usage  

### 1. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
run train.ipynb
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## 📊 Visualization

During training, we log input vs ground-truth vs model output:

![Model Output](output.png)

---


## 🚀 Future Work

- Add GAN-based denoising for more realistic outputs.

- Train with larger real-world datasets (scanned documents, legal papers).

- Deploy as a web service / API for integration into digital signing platforms.
