# ====================================================
# Visualization
# ====================================================
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import random
import tensorflow as tf # type: ignore


def show_noisy_clean_pairs(noisy, clean, num_pairs=2):
    plt.figure(figsize=(14, 8))
    idxs = random.sample(range(len(noisy)), num_pairs)
    for i, idx in enumerate(idxs):
        plt.subplot(num_pairs, 2, 2*i+1)
        plt.imshow(noisy[idx].squeeze(), cmap="gray")
        plt.axis("off"); plt.title("Noisy Augmented")
        
        plt.subplot(num_pairs, 2, 2*i+2)
        plt.imshow(clean[idx].squeeze(), cmap="gray")
        plt.axis("off"); plt.title("Clean")
    plt.show()

def visualize_prediction( noisy_img, clean_img, idx=0, mdl = tf.keras.models.load_model("experiments/checkpoints/unet_best.keras", compile=True)):

    # Ensure correct shape (batch, h, w, c)
    noisy = noisy_img[idx]
    clean = clean_img[idx]
    model = mdl

    # Model prediction
    pred = model.predict(noisy[np.newaxis, ...])[0]

    # Squeeze channels if grayscale
    noisy_disp = np.squeeze(noisy)
    clean_disp = np.squeeze(clean)
    pred_disp = np.squeeze(pred)

    # Plot
    plt.figure(figsize=(15, 5))
    
    titles = ["Given (Noisy Input)", "Ground Truth (Clean)", "Model Output (Denoised)"]
    images = [noisy_disp, clean_disp, pred_disp]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i], fontsize=14)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
