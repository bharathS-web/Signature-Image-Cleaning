import cv2 # type: ignore
import numpy as np
import random
import string
from PIL import Image, ImageDraw, ImageFont # type: ignore 
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore


# ====================================================
# Word/Text Generators
# ====================================================
primary_words = [
    "Board of Trustees", "Trustee", "Meeting", "Board", 
    "Best Wishes", "Yours Truly", "Sincerely", "President", 
    "Chairman", "Phd"
]

words = ["signature", "sample", "document", "verify", "authentic"]

font_list = []
font_list = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # "arial.ttf"
]


def get_pil_font():
    font_path = random.choice(font_list)
    try:
        return ImageFont.truetype(font_path, random.randint(25, 55))
    except IOError:
        return ImageFont.load_default()

def get_word():
    word = random.choice(words)
    if random.random() > 0.7:
        word += " " + random.choice(primary_words)
    prob = random.random()
    if prob < 0.25: return word.lower()
    if prob < 0.75: return word.upper()
    return word.capitalize()

def get_text():
    punctuations = string.punctuation
    text = f"{get_word()} {random.choice(punctuations)} {get_word()}"
    if random.random() > 0.5:
        text += " " + get_word()
    return text

def draw_line(img_size=(224,224)):
    img = Image.new("L", img_size, color=255)
    draw = ImageDraw.Draw(img)
    w, h = img_size
    y_pos = random.choice([int(h*0.2), int(h*0.5), int(h*0.8)])
    draw.line([(random.randint(0,80), y_pos), (w-random.randint(0,80), y_pos)], 
              fill="black", width=random.randint(1, 6))
    text = get_text()
    draw.text((random.randint(30, 100), y_pos + random.randint(2, 20)), 
              text, fill="black", font=get_pil_font())
    return img

def draw_text(img_size=(224,224)):
    img = Image.new("L", img_size, color=255)
    draw = ImageDraw.Draw(img)
    w, h = img_size
    text = get_text()
    draw.text((random.choice([0, w//10, w//5]), random.choice([h//5, h//2, int(h*0.8)])), 
              text, fill="black", font=get_pil_font())
    return img

# ====================================================
# Classic Noise Functions
# ====================================================
def add_gaussian_noise(img, mean=0, var=0.00001):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    noisy = img / 255.0 + noise
    noisy = np.clip(noisy, 0., 1.)
    return (noisy * 255).astype(np.uint8)

def add_salt_pepper(img, amount=0.00004, s_vs_p=0.5):
    out = np.copy(img)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out

def add_speckle(img):
    noise = np.random.randn(*img.shape)
    noisy = img / 255.0 + img / 255.0 * noise
    noisy = np.clip(noisy, 0., 1.)
    return (noisy * 255).astype(np.uint8)

def add_motion_blur(img, kernel_size=3):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def add_random_occlusion(img, size=30):
    out = np.copy(img)
    h, w = img.shape[:2]
    x, y = np.random.randint(0, w-size), np.random.randint(0, h-size)
    out[y:y+size, x:x+size] = 0
    return out

# ====================================================
# Noise Selector
# ====================================================
def apply_random_noise(img):
    ch = ["gaussian", "s&p", "speckle", "motion", "occlusion", "none"] + \
        ["gaussian", "s&p", "speckle", "motion", "occlusion"]
    choice = random.choice(ch)
    if choice == "gaussian": return add_gaussian_noise(img)
    if choice == "s&p": return add_salt_pepper(img)
    if choice == "speckle": return add_speckle(img)
    if choice == "motion": return add_motion_blur(img, kernel_size=random.choice([3,5,7]))
    if choice == "occlusion": return add_random_occlusion(img)
    return img


# ====================================================
# Data Augmentation (Clean vs Noisy Pairs)
# ====================================================
def threshold_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def load_img_np(path, target_size=(224,224)):
    img = load_img(path, target_size=target_size, color_mode="rgb")
    return img_to_array(img).astype("uint8")

def get_aug(img_paths, img_target_size=(224,224)):
    clean_holder, aug_holder = [], []
    
    for path in img_paths:
        img_np = load_img_np(path, target_size=img_target_size)
        clean_img = threshold_image(img_np)
        clean_holder.append(clean_img / 255.)

        # add background clutter
        bg = draw_line(img_size=img_target_size)
        if random.random() > 0.5: bg.paste(draw_text(img_target_size), (0,0), bg)
        if random.random() > 0.5: bg.paste(draw_text(img_target_size), (0,0), bg)

        image = Image.fromarray(clean_img).convert("RGB")
        image.paste(bg, (0,0), image.convert("L"))

        aug_img = np.array(image)
        aug_img = apply_random_noise(aug_img)
        aug_img = threshold_image(aug_img)
        aug_holder.append(aug_img / 255.)


    clean_holder = np.expand_dims(np.array(clean_holder), -1)
    aug_holder   = np.expand_dims(np.array(aug_holder), -1)
    return clean_holder, aug_holder
