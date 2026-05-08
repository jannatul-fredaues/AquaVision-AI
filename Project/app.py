import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
model = tf.keras.models.load_model("fish_model.h5")

# -----------------------------
# LOAD LABELS (SAFE WAY)
# -----------------------------
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = ["Betta", "Guppy", "Goldfish", "Crayfish", "Koi", "Tilapia", "Salmon", "Catfish", "Tuna", "Angelfish", "Discus", "Oscar"]  # fallback

IMG_SIZE = 224

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_fish(image):
    try:
        # Convert image safely
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        result = f"Fish: {class_names[class_index]}\nConfidence: {confidence*100:.2f}%"

        return result

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# GRADIO UI (CLEAN & SAFE)
# -----------------------------
interface = gr.Interface(
    fn=predict_fish,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="🐟 Fish Classification AI",
    description="Upload a fish image and get prediction instantly."
)

# -----------------------------
# LAUNCH (IMPORTANT FOR SPACES)
# -----------------------------
if __name__ == "__main__":
    interface.launch()