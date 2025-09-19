
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import json

app = Flask(__name__)

# ---------------- Load Trained Model ---------------- #
model = load_model("skin_disease_model.h5")
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)
# Labels of your dataset
# class_labels = ["Acne", "Candidiasis","Eczema", "Psoriasis","Actinic_Keratosis","Benign_tumors","Bullous","DrugEruption","Lichen","Lupus","Infestations_Bites"] #

# ---------------- Preprocess Function ---------------- #
def preprocess_image(img_path):
    """
    Load image and preprocess it to match the model input
    """
    # Make sure color_mode matches your trained model (rgb or grayscale)
    img = image.load_img(img_path, target_size=(128, 128), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Routes ---------------- #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index")
def index_page():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result")
def result():
    return render_template("result.html")

# ---------------- Prediction API ---------------- #

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save image temporarily with a unique filename
    os.makedirs("uploads", exist_ok=True)
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    try:
        # Preprocess image
        img_array = preprocess_image(filepath)

        # Predict
        prediction = model.predict(img_array)
        print("Raw model prediction:", prediction)  # Debugging

        predicted_class_index = np.argmax(prediction)  # Safer than axis=1
        result = class_labels[predicted_class_index]

        print("Predicted class:", result)  # Debugging

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

# ---------------- Run Flask ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
