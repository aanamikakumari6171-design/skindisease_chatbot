
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os, uuid, json
from functools import wraps

app = Flask(__name__)
app.secret_key = "replace_with_a_secure_random_key"  # change in production

# ---------------- Load Model ---------------- #
model = load_model("skin_disease_model.h5")
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# ---------------- Users Storage ---------------- #
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": {}}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- Login Required Decorator ---------------- #
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ---------------- Image Preprocess ---------------- #
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Routes ---------------- #
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not email or not password:
            error = "Email and password required."
        else:
            data = load_users()
            if email in data["users"]:
                error = "Account already exists."
            else:
                hashed = generate_password_hash(password)
                data["users"][email] = {"password": hashed}
                save_users(data)
                session["user"] = email
                return redirect(url_for("index_page"))

    return render_template("signup.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        data = load_users()
        user = data["users"].get(email)
        if user and check_password_hash(user["password"], password):
            session["user"] = email
            return redirect(url_for("index_page"))
        else:
            error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/index")
@login_required
def index_page():
    return render_template("index.html")

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/result")
@login_required
def result():
    return render_template("result.html")

# ---------------- Prediction API ---------------- #
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    os.makedirs("uploads", exist_ok=True)
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        result = class_labels[predicted_class_index]
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ---------------- Run ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
