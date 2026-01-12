from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os, uuid, json
from functools import wraps
from db import get_db, close_db

app = Flask(__name__)
@app.teardown_appcontext
def teardown_db(exception):
    close_db(exception)

app.secret_key = "replace_with_a_secure_random_key"

# ---------------- Load Model ---------------- #
model = load_model("skin_disease_model.h5")
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# ---------------- Decorators ---------------- #
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ---------------- Image Preprocess ---------------- #
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------- Routes ---------------- #
@app.route("/")
def home():
    return redirect(url_for("index_page"))

@app.route("/index")
def index_page():
    return render_template("index.html")

# ---------------- SIGNUP ---------------- #
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email").lower()
        phone = request.form.get("phone")
        age = request.form.get("age")
        gender = request.form.get("gender")
        role = request.form.get("role")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if password != confirm:
            error = "Passwords do not match"
        else:
            hashed = generate_password_hash(password)
            try:
                db = get_db()
                cur = db.cursor()
                cur.execute("""
                    INSERT INTO users (name, email, phone, age, gender, role, password)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (name, email, phone, age, gender, role, hashed))
                db.commit()

                session["user"] = email
                session["role"] = role
                if role == "admin":
                    return redirect(url_for("admin_dashboard"))
                return redirect(url_for("chatbot"))

            except Exception as e:
                print("Signup Error:", e)
                error = "User already exists"
    return render_template("signup.html", error=error)

# ---------------- LOGIN ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email").lower()
        password = request.form.get("password")

        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT password, role FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        if user and check_password_hash(user[0], password):
            session["user"] = email
            session["role"] = user[1]

            cur.execute("INSERT INTO login_history (email) VALUES (?)", (email,))
            db.commit()


            if session["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("chatbot"))
        else:
            error = "Invalid credentials"
            db.close()
    return render_template("login.html", error=error)

# ---------------- LOGOUT ---------------- #
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- CHATBOT ---------------- #
@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- ADMIN DASHBOARD ---------------- #
@app.route("/admin-dashboard")
@login_required
@admin_required
def admin_dashboard():
    db = get_db()
    cur = db.cursor()
    # Fetch all patient diagnosis entries
    cur.execute("""
        SELECT id, user_email, disease, confidence, date, status, image_name
        FROM diagnosis
        ORDER BY date DESC
    """)
    patients = cur.fetchall()


    # Convert to list of dicts
    patient_list = [
        {
            "id": row[0],
            "name": row[1],
            "email": row[1],
            "disease": row[2],
            "confidence": row[3],
            "date": row[4],
            "status": row[5] or "Pending",
            "img": row[6]
        } for row in patients
    ]
    return render_template("admin_dashboard.html", patients=patient_list)

# ---------------- APPROVE / REJECT / DELETE ---------------- #
@app.route("/admin/<action>/<int:id>", methods=["POST"])
@login_required
@admin_required
def admin_action(action, id):
    db = get_db()
    cur = db.cursor()
    if action.lower() == "approve":
        cur.execute("UPDATE diagnosis SET status='Reviewed' WHERE id=?", (id,))
    elif action.lower() == "reject":
        cur.execute("UPDATE diagnosis SET status='Rejected' WHERE id=?", (id,))
    elif action.lower() == "delete":
        cur.execute("DELETE FROM diagnosis WHERE id=?", (id,))
    db.commit()
    
    return jsonify({"success": True})

# ---------------- PREDICT ---------------- #
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", filename)
    file.save(path)

    try:
        img_array = preprocess_image(path)
        pred = model.predict(img_array)
        index = int(np.argmax(pred))
        disease = class_labels[index]
        confidence = float(np.max(pred))

        db = get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO diagnosis (user_email, disease, confidence, date, status, image_name)
            VALUES (?, ?, ?, date('now'), 'Pending', ?)
        """, (session["user"], disease, confidence, filename))
        db.commit()
        

        return jsonify({"prediction": disease, "confidence": confidence})

    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

