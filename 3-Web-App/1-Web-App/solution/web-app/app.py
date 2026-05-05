import hashlib
import os
import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

def _verify_model_integrity(path):
    expected = os.environ.get("MODEL_SHA256", "")
    if not expected:
        raise RuntimeError("MODEL_SHA256 environment variable must be set to the expected SHA-256 hex digest of the model file")
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    if digest != expected:
        raise RuntimeError("Model integrity check failed: file hash does not match MODEL_SHA256")

_MODEL_PATH = "../ufo-model.pkl"
_verify_model_integrity(_MODEL_PATH)
model = joblib.load(_MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=False)
