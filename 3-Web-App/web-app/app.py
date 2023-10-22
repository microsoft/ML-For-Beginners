## import the needed libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

# filling in the flask boilerplate code
app = Flask(__name__)

## importing the model
model = pickle.load(open("./ufo-model.pkl", "rb"))

## routing the model onto the web application
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods= ["POST"])
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
    app.run(debug=True)