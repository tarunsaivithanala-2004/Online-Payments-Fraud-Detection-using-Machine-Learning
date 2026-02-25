# ----------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------
from flask import Flask, render_template, request
import numpy as np
import pickle

# ----------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------
model = pickle.load(open("model/payments.pkl", "rb"))

# ----------------------------------------
# CREATE FLASK APP
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# HOME PAGE
# ----------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------------------
# PREDICT PAGE
# ----------------------------------------
@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# ----------------------------------------
# PREDICTION LOGIC
# ----------------------------------------
@app.route("/pred", methods=["POST"])
def predict():

    # Get values from form
    step = float(request.form["step"])
    type_ = float(request.form["type"])
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])

    # ADD 8th FEATURE (isFlaggedFraud = 0)
    isFlaggedFraud = 0.0

    # VERY IMPORTANT: order must match training
    features = np.array([[step, type_, amount,
                          oldbalanceOrg, newbalanceOrig,
                          oldbalanceDest, newbalanceDest,
                          ]])

    # Prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "is Fraud"
    else:
        result = "is not Fraud"

    return render_template("submit.html",
                           prediction_text=f"The predicted fraud for the online payment is ['{result}']")


# ----------------------------------------
# RUN APP
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)