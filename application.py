from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            # collect form data correctly
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("race_ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score")),
            )

            # convert to DataFrame
            pred_df = data.get_data_as_DataFrame()
            print("Input DataFrame:\n", pred_df)

            # run prediction pipeline
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template("home.html", results=results[0])

        except Exception as e:
            return render_template("home.html", results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
