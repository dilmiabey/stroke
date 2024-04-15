from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import joblib

app = Flask(__name__)

# Load your trained XGBoost model
#xgboost_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
#xgboost_model.load_model("xgboost_model.model")  # Load your trained model here

# Load the trained XGBoost model

pipeline_with_preprocessor = joblib.load('xgboost_model_with_pipeline.joblib')
# Extract the preprocessor and the XGBoost model from the pipeline
preprocessor = pipeline_with_preprocessor.named_steps['preprocessor']
xgboost_model = pipeline_with_preprocessor.named_steps['classifier']
has_transform = hasattr(preprocessor, 'transform')
print("Preprocessor has transform method:", has_transform)


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_data = {
        "gender": request.form['gender'],
        "age": float(request.form['age']),
        "hypertension": int(request.form['hypertension']),
        "heart_disease": int(request.form['heart_disease']),
        "ever_married": request.form['ever_married'],
        "work_type": request.form['work_type'],
        "Residence_type": request.form['residence_type'],
        "avg_glucose_level": float(request.form['avg_glucose_level']),
        "bmi": float(request.form['bmi']),
        "smoking_status": request.form['smoking_status']
    }

    # Preprocess input data
    input_df = pd.DataFrame(input_data, index=[0])
    preprocessed_input = preprocessor.transform(input_df)

    # Make prediction
    prediction = xgboost_model.predict(preprocessed_input)[0]

    # Determine prediction result
    if prediction == 1:
        result = "likely to have a stroke"
    else:
        result = "unlikely to have a stroke"

    return render_template('result2.html', result=result)

#if __name__ == '__main__':
    #app.run(debug=True)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
