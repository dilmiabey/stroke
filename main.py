from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model
xgboost_model = xgb.Booster()
xgboost_model.load_model('xgboost_model.model')

# Define preprocessing steps
# Example: Define column transformer for preprocessing categorical and numerical features
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the pipeline including preprocessing and XGBoost model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgboost_model)
])

@app.route('/')
def index():
    return render_template('index.html')

X_train = pd.read_csv('./X_train.csv')
preprocessor.fit(X_train)
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    gender = request.form.get('gender')
    age = float(request.form.get('age'))
    hypertension = int(request.form.get('hypertension'))
    heart_disease = int(request.form.get('heart_disease'))
    ever_married = request.form.get('ever_married')
    work_type = request.form.get('work_type')
    Residence_type = request.form.get('Residence_type')
    avg_glucose_level = float(request.form.get('avg_glucose_level'))
    bmi = float(request.form.get('bmi'))
    smoking_status = request.form.get('smoking_status')

    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    }


    # Prepare input data for prediction
    input_data=pd.DataFrame(data)

    #input_data = np.array([[ gender,age, hypertension, heart_disease, ever_married,work_type,  Residence_type,avg_glucose_level, bmi,  smoking_status]])
    print(input_data)
    #input_data_encoded = pipeline.named_steps['preprocessor'].transform(input_data)
    #print(input_data_encoded)
    # Make prediction
    input_data_encoded = pd.get_dummies(input_data)
    print('input_data_encoded {}'.format(input_data_encoded))
    input_data_dmatrix = xgb.DMatrix(input_data_encoded)
    # Convert DMatrix object to numpy array
    print('input_data_matrix  {}'.format(input_data_dmatrix))
    input_data_array = input_data_dmatrix.get_data()
    print('input_daa array  {}'.format(input_data_array))


    # Reshape input data if it contains a single sample
    if input_data_array.ndim == 1:
        input_data_array = input_data_array.reshape(1, -1)
        print(input_data_array.ndim)

    #prediction = pipeline.predict(input_data_dmatrix)  # Assuming binary classification, returning probability of positive class
    # Convert input_data numpy array to a DataFrame
    #input_df = pd.DataFrame(input_data_encoded)
    prediction = pipeline.predict(input_data_array)

    # Make prediction
    #prediction = pipeline.predict(input_df)
    # Return prediction result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
