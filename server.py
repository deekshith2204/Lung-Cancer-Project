import joblib
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained machine learning model
model=joblib.load("./artificts/lungcancerprediction.pickle")

@app.route('/', methods=['GET'])
def index():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    gender = request.form['gender']
    age = int(request.form['age'])
    smoking = str(request.form['smoking'])
    yellow_fingers = str(request.form['yellow_fingers'])
    anxiety =str(request.form['anxiety'])
    peer_pressure = str(request.form['peer_pressure'])
    chronic_disease = str(request.form['chronic_disease'])
    fatigue = str(request.form['fatigue '])
    allergy = str(request.form['allergy '])
    wheezing = str(request.form['wheezing'])
    alcohol_consuming = str(request.form['alcohol_consuming'])
    coughing = str(request.form['coughing'])
    shortness_of_breath = str(request.form['shortness_of_breath'])
    swallowing_difficulty = str(request.form['swallowing_difficulty'])
    chest_pain =str(request.form['chest_pain'])
    # Add more inputs as needed...
    coder = LabelEncoder()

    # Create a DataFrame with user inputs
    lung_df= pd.DataFrame({'GENDER': [gender], 'AGE': [age], 'SMOKING': [smoking],'YELLOW_FINGERS': [yellow_fingers],'ANXIETY': [anxiety],'PEER_PRESSURE': [peer_pressure],'CHRONIC_DISEASE': [chronic_disease],'FATIGUE ': [fatigue],'ALLERGY ':[allergy],'WHEEZING': [wheezing],'ALCOHOL_CONSUMING': [alcohol_consuming],'COUGHING': [coughing],'SHORTNESS_OF_BREATH': [shortness_of_breath],'SWALLOWING_DIFFICULTY': [swallowing_difficulty],'CHEST_PAIN': [chest_pain]})
    # Preprocess the data if needed (e.g., encoding categorical variables, scaling numerical variables)
        # Make predictions using the model


    prediction = model.predict(lung_df)

    # Display the prediction result
    if prediction[0] == 1:
        result = 'You have a high chances of getting the cancer please consult a doctor'
    else:
        result = 'Your health condition is good and you are away from the lung cancer'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
