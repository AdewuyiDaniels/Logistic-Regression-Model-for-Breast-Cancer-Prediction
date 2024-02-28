from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and dataset
file_path = r"C:\Users\USER PC\Downloads\BreastDatasets\Coimbra_breast_cancer_dataset.csv"
df = pd.read_csv(file_path)

# Extract features (X) and target variable (y)
features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
X = df[features]
y = df['Classification']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        input_features = [
            float(request.form['age']),
            float(request.form['bmi']),
            float(request.form['glucose']),
            float(request.form['insulin']),
            float(request.form['homa']),
            float(request.form['leptin']),
            float(request.form['adiponectin']),
            float(request.form['resistin']),
            float(request.form['mcp1'])
        ]
        
        # Make predictions
        input_data = [input_features]
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        # Get the probability of the prediction
        likelihood_percentage = model.predict_proba(input_data_scaled)[0][1] * 100

        # Display the result
        return render_template('result.html', prediction=prediction, likelihood_percentage=likelihood_percentage)

    # If the method is not POST, redirect to the index or display an error message
    return render_template('error.html', error_message="Invalid request method")

if __name__ == '__main__':
    app.run(debug=True)
