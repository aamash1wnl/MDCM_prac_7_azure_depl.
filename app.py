from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved decision tree pipeline
with open('decision_tree_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form['pregnancies']),
                    float(request.form['glucose']), 
                    float(request.form['bloodpressure']),
                    float(request.form['skinthickness']),
                    float(request.form['insulin']),
                    float(request.form['bmi']),
                    float(request.form['age']),
                    float(request.form['DiabetesPedigreeFunction'])
                    ]

        features = np.array(features).reshape(1, -1)

        prediction = pipeline.predict(features)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
