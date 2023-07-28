from flask import Flask, render_template, request
import pandas as pd
import pickle as pkl
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Programs\Cleaned_data.csv')
pipe = pkl.load(open('Programs\RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods = ['POST'])
def predict():
    # to access the requests when user selects
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    # printing the prediction total of model
    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,sqft, bath, bhk]], columns = ['location','total_sqft','bath','bhk'])
    # at [0] index we get prediction that is why we use [0]
    prediction = pipe.predict(input)[0] * 1e5
    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug= True, port = 5001)