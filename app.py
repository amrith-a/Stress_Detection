import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template


with open('Model.pickle', 'rb') as f:
    model,count_vect = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data= request.form['message']
        input_query = count_vect.transform([data]).toarray()
        result = model.predict(input_query)

        return render_template("test.html", data=json.dumps(str(result)))


if __name__ == '__main__':
    app.run(debug=True)
