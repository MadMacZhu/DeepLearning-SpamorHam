from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
import json
import re
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer_file = './model/tokenizer.json'
model_file = './model/spam_gru32.h5'
with open(tokenizer_file) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
model = load_model(model_file)
max_length = 20
trunc_type = 'post'

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		message = re.sub('[^A-Za-z]', ' ', message)
		message = message.lower()
		data = [message]
		sequence = tokenizer.texts_to_sequences(data)
		padded = pad_sequences(sequence,
		 					   maxlen = max_length,
		  					   truncating = trunc_type)

		my_prediction = model.predict(padded)
	return render_template('result.html',prediction = my_prediction[0][0])



if __name__ == '__main__':
	app.run(debug=True)