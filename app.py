from flask import Flask, render_template, url_for, request
import pandas as pd 
import random
import pickle
import json
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer_file = './model/tokenizer.json'
model_file = './model/spam_gru32.h5'
with open(tokenizer_file) as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
model = load_model(model_file)
max_length = 20
trunc_type = 'post'

df = pd.read_csv('spam.csv', encoding = 'latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
df.dropna(inplace = True)
messages = df['message'].values.tolist()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html', sample_messages = None)

@app.route('/', methods=['POST'])
def generate():
	if request.method == 'POST':
		sample_texts = random.sample(messages, 2)
		for i in range(2):
			if len(sample_texts[i]) > 120:
				sample_texts[i] = sample_texts[i][ :120]
	return render_template('home.html', sample_messages = sample_texts)

@app.route('/predict', methods=['POST'])
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