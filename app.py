#################################################################
# This file uses a streamlit app to serve the model
# الله المستعان
#################################################################

# >> Load packages, models, and word_to_idx ict
from flask import Flask, request, render_template, json
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model("models/sentiment_analysis_model_v2.keras")
with open("data/word_to_idx.pkl", 'rb') as f:
    word_to_idx = pickle.load(f)

# -----------------------------------------------------------------------------------------



# >> Inference Functions
def sentence_to_indices(sentence, word_to_idx, seq_len):
    """
    Convert a single sentence [the input sentence] to array of indices
    """
    print(seq_len)
    indices = np.zeros((1, seq_len))
    words = sentence.lower().split() 
    for j, w in enumerate(words):
        if j >= seq_len:
            break # turncates sequences longer than seq_len
        if w in word_to_idx:
            indices[0, j] = word_to_idx[w]
    return indices


def predict_sentiment(input_sentence):
    seq_len = model.input_shape[1]
    X = sentence_to_indices(input_sentence, word_to_idx, seq_len)
    pred = model.predict(X)

    if pred[0, 0] >= 0.5:
        return 'positive'
    else:
        return 'negative'
# -----------------------------------------------------------------------------------------


# >> Flask Logic
@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route("/get_sentiment", methods = ["POST"])
def get_sentiment():
    payload = request.get_json(silent = True)
    if not "sentence" in payload:
        return {
            'error'  : 1,
            'message': "Please, Enter a message"
        }

    sentence = payload["sentence"]
    sentiment = predict_sentiment(sentence)

    color = 'danger' if sentiment is 'negative' else 'success'

    return {
        "error" : 0,
        "sentiment" : sentiment,
        "color" : color
    }
# -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    app.run(
        debug = True,
        port = 9999
    )



