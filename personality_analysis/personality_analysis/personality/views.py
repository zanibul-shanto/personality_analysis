# string_input_app/views.py
import os
from django.shortcuts import render
import json

# def test(s):
#   s 
#   sentences = np.asarray([s])
#   enc_sentences = prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
#   predictions = model.predict(enc_sentences)
#   for sentence, pred in zip(sentences, predictions):
#       pred_axis = []
#       mask = (pred > 0.5).astype(bool)
#       for i in range(len(mask)):
#           if mask[i]:
#             pred_axis.append(axes[i][2])
#           else:
#             pred_axis.append(axes[i][0])
#       print('-- comment: '+sentence.replace("\n", "").strip() +
#             '\n-- personality: '+str(pred_axis) +
#             '\n-- scores:'+str(pred))

from transformers import TFBertModel, BertTokenizer
seed_value = 29
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
np.set_printoptions(precision=2)
import tensorflow as tf
tf.random.set_seed(seed_value)
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

N_AXIS = 4
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
'''
EMOTIONAL AXES:
Introversion (I) – Extroversion (E)
Intuition (N) – Sensing (S)
Thinking (T) – Feeling (F)
Judging (J) – Perceiving (P)
'''
axes = ["I-E","N-S","T-F","J-P"]
classes = {"I":0, "E":1, # axis 1
           "N":0,"S":1, # axis 2
           "T":0, "F":1, # axis 3
           "J":0,"P":1} # axis 4

def prepare_bert_input(sentences, seq_len, bert_name):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                max_length=seq_len)
    input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
               np.array(encodings["attention_mask"])]
    return input

def personality_test(s):
    model = create_model()
            
    model.load_weights('D://personality_analysis//personality_analysis//personality//weights.h5')
    s 
    sentences = np.asarray([s])
    enc_sentences = prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
    predictions = model.predict(enc_sentences)
    for sentence, pred in zip(sentences, predictions):
        pred_axis = []
        mask = (pred > 0.5).astype(bool)
        for i in range(len(mask)):
            if mask[i]:
                pred_axis.append(axes[i][2])
            else:
                pred_axis.append(axes[i][0])
        print('-- comment: '+sentence.replace("\n", "").strip() +
            '\nPersonality: '+str(pred_axis) + " "+
            '\nScores:'+str(pred))
        comment = 'Comment: '+sentence.replace("\n", "").strip()
        personality = 'Personality: '+str(pred_axis)
        scores = 'Scores: '+str(pred)
    return comment, personality, scores

def create_model():
    N_AXIS = 4
    MAX_SEQ_LEN = 128
    BERT_NAME = 'bert-base-uncased'
    '''
    EMOTIONAL AXES:
    Introversion (I) – Extroversion (E)
    Intuition (N) – Sensing (S)
    Thinking (T) – Feeling (F)
    Judging (J) – Perceiving (P)
    '''
    input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(N_AXIS, activation="sigmoid")(avg)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def home(request):
    if request.method == 'POST':
        # Retrieve the input string from the form
        input_string = request.POST.get('input_string')

        # Load the .pkl file
        # with open('D://personality_analysis//personality_analysis//personality//personality.pkl', 'rb') as file:
        #     model = pickle.load(file)
        # with open('D://personality_analysis//personality_analysis//personality//personality.pkl', 'rb') as f:
        #     pred = pickle.load(f)

            
 

        # Process the input string using the loaded model
        comment, personality, scores = personality_test(input_string)

        # Pass the input, output, and the form to the template for rendering
        context = {
            'comment': comment,
            'personality': personality,
            'scores': scores
        }
        print(context)
        return render(request, 'input.html', context)
    
    return render(request, 'input.html')
