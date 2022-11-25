import numpy as np
import pandas as pd
import re
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import streamlit as st


# Carregando Modelos
bert_model = TFBertForSequenceClassification.from_pretrained('bert_model', local_files_only=True)
tokenizer = BertTokenizer.from_pretrained('tokenizer', local_files_only=True)


# Funções
def encoder(text):
  tokens = tokenizer.batch_encode_plus(text ,max_length = 65, pad_to_max_length = True,return_attention_mask = True, truncation = True)
  return tokens

def clean(text):
    
    # remove links
    text = re.sub(r'http\S+',' ', text)
    
    # remove email
    text = re.sub(r'\S*@\S*\s?',' ', text)
    
    # substitui & por e
    text = re.sub (r"&","e",text)
    
    # deixa apenas letras e !
    text = re.sub("[^a-zA-Z!]", ' ', text)
    
    # tirar os espaços a mais
    text = re.sub(r"\s+"," ",text)
    
    # Colocando o texto em lower case
    return text.lower()


def sentiment(text):
  text = pd.DataFrame([text], columns=['text'])
  text_clean = text.text.apply(clean)
  text_encoder = encoder(text_clean.tolist())
  text_inputIds = np.asarray(text_encoder['input_ids'])
  text_mask = np.array(text_encoder['attention_mask'])
  data_val = [text_inputIds, text_mask]
  predict = bert_model.predict(data_val)
  pred = np.argmax(predict.logits, axis=1)
  if pred[0] == 0:
    return 'Sentimento Negativo'
  else:
    return 'Sentimento Positivo'

# App
st.title('Análise de sentimentos Olist')

st.markdown('Preencha a caixa de texto abaixo e espere o resultado')

review = st.text_input('Review')

sentimento = sentiment(review)

if st.button('Previsão'):
  if sentimento == 'Sentimento Positivo':
    st.success(sentimento)
  else:
    st.error(sentimento)