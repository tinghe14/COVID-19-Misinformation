import streamlit as st
import numpy as np
import pandas as pd
import tweepy 
import datetime as dt
from pages.BERT.TweetsPredictor import TweetsPredictor
from pages.BERT.BertClassifier import BertClassifier
import torch

st.title('COVID Misleading Informaiton Detection 🔥')
# st.header("Timely Prediction 🚨")
st.sidebar.markdown("Timely Prediction 🚨")

# TOKEN
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAE70VgEAAAAA4r5dhYN1Cx3lvTddukt8uVNqwPg%3Di3ydR4w3LI2XPkN2bxD41aPZ0BhzD8g5cpEHJnKSvJ4BLEWAUC'
auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth) 

# MAIN TABLE
st.subheader('Retrive the Most Recent Tweets Given the COVID related Keyword')
with st.form(key='Enter name'):
    search_words = st.text_input('Key Word:')
    num_of_tweets = st.number_input('Number of the Latest Tweets You Want to Retrieve:',0, 50, 10)
    submit_button = st.form_submit_button(label='Submit & Rerun Model')

if submit_button:
    with st.spinner(f'Searching and Modeling ...'):
        predict_list=[]
        tweets = tweepy.Cursor(api.search_tweets, q=search_words, lang='en').items(num_of_tweets)
        tweet_list = [i.text for i in tweets]
        predict = [0]*num_of_tweets
        device = torch.device('cpu')
        pretrain_model = BertClassifier()
        pretrain_model.load_state_dict(torch.load('/Users/tinghe/Github Repos/COVID-19-Misinformation/bert_classifer0817.pth', map_location=device))
        for i in tweet_list:
            tweet= TweetsPredictor(i, pretrain_model)
            predict = tweet.predict(i)
            predict_np = predict[0][:,1]
            predict_list.append(predict_np.numpy())
        df = pd.DataFrame(list(zip(tweet_list, predict_list)), columns=['The Latest '+str(num_of_tweets)+' Tweets '+'on '+search_words, 'Percentage of being misleading Info'])
        st.dataframe(df)


