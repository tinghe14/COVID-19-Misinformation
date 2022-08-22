import re
#import nltk
import pandas as pd
from textblob import TextBlob
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from pages.BERT.BertClassifier import BertClassifier
import torch.nn as nn
from transformers import BertModel


class TweetsPredictor:

    def __init__(self, df, model):
        self.df = df
        self.model = model

    def sentiment(self):
        self.df['polarity'] = self.df['text'].apply(lambda x:TextBlob(x).sentiment.polarity)
        self.df['subjectivity'] = self.df['text'].apply(lambda x:TextBlob(x).sentiment.subjectivity)

    def clean_tweets(self, flg_stemm=False):
        # @input: pandas series 
        # @output: clean pandas series
        cleaned_tweets = []
        for line in self.df['text']:
            # clean (convert to lowercase, remove punctuations and numbers and then strip)
            line = re.sub(r'[^\w\s]', '', str(line).lower().strip())
            clean_tokens = line.split()
            # remove stop words (keep personal pronoun)
            stopwords = nltk.corpus.stopwords.words("english")
            personal_pronoun = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves']
            stopwords_without_person = list(set(stopwords) - set(personal_pronoun))
            non_stop = [clean_non_stop for clean_non_stop in clean_tokens if clean_non_stop not in stopwords_without_person]
            # remove website link by remove any string begin with https
            prefix = 'https'
            non_stop_wo_website = [x for x in non_stop if not x.startswith(prefix)]
            # stemming (remove -ing, -ly, ...)
            if flg_stemm == True:
                ps = nltk.stem.porter.PorterStemmer()
                non_stop_ps = [ps.stem(word) for word in non_stop_wo_website]
                # from list to string
                text = " ".join(non_stop_ps)
            else:
                text = " ".join(non_stop_wo_website)
            cleaned_tweets.append(text)
        return cleaned_tweets

    def clean_df(self):
        self.sentiment()
        # missing data
        self.df.dropna(subset=['text','user_location','polarity','subjectivity'], inplace=True)
        # convert them into numeric
        cat_columns = self.df.columns[~(self.df.columns.isin(['text','user_location']))]
        df_non_cat = self.df[cat_columns].apply(pd.to_numeric, errors='coerce')
        # normalize data
        for i in df_non_cat.columns:
            column_min = min(df_non_cat[i])
            column_max = max(df_non_cat[i])
            if column_min == column_max == 0:
                pass
            else: 
                df_non_cat[i] = (df_non_cat[i] - column_min) / (column_max - column_min)
        # concatenate df
        cleaned_df = pd.concat([df_non_cat, self.df[['text','user_location']]], axis=1)
        # clean tweets
        cleaned_df['clean text'] = self.clean_tweets(flg_stemm=True)
        cleaned_df['clean text without stem'] = self.clean_tweets(flg_stemm=False)
        return cleaned_df

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self,data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []
        # For every sentence...
        data = self.clean_df()['text']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for sent in data.values:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = tokenizer.encode_plus(
                text = sent,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=80,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )       
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks

    def evaluate(self, model, test_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        prob_list = []
        if torch.cuda.is_available():       
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # For each batch in our validation set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
                probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_list.append(probs)
        return prob_list

    def test(self):
        test_inputs, test_masks = self.preprocessing_for_bert(self.df['text'])
        cleaned_df =self.clean_df()
        return test_inputs.shape, test_masks.shape, cleaned_df.shape

    def predict(self, data):
        data = self.clean_df()
        test_inputs, test_masks = self.preprocessing_for_bert(data['text'])
        test_labels = torch.tensor([1]*data.shape[0]).type(torch.LongTensor)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
        probs = self.evaluate(self.model, test_dataloader)
        return probs
