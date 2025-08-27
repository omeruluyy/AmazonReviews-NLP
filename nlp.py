
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


## text preprocessing
df = pd.read_csv("datasets/amazon_reviews.csv",sep=",")
df.head()


## Normalizing case folding

df['reviewText']= df['reviewText'].str.lower()
print(df['reviewText'])

## remove punctuation
df['reviewText']= df['reviewText'].str.replace('[^\w\s]','')

## remove numbers
df['reviewText']= df['reviewText'].str.replace('\d','')