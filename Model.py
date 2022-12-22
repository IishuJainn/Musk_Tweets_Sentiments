import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from datetime import datetime
df=pd.read_csv("rawdata.csv")
print(df.head())
print(df.info())
def cleantwt (twt):
  emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
  twt=re.sub("RT"," ",twt)
  twt=re.sub("#[A-Za_z0-9]+","",twt)
  twt=re.sub("\\n","",twt)
  twt=re.sub("https?:\/\/\S+", "", twt)
  twt = re.sub('@[\S]*', '', twt)  # remove @mentions
  twt = re.sub('^[\s]+|[\s]+$', '', twt)  # remove leading and trailing whitespaces
  twt = re.sub(emoj, '', twt)  # remove emojis
  return twt

df["Cleaned_Tweets"]=df["Tweets"].apply(cleantwt)
print(df.head())

df.drop(df[df["Cleaned_Tweets"]==""].index,inplace=True)
print(df.duplicated().sum())

def getSubjectivity(twt):
  return TextBlob(twt).sentiment.subjectivity

def getPolarity(twt):
  return TextBlob(twt).sentiment.polarity

df['Subjectivity'] = df['Cleaned_Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Cleaned_Tweets'].apply(getPolarity)

def set_pol(num):
    if num>0:
        return "Positive"
    elif num==0:
        return "Neutral"
    else:
        return "Negative"
df['Sentiment'] = df['Polarity'].apply(set_pol)
print(df.head())

sns.set_style('darkgrid')
plt.figure(figsize = (8,6))
markers = {'Positive':'o', 'Neutral':'s','Negative':'X'}
sns.scatterplot(data=df, x='Polarity', y='Subjectivity', hue = 'Sentiment', style = 'Sentiment', markers = markers, palette = 'gist_earth').set(xlim=(-1,1))
plt.title('Scatter Plot')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.tight_layout()

plt.figure(figsize = (8,6))
sns.histplot(df, x = 'Sentiment', color = '#ccd5ae', shrink = 0.9).set(xlabel = None)
plt.title('Number of tweets by Sentiment')
plt.tight_layout()

df['Month'] = pd.DatetimeIndex(df['Date']).month
plt.figure(figsize = (10, 6))
months = df['Month'].unique()
sns.lineplot(x = 'Month', y = 'Likes', hue = 'Sentiment', ci = None, data = df, palette = 'viridis')
plt.title('Number of likes by sentiment and month')
plt.xlabel('Month')
plt.xticks(ticks = [m for m in months])
plt.tight_layout()

plt.figure(figsize = (10, 5))
data = df['Sentiment'].value_counts()
colors = sns.color_palette('GnBu')
labels = ['Positive', 'Neutral', 'Negative']
plt.pie(data, labels = labels, autopct = '%.0f%%', colors = colors)
plt.title('Number of likes by sentiment')
plt.tight_layout()

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.tight_layout()
plt.show()