# Sentiment Analysis of Twitter Data
This project uses python libraries to perform sentiment analysis on Twitter data. The data used in this project is in the form of a csv file which contains tweets and the information associated with each tweet such as the date of the tweet, number of likes and retweets, etc. The goal of this project is to perform sentiment analysis on these tweets and perform various visualizations on the results obtained from the analysis.

## Requirements
The following libraries are required to run this project:

Pandas

Numpy

Matplotlib

Seaborn

TextBlob

re

warnings

## Data Cleaning
The data is first loaded into a Pandas dataframe. Then, the tweets are cleaned by removing special characters, emojis, @ mentions, links, etc. The cleaned tweets are then stored in a new column in the dataframe. After cleaning, the dataframe is checked for duplicates and the duplicates are removed.

## Sentiment Analysis
The sentiment analysis is performed using the TextBlob library. The subjectivity and polarity of each tweet are calculated and the polarity values are used to determine the sentiment of each tweet. The sentiment is classified as Positive, Neutral, or Negative.

## Visualizations
Various visualizations are performed on the sentiment analysis results. These visualizations include:

Scatter plot showing the polarity and subjectivity of each tweet
Histogram showing the distribution of sentiments
Line plot showing the number of likes by sentiment and by month
Bar plot showing the number of tweets by sentiment

## Conclusion
This project demonstrates the use of Python libraries to perform sentiment analysis on Twitter data. The results of the sentiment analysis are visualized using various plots to give a better understanding of the sentiment of the tweets.
