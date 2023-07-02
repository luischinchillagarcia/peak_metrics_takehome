
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

from utils.transfer_learning_models import SentimentTask

nltk.download('stopwords')
sns.set_theme(style="darkgrid")


class Topics:
    
    @staticmethod
    def filter_sentiment(df, feature, sentiment):
        df = df[~df[feature].isna()]
        df = df[df[f'{feature}_sentiment'] == sentiment]
        return df
    
    @staticmethod
    def all_topics(df, feature):
        
        tfidf = TfidfVectorizer()
    
        feature_nonnull = df[feature]
        tfidf.fit(feature_nonnull)
        
        topics = tfidf.vocabulary_.items()
        topics = pd.DataFrame(topics, columns=['topic', 'count'])
        topics = topics[~topics.topic.isin(stopwords.words('english'))]
        topics = topics.sort_values(by='count', ascending=False)
        return topics
    
    @staticmethod
    def top_topics(topics, filter_by_sentiment):
        topics['sentiment'] = topics.topic.map(lambda topic: SentimentTask.model(topic))
        topics.sentiment = topics.sentiment.map(lambda sentiment: sentiment[0]['label'])
        topics = topics[topics.sentiment == filter_by_sentiment]
        return topics
    
    @staticmethod
    def plot(df, title, top_k=15, show=False, save_to=None):
        _, ax = plt.subplots(figsize = (10,4))

        ax = sns.barplot(x='topic', y='count', data=df.head(top_k))
        ax.tick_params(axis='x', labelrotation = 45)
        ax.set_title(title)
        
        if show:
            plt.show()
        
        if save_to:
            plt.savefig(save_to)
        return None
