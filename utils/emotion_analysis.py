from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

from utils.transfer_learning_models import EmotionTaskLarge


class EmotionsAnalysis:
    
    @staticmethod
    def get_sentiment(df, sentiment, feature):
        df = df[~df[feature].isna()]
        sentiment_filter = df[f'{feature}_sentiment'] == sentiment
        df = df[sentiment_filter]
        return df
    
    @staticmethod
    def add_lg_emotions(df, feature, max_tokens):
        emotions = df[feature].map(lambda text: EmotionTaskLarge.model(text[:max_tokens]))
        emotions = emotions.map(lambda emotions: {d['label']:d['score'] for d in emotions[0]}.items())
        
        df['emotions'] = emotions
        df = df.explode('emotions')
        
        hue = df['emotions'].map(lambda emotion: emotion[0])
        emotion_percents = df['emotions'].map(lambda emotion: emotion[1])
        
        df['hue'] = hue
        df['emotion_percents'] = emotion_percents
        
        df = df[['publish_date', 'hue', 'emotion_percents']]
        return df
    
    @staticmethod
    def resample(df, by='D', strftime='%Y-%m-%d'):
        df_plt = df.set_index('publish_date').groupby('hue').resample(by).mean().fillna(0).reset_index()
        df_plt.publish_date = df_plt.publish_date.dt.strftime(strftime)
        return df_plt
    
    @staticmethod
    def filter_negative_emotions(df):
        negative_emotions = [
            'anger', 'annoyance', 'confusion', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'fear',
            'grief', 'nervousness', 'remorse', 'sadness',
        ]
        df = df[df.hue.isin(negative_emotions)]
        
        return df

    @staticmethod
    def plot(df, title, figsize=(20,6), legend=False, show=False, save_to=None):
        _, ax = plt.subplots(figsize=figsize)

        ax = sns.pointplot(x='publish_date', y='emotion_percents', hue='hue', data=df)
        ax.tick_params(axis='x', labelrotation = 45)
        ax.set_title(title)

        if not legend:
            plt.legend([],[], frameon=False)
        plt.legend(loc="upper left", mode = "expand", ncol = 3) 
        
        if show:
            plt.show()
        
        if save_to:
            plt.savefig(save_to)
            
        return None
