from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from utils.transfer_learning_models import NERModel, SentimentTask, EmotionTask


@dataclass
class DataCleaning:
    
    @staticmethod
    def read_data(path):
        data_dir = Path(path)
        df = pd.concat(pd.read_parquet(parquet) for parquet in data_dir.glob('*.parquet'))
        return df
    
    @staticmethod
    def base_filtering(df):
        df.publish_date = pd.to_datetime(df.publish_date, unit='s')
        
        # Extract English texts
        df_en = (df.language == 'en')
        df = df.loc[df_en]
        
        return df
        
    @staticmethod
    def add_ner_feature(df, feature, max_tokens):
        df = df[~df[feature].isna()]
        ner_feature = df[feature].map(lambda text: NERModel.model(text[:max_tokens]))
        df[f'{feature}_ner'] = ner_feature
        return df
        
    @staticmethod
    def add_sentiment_feature(df, feature, max_tokens):
        df = df[~df[feature].isna()]
        sentiment_feature = df[feature].map(lambda text: SentimentTask.model(text[:max_tokens]))
        sentiment_feature = sentiment_feature.map(lambda sentiment: sentiment[0]['label'])
        df[f'{feature}_sentiment'] = sentiment_feature
        return df
        
    @staticmethod
    def add_emotion_feature(df, feature, max_tokens):
        df = df[~df[feature].isna()]
        sentiment_feature = df[feature].map(lambda text: EmotionTask.model(text[:max_tokens]))
        df[f'{feature}_emotion'] = sentiment_feature
        return df
        
    @staticmethod
    def add_brands(df, feature, brands_to_filter):
        organizations = df[f'{feature}_ner'].map(lambda entities: [entity['word'] for entity in entities if entity['entity_group'] == 'ORG'])
        
        brands = organizations.map(lambda orgs: [org for org in orgs if org in brands_to_filter])
        df['brands'] = brands
        
        filter_no_brands = df['brands'].str.len() != 0
        df = df[filter_no_brands]
        return df
