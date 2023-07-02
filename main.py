from dataclasses import dataclass
import argparse

from utils.data_cleaning import DataCleaning
from utils.sentiment_analysis import SentimentAnalysis
from utils.topics import Topics
from utils.emotion_analysis import EmotionsAnalysis
from utils.brand_sentiment import BrandSentiment


@dataclass
class BaseConfig:
    MAX_TOKENS = 512
    BRAND = 'United Airlines'
    COMPETITOR_BRANDS = ['Southwest Airlines', 'Spirit Airlines', 'American Airlines', 'United', 'Alaska Airlines', 'JetBlue']
    IMPLICIT_IDENTIFIERS = ['Airline', 'Airlines', 'General Aviation', 'FAA', 'Boeing']
    SAVE_TO = './plots'


@dataclass
class BlogConfig(BaseConfig):
    TITLE = 'title'
    BODY = 'body'
    PATH = './data-science/blog'


@dataclass
class NewsConfig(BaseConfig):
    TITLE = 'headline'
    BODY = 'body'
    PATH = './data-science/news'


@dataclass
class SocialConfig(BaseConfig):
    TITLE = 'title'
    BODY = 'body'
    PATH = './data-science/social'


def data_cleaning(config):
    data_df = DataCleaning.read_data(config.PATH)
    data_df = DataCleaning.base_filtering(df=data_df)
    
    data_df = DataCleaning.add_ner_feature(df=data_df, feature=config.TITLE, max_tokens=config.MAX_TOKENS)
    data_df = DataCleaning.add_ner_feature(df=data_df, feature=config.BODY, max_tokens=config.MAX_TOKENS)
    
    data_df = DataCleaning.add_sentiment_feature(df=data_df, feature=config.TITLE, max_tokens=config.MAX_TOKENS)
    data_df = DataCleaning.add_sentiment_feature(df=data_df, feature=config.BODY, max_tokens=config.MAX_TOKENS)
    
    data_df = DataCleaning.add_brands(df=data_df, feature=config.TITLE, brands_to_filter=([config.BRAND] + config.COMPETITOR_BRANDS + config.IMPLICIT_IDENTIFIERS))
    return data_df


def sentiment_analysis(config, data_df):
    title_sentiment_ts = SentimentAnalysis.sentiment_graph(brands_df=data_df, feature=config.TITLE, resample_by='H', strftime='%Y-%m-%d')
    SentimentAnalysis.plot(title_sentiment_ts, title='Title Sentiment Analysis Over Time', feature=config.TITLE, save_to='title_sentiment_over_hourly')

    title_sentiment_ts = SentimentAnalysis.sentiment_graph(brands_df=data_df, feature=config.TITLE, resample_by='D', strftime='%Y-%m-%d')
    SentimentAnalysis.plot(title_sentiment_ts, title='Title Sentiment Analysis Over Time', feature=config.TITLE, save_to='title_sentiment_over_hourly')

    title_sentiment_ts = SentimentAnalysis.sentiment_graph(brands_df=data_df, feature=config.BODY, resample_by='D', strftime='%Y-%m-%d')
    SentimentAnalysis.plot(title_sentiment_ts, title='Body Sentiment Analysis Over Time', feature=config.BODY, save_to='body_sentiment_over_daily')
    
    return None


def data_topics(config, data_df):
    negative_body_sentiments_df = Topics.filter_sentiment(df=data_df, feature=config.BODY, sentiment='negative')
    negative_body_topics = Topics.all_topics(df=negative_body_sentiments_df, feature=config.BODY)
    top_negative_body_topics = Topics.top_topics(topics=negative_body_topics, filter_by_sentiment='negative')

    Topics.plot(df=top_negative_body_topics, title='Most Common Topics in Body Negative Sentiment', top_k=20, save_to='topics_body_negative')
    
    return None


def emotion_analysis(config, data_df):
    negative_df = EmotionsAnalysis.get_sentiment(df=data_df, sentiment='negative', feature=config.BODY)
    negative_df = EmotionsAnalysis.add_lg_emotions(df=negative_df, feature=config.TITLE, max_tokens=config.MAX_TOKENS)
    negative_df = EmotionsAnalysis.resample(df=negative_df, by='D', strftime='%Y-%m-%d')
    negative_df = EmotionsAnalysis.filter_negative_emotions(df=negative_df)

    EmotionsAnalysis.plot(df=negative_df, title='Title Emotions Over Time', save_to='negative_emotions_daily')
    
    return None


def brand_sentiment(config, data_df):
    brand_sentiment = BrandSentiment.brand_filtering(data_df, brands=config.COMPETITOR_BRANDS + [config.BRAND], feature=config.BODY)
    brand_sentiment = BrandSentiment.resample(brand_sentiment, by='D')
    BrandSentiment.plot(brand_sentiment, title='All Brands Body Sentiment Over Time')

    brand_sentiment = BrandSentiment.brand_filtering(data_df, brands=['United'], feature=config.BODY)
    brand_sentiment = BrandSentiment.resample(brand_sentiment, by='D')
    BrandSentiment.plot(brand_sentiment, title='Brand Body Sentiment Over Time')
    
    return None


def main(config):
    data_df = data_cleaning(config)
    data_df = sentiment_analysis(config, data_df)
    data_df = data_topics(config, data_df)
    data_df = emotion_analysis(config, data_df)
    data_df = brand_sentiment(config, data_df)
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Choose --data-name between blog, news, social",
    )
    parser.add_argument('-d', '--data-name', choices=['blog', 'news', 'social'], default='blog')
    args = parser.parse_args()
    
    config = {
        'blog': BlogConfig,
        'news': NewsConfig,
        'social': SocialConfig,
    }

    main(config=config[args.data_name])
