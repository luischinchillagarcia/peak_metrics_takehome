import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")


class SentimentAnalysis:
    
    @staticmethod
    def sentiment_graph(brands_df, feature, resample_by, strftime='%Y-%m-%d'):
        timeseries_df = brands_df.set_index('publish_date')

        negative_sentiment = timeseries_df[f'{feature}_sentiment'] == 'negative'
        neutral_sentiment = timeseries_df[f'{feature}_sentiment'] == 'neutral'
        positive_sentiment = timeseries_df[f'{feature}_sentiment'] == 'positive'

        neg_trends = timeseries_df.loc[negative_sentiment, f'{feature}_sentiment'].resample(resample_by).count().reset_index()
        neu_trends = timeseries_df.loc[neutral_sentiment, f'{feature}_sentiment'].resample(resample_by).count().reset_index()
        pos_trends = timeseries_df.loc[positive_sentiment, f'{feature}_sentiment'].resample(resample_by).count().reset_index()

        neg_trends['label'] = 'negative'
        neu_trends['label'] = 'neutral'
        pos_trends['label'] = 'positive'

        all_labels_trends = pd.concat([neg_trends, neu_trends, pos_trends])

        all_labels_trends = all_labels_trends.set_index('publish_date').reset_index()
        all_labels_trends.publish_date = all_labels_trends.publish_date.dt.strftime(strftime)
        
        return all_labels_trends

    @staticmethod
    def plot(df, title, feature, show=False, save_to=None):
        _, ax = plt.subplots(figsize=(10,4))

        ax = sns.pointplot(x='publish_date', y=f'{feature}_sentiment', data=df, hue='label', palette={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})
        ax.tick_params(axis='x', labelrotation = 45)
        ax.set_title(title)
        
        if show:
            plt.show()
        
        if save_to:
            plt.savefig(save_to)
            
        return None
