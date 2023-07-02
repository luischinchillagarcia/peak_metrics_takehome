import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")


class BrandSentiment:
    
    @staticmethod
    def brand_filtering(blog_brands, brands, feature):
        brand_sentiment = blog_brands[[f'{feature}_sentiment', 'brands', 'publish_date']]
        brand_sentiment = brand_sentiment.rename(columns={f'{feature}_sentiment': 'sentiment'})

        brand_sentiment.brands = brand_sentiment.brands.map(lambda brand: brand[0])
        brand_filter = brand_sentiment.brands.isin(brands)
        brand_sentiment = brand_sentiment[brand_filter]
        return brand_sentiment
    
    @staticmethod
    def resample(brand_sentiment, by):
        brand_sentiment = brand_sentiment[['publish_date', 'brands', 'sentiment']].set_index('publish_date')
        brand_sentiment = brand_sentiment.groupby(['brands', 'sentiment']).resample(by).agg(count=('sentiment', 'count')).reset_index()
        
        brand_sentiment.publish_date = brand_sentiment.publish_date.dt.strftime('%Y-%m-%d')
        return brand_sentiment
    
    @staticmethod
    def plot(df, title, figsize=(10, 4), show=False, save_to=None):
        _, ax = plt.subplots(figsize=figsize)

        ax = sns.pointplot(x='publish_date', y='count', hue='sentiment', data=df, palette={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})
        ax.tick_params(axis='x', labelrotation = 45)
        ax.set_title(title)
        
        if show:
            plt.show()
        
        if save_to:
            plt.savefig(save_to)
            
        return None
