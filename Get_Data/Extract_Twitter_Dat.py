from tweepy import OAuthHandler
import tweepy as tw
import pandas as pd
from constants.values import *

def twitter_data():
    # Collect tweets
    tweets = tw.Cursor(API.search,
                       q=SEARCH_WORDS,
                       lang="en",
                       since=DATE_SINCE).items(10)

    tweets_copy = []
    for tweet in tweets:
        tweets_copy.append(tweet)

    tweets_df = pd.DataFrame()
    for tweet in tweets_copy:
        hashtags = []
        # try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
            text = API.get_status(id=tweet.id, tweet_mode='extended').full_text
        # except:
        #     pass
        tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name,
                                                   'user_location': tweet.user.location,
                                                   'user_description': tweet.user.description,
                                                   'user_created': tweet.user.created_at,
                                                   'user_followers': tweet.user.followers_count,
                                                   'user_friends': tweet.user.friends_count,
                                                   'user_favourites': tweet.user.favourites_count,
                                                   'user_verified': tweet.user.verified,
                                                   'date': tweet.created_at,
                                                   'text': text,
                                                   'hashtags': [hashtags if hashtags else None],
                                                   'source': tweet.source,
                                                   'is_retweet': tweet.retweeted}, index=[0]))
    print(tweets_df.head())

        # saving the dataframe as csv file
        #tweets_df.to_csv('twitter2_Covid_19.csv')
        #df = pd.read_csv(r"C:\Users\Admin\Desktop\Data_Science_Revision\Data_Science_Practical\NLP\covid19_tweets.csv")
    return tweets_df