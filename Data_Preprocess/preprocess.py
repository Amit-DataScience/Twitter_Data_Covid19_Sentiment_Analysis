import neattext.functions as nfx
from textblob import TextBlob
import re
import string

def preprocess_dat(df):

    # Create one more column 'extracted_hashtags' and store all extracted hashtags
    df['extracted_hashtags'] = df['text'].apply(nfx.extract_hashtags)

    # Remove hashtags from  Text and store it in new column clean_tweet
    df['clean_tweet'] = df['text'].apply(nfx.remove_hashtags)

    #Remove userhandles
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: nfx.remove_userhandles(x))

    # Cleaning Text: Multiple WhiteSpaces
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_multiple_spaces)

    # Cleaning Text : Remove urls
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)

    # Cleaning Text: Punctuations
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_puncts)

    # remove special characters, numbers and punctuations
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

    df['clean_tweet']= df['clean_tweet'].str.lower()

    # remove short words
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

    return df