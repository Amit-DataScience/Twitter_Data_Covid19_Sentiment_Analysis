import  seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import neattext.functions as nfx
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import plotly.express as px

# Get Most Commonest Keywords

def Positive_tokens(df):
    positive_tweet = df[df['sentiment'] == 'Positive']['clean_tweet']
    positive_tweet_list = positive_tweet.apply(nfx.remove_stopwords).tolist()
    pos_tokens = [token for line in positive_tweet_list for token in line.split()]
    return pos_tokens

def Negative_tokens(df):
    negative_tweet = df[df['sentiment'] == 'Negative']['clean_tweet']
    negative_tweet_list = negative_tweet.apply(nfx.remove_stopwords).tolist()
    neg_tokens = [token for line in negative_tweet_list for token in line.split()]
    return neg_tokens

def Neutral_tokens(df):
    neutral_tweet = df[df['sentiment'] == 'Neutral']['clean_tweet']
    neutral_tweet_list = neutral_tweet.apply(nfx.remove_stopwords).tolist()
    neut_tokens = [token for line in neutral_tweet_list for token in line.split()]
    return neut_tokens


def get_tokens(docx,num=30):
    word_tokens = Counter(docx)
    most_common = word_tokens.most_common(num)
    result = dict(most_common)
    return result

def count_words(df):
    # Plot with seaborn
    #print("Sentiment plot with seaborn\n")
    #sns.countplot(df['sentiment'])
    #plt.show()
    fig = plt.figure()
    sns.countplot(df['sentiment'])
    st.pyplot(fig)

    # Plot the Bar graph:
def Most_Common_Positive_Words(df):
    pos_tokens=Positive_tokens(df)
    most_common_pos_words = get_tokens(pos_tokens)
    pos_df = pd.DataFrame(most_common_pos_words.items(), columns=['words', 'scores'])
    fig = px.bar(pos_df, x='words', y='scores')
    st.plotly_chart(fig)

def Most_Common_Negative_Words(df):
    neg_tokens =Negative_tokens(df)
    most_common_neg_words = get_tokens(neg_tokens)
    neg_df = pd.DataFrame(most_common_neg_words.items(), columns=['words', 'scores'])
    #print("Most Common Negative words\n",neg_df.head(),"\n")
    #plt.figure(figsize=(20, 10))
    fig=px.bar(neg_df,x='words', y='scores')
    st.plotly_chart(fig)

def Most_Common_Neutral_Words(df):
    neut_tokens =Neutral_tokens(df)
    most_common_neut_words = get_tokens(neut_tokens)
    neu_df = pd.DataFrame(most_common_neut_words.items(), columns=['words', 'scores'])
    fig = px.bar(neu_df, x='words', y='scores')
    st.plotly_chart(fig)

#Word Cloud
def plot_wordcloud(docx):
    mywordcloud = Word Cloud().generate(docx)
    plt.imshow(mywordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

    # Plot WordCloud Image
def pos_word_cloud(df):
    pos_docx = ' '.join(Positive_tokens(df))
    plot_wordcloud(pos_docx)

def neg_word_cloud(df):
    neg_docx = ' '.join(Negative_tokens(df))
    plot_wordcloud(neg_docx)

def neut_word_cloud(df):
    neut_docx = ' '.join(Neutral_tokens(df))
    plot_wordcloud(neut_docx)


