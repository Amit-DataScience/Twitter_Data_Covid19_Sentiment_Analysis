import  pandas as pd
from Get_Data import Extract_Twitter_Dat as gd
from constants import values as val
from Data_Preprocess import preprocess as pr
# Press the green button in the gutter to run the script.
from Sentiment_Analysis import sentiment_logic as sl
from Graphs import plots as pl
from  Modeling import model as mdl
import streamlit as st
if __name__ == '__main__':

    # title
    st.title('Tweet Sentiment Analysis')

    # markdown
    st.markdown('This application is all about twitter sentiment analysis of COVID-19 related tweets')

    # sidebar
    st.sidebar.title('Sentiment analysis of COVID-19 tweets')

    # loading the data
    df = pd.read_csv(r"covid19_tweets.csv")
    st.markdown('Check the Data set Information')
    # checkbox to show data
    if st.checkbox("Show Data"):
        st.write(df.head(10))
    if st.checkbox("Column Names"):
        st.write(df.columns)
    if st.checkbox("Length of data set"):
        st.write(len(df))
    if st.checkbox("Statistical insight of the data set"):
        st.write(df.describe())

    #Data Prepocessing
    df=pr.preprocess_dat(df)

    #print(df.head())

    #Sentiment Analysis
    df['sentiment_results'] = df['clean_tweet'].apply(sl.get_sentiment)
    df = df.join(pd.json_normalize(df['sentiment_results']))

    st.markdown('Check the tweets sentiments')
    if st.checkbox("Positive Sentiment"):
        st.write(df[df['sentiment']=='Positive'].head())
    if st.checkbox("Negative Sentiment"):
        st.write(df[df['sentiment']=='Negative'].head())
    if st.checkbox("Neutral Sentiment"):
        st.write(df[df['sentiment']=='Neutral'].head())

    #Ploting the grapgs:
    select = st.sidebar.selectbox('Visualisation Of Tweets', ['Count Tweets','Most_Common_Positive_Words','Most_Common_Negative_Words','Most_Common_Neutral_Words','Word_Cloud'])
    if select=='Count Tweets':
        pl.count_words(df)

    # Count of positive ,negative and neutral tweets

    if select == 'Most_Common_Positive_Words':
        pl.Most_Common_Positive_Words(df)

    elif select == 'Most_Common_Negative_Words':
        pl.Most_Common_Negative_Words(df)

    elif select == 'Most_Common_Neutral_Words':
        pl.Most_Common_Neutral_Words(df)

    elif select == 'Word_Cloud':
        page_names=['positive_word_cloud','negative_word_cloud','neutral_word_cloud']
        page=st.radio('Word_Cloud',page_names)
        if page=='positive_word_cloud':
            pl.pos_word_cloud(df)
        elif page == 'negative_word_cloud':
                pl.neg_word_cloud(df)
        else:
                pl.neut_word_cloud(df)



    #Data processing for modeling
    df['label']=mdl.label(df)

    # remove stop words from text messages
    df=mdl.rem_stop_words(df)

    #get the word stems using a Porter stemmer
    df=mdl.stemming(df)

    #model_Result
    prediction, accuracy, class_report = mdl.modeling(df)
    st.markdown('Model Results')
    if st.checkbox("Prediction"):
        st.write(prediction)

    if st.checkbox("Accuracy"):
        st.write(accuracy)

    if st.checkbox("Class_Report"):
        st.write(class_report)








