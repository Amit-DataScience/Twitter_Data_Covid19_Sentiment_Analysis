import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,fbeta_score,classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

# convert class labels to binary values, 0 = ham and 1 = spam
def label(df):
    encoder = LabelEncoder()
    y= encoder.fit_transform(df["sentiment"])
    return y

def rem_stop_words(df):
    stop_words = set(stopwords.words('english'))

    df['clean_tweet']=df['clean_tweet'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    return df

def stemming(df):
    # get the word stems using a Porter stemmer
    ps = nltk.PorterStemmer()

    df['clean_tweet']=df['clean_tweet'].apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
    return df

def modeling(df):
    X = df["clean_tweet"]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])

    logreg.fit(X_train, y_train)

    y_pred=logreg.predict(X_test)

    accu_score=accuracy_score(y_pred, y_test)

    class_report=classification_report(y_test, y_pred)

    return(y_pred,accu_score,class_report)
