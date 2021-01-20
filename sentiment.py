
#we import the regular expression library
import re 
#we import the Natural Language Toolkit
import nltk 
#we import the function load_files of sklearn to read our dataset
from sklearn.datasets import load_files
#we import the stopwords from nltk
from nltk.corpus import stopwords

#we load the dataset in the variable "data"
data = load_files(r"D:\Escritorio\ASIGNATURAS ERASMUS\IA\Proyecto\tokens")

"""The variable X is an array that stores the pos and neg reviews while the variable y is an array that stores the values 0 or 1 if the current review is neg or pos"""
X,y = data.data, data.target

documents = []

#we import the word lemmatizer from nltk
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    # here we can see the use of the "re" library
    # Remove all the special characters. Numeros, caracteres especiales...
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters (ej. David's se quita la s)
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)


from sklearn.feature_extraction.text import CountVectorizer


""" We create a bag of words
    max_features -> it is the number of most frequent words we want to use. In this case we are using 16165 unigrams
    min_df -> it is the minimum number of documents that should have the feature. We include the words present at least in that number of documents (5 in this case)
    max_df -> we include those words that appear in a maximun of this percentage of documents (70% in this case)
    We do it that way because words that appear in 100% of the documents aren't usually a good option for the classifier because they don't give special info about the document (articles for example)
    stopwords -> we delete the stopwords 
    ngram_range -> here we specify if we want to use unigrams, bigrams,... ngrams
"""

vectorizer = CountVectorizer(max_features=16165, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1,1))
#the function fit_transform() converts texts into numeric features
X = vectorizer.fit_transform(documents).toarray()

#TFIDF (Term Frequency Inverse Document Frequency)

#TFIDF tells us how relevant a word in a document is
# TFIDF gets higher values if the word appears more times in the document but at the same time it is compensated with the frequency of appearance it has in the document collection
# this allows us to avoid errors such as considering articles (words that appear a lot of times in a lot of documents) important words
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split


#We divide the dataset into trainingset and testset
""" INPUT PARAMETERS 
    test_size = 0.2 -> 20% of our data is going to be assigned to the testing set
    random_state = 0 -> everytime the code is run, our splitted dataset is composed of different reviews
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#We import the decision tree classifier function from sklearn
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
#with this method train the classifier
classifier.fit(X_train, y_train)
#we make the predictions with the metod predict()
y_pred = classifier.predict(X_test)

#finally we return the confussion matrix and the classification report
from sklearn.metrics import classification_report, confusion_matrix
print("CONFUSSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("")
print("MAIN CLASSIFICATION METRICS")
print(classification_report(y_test, y_pred))

#import pandas for an aesthetic view of the confussion matrix
import pandas as pd
pd.DataFrame(confusion_matrix(y_test, y_pred),
    columns=['Predicted Negative', 'Predicted Positive'],
    index=['True Negative', 'True Positive'])