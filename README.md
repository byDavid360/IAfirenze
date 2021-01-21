# Sentiment Classification using Decision Trees. 
Here I store the code and explanation on how to use the code I wrote of my assignment for the course INTELLIGENZA ARTIFICIALE of *Università degli Studi di Firenze*.

## What is inside this repository?
There's only one code file called **sentiment.py** where it is written all the code needed in order to reproduce the experiment.

## Which dataset do I need to use? Where can I find it?
The dataset needed for this experiment (and the one that I used) can be found on the following [link](http://www.cs.cornell.edu/people/pabo/movie-review-data/mix20_rand700_tokens_cleaned.zip)

Once you have downloaded the **zip** file you only have to extract it and there you will a folder called **tokens** inside which there'll be two folders:
- pos: the folder that contains the 700 txt positive reviews.
- neg: the folder that contains the 700 txt negative reviews.

## Which librarys do I need?
You will need the following python libraries in order to reproduce the results:
- **NLTK (Natural language toolkit)** : needed for operating with the text using its methods, functions and utilities (such as the stopwords).
- **Scikit-Learn** : needed for implementing our machine learning classifier
- **Regular expression library (re)**: needed for text lemmatization
- **Pandas**: used for returning the confussion matrix aesthetically

## How to use the code
Although the code has a detailed explanation, here I am going to remind which are the parts where the code needs to be edited in order to run the different scenarios
There are 4 scenarios:
1. Only using unigrams.
2. Only using bigrams.
3. Using unigrams and bigrams.
4. Using the top 2633 unigrams.

Those scenarios are achieved by changing some parameters frome the **CountVectorizer** function from **scikit-learn**

### Scenario 1 (unigrams)
If we want to use unigrams, we don't have to tell it to the CountVectorizer because it analyzes unigrams by default. 
In any case, the necessary parameter is **ngram_range=(1,1)**
NOTE: we are using the top 16165 unigrams for the analysis as seen in the parameter *max_features* (this can be edited of course)
```
vectorizer = CountVectorizer(max_features=16165, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1,1))

```

### Scenario 2 (bigrams)
In this case, it is necessary to edit the parameter *ngram_range* and set it to **ngram_range=(2,2)**
NOTE: we are using the top 16165 bigrams for the analysis as seen in the parameter *max_features* (this can be edited of course)
```
vectorizer = CountVectorizer(max_features=16165, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(2,2))

```

### Scenario 3 (unigrams + bigrams)
Here we need to change two parameters: *max_features* to **32330** (top 16165 unigrams + top 16165 bigrams) and *ngram_range* to **(1,2)** (unigrams + bigrams)
```
vectorizer = CountVectorizer(max_features=32330, min_df=5, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1,2))

```

### Scenario 4 (top 2633 unigrams)
This time we only need to change the parameter *max_features* to **2633**
NOTE: remeber that the parameter *ngram_range* is **(1,1)** (unigrams) by default so there is no need to write it.
```
vectorizer = CountVectorizer(max_features=2633, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

```

## Other sources
Although the majority of the code was written using the guides of the libraries, the word lemmatizer it's not mine since it was obtained from an amateur [website](https://stackabuse.com/text-classification-with-python-and-scikit-learn/)

```
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters.
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters (ej. David's without 's)
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
 ```
    


