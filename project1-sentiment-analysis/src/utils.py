import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

def review_to_words(review):
    """
    Convert a raw review to a sequence of words.
    """
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case, remove non-alphanumeric characters
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [stemmer.stem(w) for w in words] # stem
    
    return words

def convert_and_pad(word_dict, sentence, pad=500):
    """
    Convert a sentence into a vector of word indices and pad with zeros.
    """
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and 1 to represent the infrequent words, i.e., words not in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)
