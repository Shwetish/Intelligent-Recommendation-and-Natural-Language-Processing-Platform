
sentence6="Mando talked for 3 hours although talking isn't his thing"

from nltk.stem.porter import PorterStemmer

ps_stemmer=PorterStemmer()

words=sentence6.split()

' '.join([ps_stemmer.stem(wd) for wd in words])
###########################################

sentence7="eating eats eat ate adjustable rafting ability meeting better"

from nltk.stem.porter import PorterStemmer

ps_stemmer=PorterStemmer()

words=sentence7.split()

' '.join([ps_stemmer.stem(wd) for wd in words])


##################################################################
import nltk
import pandas as pd
import spacy
from nltk.stem import PorterStemmer

# Initialize tools
ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

# Use ONE master list for both operations
lst_words = [
    "running",
    "painting",
    "walking",
    "dressing",
    "likely",
    "children",
    "whom",
    "good",
    "ate",
    "fishing",
]

# 1. Perform Stemming (NLTK)
stemmed_words = [ps.stem(word) for word in lst_words]

# 2. Perform Lemmatization (spaCy) safely word-by-word
# By processing individual words directly, we guarantee the list length stays exactly 10
lemmatized_words = [nlp(word)[0].lemma_ for word in lst_words]

# 3. Create a Comparison DataFrame (Now columns match perfectly!)
df = pd.DataFrame(
    {
        "Original Word": lst_words,
        "Stemmed (NLTK)": stemmed_words,
        "Lemmatized (spaCy)": lemmatized_words,
    }
)

print(df)
##############################################################################

