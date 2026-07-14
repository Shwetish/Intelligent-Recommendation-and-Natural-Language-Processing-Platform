import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data packages (only need to run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Given text
text = """Latha is very multi talented girl.She is good at many skills like dancing, running, singing, playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this, she is a wonderful at cooking too.
"""

# Step 1: Tokenize the text into individual words
# We clean it slightly by converting to lowercase to ensure consistency
tokens = word_tokenize(text.lower())

# Step 2: Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Step 3: Apply Stemming and Lemmatization
stemmed_words = [stemmer.stem(word) for word in tokens]
lemmatizer_words = [lemmatizer.lemmatize(word) for word in tokens]

# Print the results in a clean, comparative format
print(f"{'Original Word':<15} | {'Stemmed (Porter)':<15} | {'Lemmatized (WordNet)':<15}")
print("-" * 53)

# Filter out punctuation for a cleaner display of the base forms
for orig, stem, lem in zip(tokens, stemmed_words, lemmatizer_words):
    if orig.isalnum():  # Displays only alphanumeric words
        print(f"{orig:<15} | {stem:<15} | {lem:<15}")
        
        
        
        
#################################################################
from collections import Counter
import spacy

# 1. Load the English language model
nlp = spacy.load("en_core_web_sm")

# 2. Read the text file
with open("C:/4-python_NLP/News_story.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 3. Process the text using spaCy
doc = nlp(text)

# --- Task 1: Extract all NOUN tokens ---
noun_list = [token.text for token in doc if token.pos_ == "NOUN"]

# --- Task 2: Extract all numbers (NUM) ---
number_list = [token.text for token in doc if token.pos_ == "NUM"]

# --- Task 3: Count and print all POS (Part-of-Speech) tags ---
pos_counts = Counter([token.pos_ for token in doc])

# Displaying the results
print("--- EXTRACTED NOUNS ---")
print(noun_list)
print("\n--- EXTRACTED NUMBERS ---")
print(number_list)

print("\n--- POS TAG COUNTS ---")
for pos, count in pos_counts.items():
    print(f"{pos}: {count}")
    
#######################################################################
import spacy

# 1. Load the English language model
nlp = spacy.load("en_core_web_sm")

# 2. Provide the input text
text = """Kiran want to know the famous foods in each 
state of India. So, he opened Google and search for this question. Google showed that 
in Delhi it is Chaat, in Gujarat it is Dal Dhokli, in Tamilnadu it is 
Pongal, in Andhrapradesh it is Biryani, in Assam it is Papaya Khar, 
in Bihar it is Litti Chowkha and so on for all other states"""

# 3. Process the text
doc = nlp(text)

# 4. Extract entities where the label is 'GPE'
geo_locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

# 5. Print the expected output
print("Geographical location Names:", geo_locations)
#######################################################################
import spacy

# 1. Load the English language model
nlp = spacy.load("en_core_web_sm")

# 2. Provide the input text
text = """Sachin Tendulkar was born on 24 April 1973, Virat Kholi was 
born on 5 November 1988, Dhoni was born on 7 July 1981 
and finally Ricky ponting was born on 19 December 1974."""

# 3. Process the text
doc = nlp(text)

# 4. Extract entities where the label is 'DATE'
birth_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

# 5. Print the expected output
print("All Birth Dates:", birth_dates)
#######################################################################
# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. Create a Dummy Movie Reviews Dataset (Replace this with your actual data load)
# -------------------------------------------------------------------------
data = {
    "review": [
        "I loved this movie! It was fantastic and full of energy.",
        "Worst movie ever. Total waste of time and money.",
        "The acting was amazing, a truly brilliant masterpiece.",
        "Boring plot, terrible directing, and awful acting.",
        "Highly recommended! An absolute joy to watch.",
        "I hated it. The story made no sense whatsoever.",
    ],
    "Category": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ],
}

df = pd.DataFrame(data)

# Convert text labels into numbers: positive -> 1, negative -> 0
df["label"] = df["Category"].map({"positive": 1, "negative": 0})


# -------------------------------------------------------------------------
# 2. Train-Test Split
# -------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["label"], test_size=0.33, random_state=42, stratify=df["label"]
)
# -------------------------------------------------------------------------
# 3. Create a Pipeline (Combines Bag of Words + Naive Bayes Model)
# -------------------------------------------------------------------------
# Using a Pipeline avoids doing CountVectorizer manually for train and test separately.
clf = Pipeline(
    [
        (
            "vectorizer",
            CountVectorizer(),
        ),  # Step 1: Convert text to Bag of Words matrix
        ("nb", MultinomialNB()),  # Step 2: Apply Multinomial Naive Bayes Classifier
    ]
)
# -------------------------------------------------------------------------
# 4. Train the Model
# -------------------------------------------------------------------------
clf.fit(X_train, y_train)


# -------------------------------------------------------------------------
# 5. Evaluate the Model
# -------------------------------------------------------------------------
y_pred = clf.predict(X_test)

# Print the classification report
print("--- MODEL EVALUATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
#####################################################################################
