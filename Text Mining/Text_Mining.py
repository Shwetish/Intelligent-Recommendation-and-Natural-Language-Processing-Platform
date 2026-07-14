#====================================================
#Business Understanding:-
#====================================================
#1. Business Problem Statement:-
#Companies sell products online but don’t know why customers are happy 
#or unhappy. Manual reading of thousands of online reviews on websites 
#like Amazon or Flipkart is impossible, causing companies to miss out
# on critical customer feedback.

#2. Business Understanding:-
#Customers openly share what they love or hate about a product in the 
#review section. If a business can automatically scan, sort, and
# understand these emotional responses at a massive scale, they can
# quickly fix product defects, upgrade popular features, and keep their
# customers loyal.

#3. Motivation:-
#Bigger Profits: Happy customers buy more and recommend products to 
#others.
#Stay Ahead of Competitors: Fixing a product flaw faster than a rival 
#company wins the market.
#Smart Decisions: Instead of guessing what customers want, the company
# uses real customer data to guide its next steps.

#4. Constraints:-
#Sarcasm & Slang: Computers struggle to understand internet slang or 
#sarcasm (e.g., "Wow, it broke in 2 days. Simply amazing!").
#Fake Reviews: Paid or bot-generated reviews can skew the real data.
#Language & Typos: Reviews often contain bad grammar, spelling mistakes,
# or a mix of languages (like Hinglish).

#5. Business Success Criteria:-
#Higher Customer Satisfaction: A measurable drop in negative reviews 
#and an increase in star ratings over time.
#Product Improvement: Successfully identifying and fixing at least 2 or
# 3 major product complaints within a quarter.
#Increased Revenue: A visible boost in sales or a reduction in product
# returns/refund requests.

#6. Machine Learning (ML) Success Criteria:-
#High Accuracy: The sentiment analysis model correctly categorizes
# reviews as Positive, Negative, or Neutral at least 80% to 85% of the
# time.
#Speed: The system can process thousands of daily incoming reviews 
#automatically in real-time or overnight batches without breaking.
#Would you like me to help you draft a specific data collection or 
#preprocessing strategy based on these constraints?

#====================================================================
#Data Understanding:-
#===================================================================
'''
Name of Feature     Description             Type               Relevance
Products            Products names       Quanlitative,Nominal       High
Price              Actual price         Quantitative,continuous    High
                   of a product
Delivery Date     Expected delivery     Quantitative,Discrete      High
                  date of any product
                    
'''

#====================================================================
# EDA - E-COMMERCE REVIEWS & SENTIMENT DATASET
#====================================================================
#Task 1:
#1.	Extract reviews of any product from e-commerce website Amazon.
#=========================================================
# Import Libraries
#=========================================================
import pandas as pd

df = pd.read_csv(
    'C:/13_NLP_Basics/ecommerce_reviews.csv',
    sep='\t'
)

print(df.columns)
print(df.head())


#=========================================================
# Summary
#=========================================================

print(df.info())

print("\nShape of Dataset")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nSummary Statistics")
print(df.describe(include='all'))
'''
Inference
Dataset loaded successfully.
No missing values are present.
Dataset contains product details and customer reviews.
Both numerical and text columns are available.
'''
#-------------------------------------------------------------------
#Univariate Analysis
#----------------------------------------------------------------

plt.figure(figsize=(5,4))
sns.countplot(x='Review', data=df)
plt.title("Positive vs Negative Reviews")
plt.show()
'''
Inference
Positive reviews are more than negative reviews.
Most customers are satisfied with the products.
'''
#-----------------------------------------------------------------------
#Product Price Distribution
#-----------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(df['Price'], bins=5)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()
'''
Inference
Most products have low to medium prices.
Only a few products are expensive.
'''
#-------------------------------------------------------------------------
#Ratings Distribution
#--------------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Rating', data=df)
plt.title("Ratings")
plt.xticks(rotation=45)
plt.show()
'''
Inference
Most products have ratings above 4 stars.
Overall customer ratings are good.
'''
#------------------------------------------------------------
#Bivariate Analysis
#--------------------------------------------------------------
#Review vs Rating
plt.figure(figsize=(6,4))
sns.countplot(x='Rating', hue='Review', data=df)
plt.title("Rating vs Review")
plt.xticks(rotation=45)
plt.show()
'''
Inference
Positive reviews mostly have higher ratings.
Negative reviews generally have lower ratings.
'''
#Product vs Review
plt.figure(figsize=(8,4))
sns.countplot(x='Products', hue='Review', data=df)
plt.xticks(rotation=45)
plt.title("Product vs Review")
plt.show()
'''
Inference
Every product has customer feedback.
Positive reviews are higher than negative reviews.
'''
#===========================================================
#Model Building
#===========================================================
#=========================================================
# Extract Review Titles
#=========================================================

reviews = df['Review Title']

print(reviews)
'''
Inference
Review titles are extracted successfully.
These reviews will be used for text analysis.
'''
#Data Cleaning
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]','',text)
    return text

df['Clean_Review'] = df['Review Title'].apply(clean_text)

print(df[['Review Title','Clean_Review']])
'''
Inference
Text converted into lowercase.
Numbers and special characters removed.
Clean text is ready for analysis.
Positive Word Cloud
'''
from wordcloud import WordCloud

positive = " ".join(df[df['Review']=="positive"]['Clean_Review'])

wordcloud = WordCloud(background_color='white').generate(positive)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive Word Cloud")
plt.show()
'''
Inference
Frequently used positive words are displayed.
It shows what customers like most.
Negative Word Cloud
'''
negative = " ".join(df[df['Review'].str.lower()=="negative"]['Clean_Review'])

wordcloud = WordCloud(background_color='white').generate(negative)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("Negative Word Cloud")
plt.show()
'''
Inference
Negative words are highlighted.
It helps identify customer complaints.
Sentiment Analysis
'''
from textblob import TextBlob

def sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Clean_Review'].apply(sentiment)

print(df[['Review Title','Sentiment']])
#Count of Sentiments
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Analysis")
plt.show()
'''
Inference
Most reviews are classified as positive.
Only a few reviews are negative.
Customers are generally satisfied.
'''

6. Benefits / Impact of the Solution
Business Benefits
Helps understand customer opinions quickly.
Identifies product strengths and weaknesses.
Improves product quality using customer feedback.
Increases customer satisfaction.
Helps companies make better business decisions.
Saves time by automatically analyzing thousands of reviews.
Improves sales and customer trust.

#=======================================================================================================================
#Task2.	Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud.
#==========================================================================================================================#

import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud

# --------------------------------------------------------------------
# STEP 1: DEFINE TARGET URL & ANTI-BLOCKING HEADERS
# --------------------------------------------------------------------
link = "https://www.amazon.in/Redmi-Pad-Wi-Fi-Cellular-AI-Enabled/dp/B0FBRVRF4F/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

print("Fetching data from Amazon...")
page = requests.get(link, headers=headers)
soup = bs(page.content, "html.parser")

# --------------------------------------------------------------------
# STEP 2: SCRAPE REVIEWS WITH FALLBACK DATA
# --------------------------------------------------------------------
titles = soup.find_all("a", {"data-hook": "review-title"})
review_titles = [
    t.get_text().strip().split("\n")[-1].strip() for t in titles
]

ratings = soup.find_all("i", {"data-hook": "review-star-rating"})
rate = [r.get_text().replace("out of 5 stars", "").strip() for r in ratings]

reviews = soup.find_all("span", {"data-hook": "review-body"})
review_body = [re.get_text().strip() for re in reviews]

min_length = min(len(review_titles), len(rate), len(review_body))

# FALLBACK SAFETY net: If Amazon blocks us, generate dummy data so the NLP code still runs smoothly
if min_length == 0:
    print(
        "\n[WARNING] Amazon blocked the live request or no reviews found. Using synthetic fallback data for demonstration."
    )
    review_titles = [
        "Great tablet",
        "Disappointed",
        "Value for money",
        "Average performance",
        "Excellent screen",
    ]
    rate = ["5", "2", "4", "3", "5"]
    review_body = [
        "The battery life is amazing and the display is crystal clear. Highly recommend this Redmi pad!",
        "It hangs a lot during gaming. Not happy with the performance and processor speed.",
        "Good value for money product. Nice battery backup and decent sound quality.",
        "The build quality is average. Sound is good but camera is terrible.",
        "Excellent screen crisp colors and smooth 90Hz refresh rate. Best budget tablet.",
    ]
    min_length = len(review_body)
else:
    review_titles = review_titles[:min_length]
    rate = rate[:min_length]
    review_body = review_body[:min_length]
    print(f"\nExtracted {min_length} matching live reviews successfully.")

# Create Directory if it doesn't exist
output_dir = "C:/13_NLP_Basics/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Construct and Save Dataframe
df = pd.DataFrame(
    {"Review_Title": review_titles, "Rate": rate, "Review": review_body}
)
csv_path = os.path.join(output_dir, "amazon_reviews.csv")
df.to_csv(csv_path, index=False)

# --------------------------------------------------------------------
# STEP 3: COMPUTE SENTIMENT POLARITY METRICS
# --------------------------------------------------------------------
df_saved = pd.read_csv(csv_path)
df_saved["polarity"] = df_saved["Review"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

print("\n--- Processed Dataset Sample Preview ---")
print(df_saved[["Review_Title", "Rate", "polarity"]].head())

# --------------------------------------------------------------------
# STEP 4: TEXT CLEANING FOR WORD CLOUDS
# --------------------------------------------------------------------
# Combine all reviews into one massive block of text
all_text = " ".join(df_saved["Review"].astype(str)).lower()

# Basic cleaning (removing punctuation and standard irrelevant words)
stopwords = {
    "the",
    "and",
    "a",
    "to",
    "is",
    "it",
    "of",
    "for",
    "in",
    "with",
    "this",
    "on",
    "i",
    "this",
    "that",
    "but",
    "my",
    "not",
    "was",
    "amazon",
    "product",
    "tablet",
    "pad",
}
words = [w for w in all_text.split() if w.isalpha() and w not in stopwords]

# --------------------------------------------------------------------
# STEP 5: GENERATE & PLOT UNIGRAM WORD CLOUD
# --------------------------------------------------------------------
unigram_text = " ".join(words)
unigram_wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate(unigram_text)

plt.figure(figsize=(10, 5))
plt.imshow(unigram_wordcloud, interpolation="bilinear")
plt.title("Unigram Word Cloud (Single Words)")
plt.axis("off")
plt.show()

# --------------------------------------------------------------------
# STEP 6: GENERATE & PLOT BIGRAM WORD CLOUD
# --------------------------------------------------------------------
# Generate pairs of connected words (Bigrams)
bigram_tuples = list(ngrams(words, 2))
# Convert [('battery', 'life')] back into string format 'battery_life' so WordCloud can parse it
bigram_strings = ["_".join(bg) for bg in bigram_tuples]
bigram_text = " ".join(bigram_strings)

# Generate and plot bigram wordcloud
bigram_wordcloud = WordCloud(
    width=800, height=400, background_color="black", colormap="jet"
).generate(bigram_text)

plt.figure(figsize=(10, 5))
plt.imshow(bigram_wordcloud, interpolation="bilinear")
plt.title("Bigram Word Cloud (Two-Word Phrases)")
plt.axis("off")
plt.show()

#==========================================================================================================
#Benifits and Impact of the Solution
#==========================================================================================================
Key Benefits (What the system does)
Saves Time & Effort: It automatically reads thousands of online customer reviews in
seconds, replacing days of exhausting manual reading.

Finds Hidden Problems: It quickly spots product defects (like a weak battery or screen
freezing) before they ruin a products reputation.

Tracks Real Feelings: It translates customer text into clear sentiment scores
(Positive, Negative, or Neutral) so you always know exactly how happy your buyers are.

Combines All Data: It blends numbers (prices, delivery times) with customer text to
give a complete, single picture of business performance.
#=============================================================================================================================
#Business Impact (How it helps the company grow)
#============================================================================================================================
Higher Profits: Happy customers buy more frequently, stay loyal to the brand, and 
recommend the products to their friends.

Beating Competitors: By identifying and fixing product flaws faster than rival 
companies, you capture a larger share of the market.

No More Guesswork: You make smart, data-driven decisions about future product features based on real customer words, instead
 of just guessing.

Fewer Returns: Finding and resolving customer complaints quickly results in a noticeable drop in product refunds and return
 requests.
#===========================================================================================================================
#	Extract reviews for any movie from IMDB and perform sentiment analysis.
#===========================================================================================================================
import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


def extract_imdb_reviews(url):
    # Set headers to mimic a real browser request, otherwise IMDb may block it
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    print("Fetching reviews from IMDb...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

   # 1. Define the function first
def get_my_reviews(soup):
    review_containers = soup.find_all("div", class_="review-container")
    reviews_data = []

    for container in review_containers:
        # Extract Title
        title_element = container.find("a", class_="title")
        title = title_element.text.strip() if title_element else ""

        # Extract Review Text
        text_element = container.find("div", class_="text show-more-__control")
        if not text_element:
            text_element = container.find("div", class_="text")
        text = text_element.text.strip() if text_element else ""

        # Extract Rating
        rating_element = container.find(
            "span", class_="rating-other-user-rating"
        )
        rating = (
            rating_element.text.replace("\n", "").strip()
            if rating_element
            else "No Rating"
        )

        if text:
            reviews_data.append(
                {"Title": title, "Review": text, "IMDb_Rating": rating}
            )

    # Correctly aligned outside the loop
    return reviews_data


# 2. Call the function later in your cell
# results = get_my_reviews(soup)


def analyze_sentiment(text):
    """Analyzes text polarity using TextBlob and categorizes it."""
    analysis = TextBlob(text)
    # Polarity ranges from -1 (very negative) to +1 (very positive)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # The URL you provided for The Lion King (1994) reviews
    target_url = "https://www.imdb.com/title/tt0110357/reviews/?ref_=tt_ov_ql_2"

    # Step 1: Extract reviews
    extracted_reviews = extract_imdb_reviews(target_url)

    if extracted_reviews:
        # Step 2: Convert to a Pandas DataFrame for structured processing
        df = pd.DataFrame(extracted_reviews)

        # Step 3: Apply Sentiment Analysis
        df["Sentiment"], df["Polarity_Score"] = zip(
            *df["Review"].apply(analyze_sentiment)
        )

        # Step 4: Display the Results
        print(f"\nSuccessfully extracted and analyzed {len(df)} reviews!")
        print("\n--- Sample Results ---")
        print(df[["Title", "Sentiment", "Polarity_Score", "IMDb_Rating"]].head())

        # Optional: Save results to a CSV file
        df.to_csv("lion_king_sentiment_analysis.csv", index=False)
        print("\nResults saved to 'lion_king_sentiment_analysis.csv'")
    else:
        print("No reviews were found. Check your URL or internet connection.")
#===========================================================================================================================
#Choose any other website on the internet and do some research on how to extract text and perform sentiment analysis
#===========================================================================================================================
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from textblob import TextBlob

# 1. Target Snapdeal URL provided
link = (
    "https://www.snapdeal.com/product/YOUR-NEW-POPULAR-PRODUCT-URL"
)

# Standard browser headers to prevent basic bot blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Fetching Snapdeal webpage content...")
page = requests.get(link, headers=headers)
soup = bs(page.content, "html.parser")

# Setup empty lists to collect structured data
review_titles = []
ratings = []
review_bodies = []

# 2. Target the Snapdeal review card container elements
# Snapdeal organizes individual user review layouts within rows containing 'commentlist' or 'user-review' classes
review_cards = soup.find_all("div", class_="commentlist")

# Fallback container check if the layout structure shifts depending on device strings
if not review_cards:
    review_cards = soup.find_all("div", class_="user-review")

print(f"Found {len(review_cards)} review element blocks.")

# 3. Safely scrape data using matched card containers to keep arrays aligned
for card in review_cards:
    # Extract Title
    title_el = card.find("div", class_="head") or card.find(
        "p", class_="review-title"
    )
    # Extract Star Rating (Snapdeal often counts active 'sd-icon-star' or 'active' span indicators)
    # Alternatively, they list numerical values inside rating classes
    rating_el = card.find("div", class_="rating") or card.find(
        "span", class_="rating-stars"
    )

    # Extract Actual Review Comment Text
    body_el = card.find("p", class_="") or card.find(
        "div", class_="user-review-text"
    )

    if body_el:
        # If title element isn't found, default it cleanly
        title_text = (
            title_el.get_text(strip=True) if title_el else "Customer Review"
        )

        # Handle Rating extraction logic cleanly
        rating_text = "No Rating"
        if rating_el:
            # Snapdeal frequently uses a string or counting class for rating stars
            rating_text = rating_el.get_text(strip=True)
            if not rating_text:  # If text is blank, inspect internal attributes
                rating_text = "Rated"

        body_text = body_el.get_text(strip=True)

        review_titles.append(title_text)
        ratings.append(rating_text)
        review_bodies.append(body_text)

# 4. Fallback Block: If Snapdeal page hasn't received public customer text reviews yet
if len(review_bodies) == 0:
    print(
        "\n[Notice] No raw text reviews found in the HTML tags for this specific item yet."
    )
    print("Populating testing mockup data based on typical tripod reviews...\n")
    review_titles = [
        "Great value product",
        "Terrible build",
        "Decent bluetooth connection",
    ]
    ratings = ["5 Stars", "1 Star", "4 Stars"]
    review_bodies = [
        "This sterling bazaar bluetooth tripod is lightweight and highly adjustable. Perfect for taking phone videos!",
        "Very poor material quality. One of the plastic clips broke during my first setup attempt. Waste of money.",
        "The remote shutter connects quickly to my android device. Holds the phone safely enough for basic shots.",
    ]

# 5. Compile into Pandas DataFrame
df = pd.DataFrame()
df["Review_Title"] = review_titles
df["Rate"] = ratings
df["Review"] = review_bodies

# 6. Apply your exact lambda logic for Sentiment Polarity analysis
df["Polarity"] = df["Review"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)


# Turn numerical scores into human-readable sentiment tags
def determine_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"


df["Sentiment"] = df["Polarity"].apply(determine_sentiment)

# 7. Output result as a CSV file
df.to_csv("snapdeal_tripod_reviews.csv", index=True)

print("--- Snapdeal Sentiment Analysis Dataset Summary ---")
print(df[["Review_Title", "Rate", "Polarity", "Sentiment"]])
