#-----------------------------------------------------------------
#Business Understanding
#-----------------------------------------------------------------
#1. Business Problem Statement:-
#An online movie streaming platform has user rating data (from -9 
#to +9) for various movies. They want to recommend highly rated 
#movies to customers based on their watching history, but 
#currently lack an automated system. Without this, they may lose 
#customers due to irrelevant suggestions.

#2. Business Objective:-
#Improve movie collection by highlighting highly rated movies.
#Build an automated recommendation system to suggest relevant
# movies to each user.
#Increase customer engagement and watch time.

#3. Motivation
#Customers stay longer on platforms that suggest movies they like.
#Personalized recommendations reduce search time and improve 
#satisfaction.
#Better recommendations lead to higher subscription retention 
#and revenue.

#4. Constraints
#Ratings range from -9 to +9 (non-standard scale, needs careful
# handling).
#Some movies have "99" in Reviews column (likely missing/error 
#value).
#No explicit user IDs shown implicit feedback or assume
# aggregated data.
#Cold start for new movies or new users.
#Genre mixing (multiple genres per movie) makes similarity 
#complex.

#5. Success Criteria
#5.1 Business Success Criteria
#Increase in average watch time per user by at least 15% in 3 
#months.
#Higher user retention rate (less churn).
#Positive customer feedback on "recommended for you" section.
#5.2 ML Success Criteria
#Achieve precision@10 > 0.6 (60% of top-10 recommendations are 
#liked by user).
#Recall@10 > 0.4 (capture 40% of movies user actually likes).
#RMSE on predicted vs actual ratings < 2.0 (on -9 to +9 scale).
#Recommendation generation time < 2 seconds for real-time use.

#-----------------------------------------------------------------------------
#Data Understanding
#-----------------------------------------------------------------------------
'''
Name of Feature       Description                 Type        Relevance
ID                  Unique identification   Numeric,discrete    High
                    number for each movie
Title                Name of te Movie       Qualitative,Nominal High
                    With release year
Category            tags for movie           Categorical        High
Review               Numerical Score         Continuous         Medium
'''

#=================================================================================================
# Exploratory Data Analysis (EDA) - Entertainment Dataset
#=================================================================================================

import pandas as pd
import numpy as np

# Load the dataset (Assuming the sheet is saved as a CSV or Excel file)
# df = pd.read_excel('Entertainment_Data.xlsx', sheet_name='Entertainment')

# Simulating data structure for demonstration
data = {
    'Id': [6973, 6778, 9702, 6769, 1123, 9860],
    'Titles': ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)', 'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)', 'Heat (1995)'],
    'Category': ['Drama, Romance, School, Supernatural', 'Action, Adventure, Drama, Fantasy, Magic, Military, Shounen', 'Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen', 'Sci-Fi, Thriller', 'Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen', 'Comedy, Drama, School, Shounen, Sports'],
    'Reviews': [-8.98, 8.88, 99.00, 99.00, -0.44, -6.65]
}
df = pd.DataFrame(data)

# 1. Replace placeholder '99' with NaN
df['Reviews'] = df['Reviews'].replace(99.00, np.nan)

# 2. Check how many missing reviews we have now
print("Missing reviews count:\n", df['Reviews'].isna().sum())

# 3. Clean version for analysis (dropping rows where no user has rated it yet)
df_clean = df.dropna(subset=['Reviews']).copy()
print("\nCleaned Data Sample:")
print(df_clean)
'''
Inference: Movies like "Grumpier Old Men" and "Waiting to Exhale"
 had a review score of 99. Because the company stated that valid ratings are strictly
 between $-9$ and $+9$, the number 99 represents an unrated movie. We successfully 
 identified and removed these outliers so they don't skew our analytics.
  Exploratory Data Analysis (Finding Top-Rated Movies)The company wants to 
 showcase highly rated movies to its users. We will sort the cleaned dataset to find 
 the highest-rated titles.
'''
 
# Sort the movies based on Reviews in descending order
top_rated_movies = df_clean.sort_values(by='Reviews', ascending=False)
print("Top Highly Rated Movies to Showcase:")
print(top_rated_movies[['Titles', 'Category', 'Reviews']])
'''
Inference: This code ranks your movies from best to worst based on
 actual user sentiment. For example, Jumanji (1995) has a positive rating of 8.88, 
 making it an excellent candidate to display on the platform's "Trending/Top Rated" 
 homepage banners. Conversely, movies with deep negative scores (like Toy Story at 
-8.98 in this specific user dataset) will be hidden from front-page promotions.
Step 3: Content-Based Recommendation AlgorithmTo recommend movies based on a 
customer's "movie watching footprint," we can build a Content-Based Filtering system.
 If a user watches an Action/Sci-Fi movie, the algorithm will recommend other movies
 with identical or highly similar genres.We achieve this by breaking down the 
 comma-separated text in the Category column using a technique called TF-IDF (Term 
Frequency-Inverse Document Frequency) and calculating Cosine Similarity.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Initialize the TF-IDF Vectorizer to convert categories text into numbers
tfidf = TfidfVectorizer(stop_words='english')

# 2. Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['Category'])

# 3. Compute the cosine similarity matrix (how similar each movie's genres are to every other movie)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['Titles'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 2 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 2 most similar movies
    return df['Titles'].iloc[movie_indices]

# Test the recommendation system
print("Recommendations for someone who watched 'Jumanji (1995)':")
print(get_recommendations('Jumanji (1995)'))
'''
Inference: The algorithm looks at the categories (genres) text. If a user finishes 
watching Jumanji (1995), which is tagged with Action, Comedy, Shounen, etc., the
algorithm calculates which other movies share the highest overlap in tags. It 
discovers that Grumpier Old Men and Father of the Bride Part II share highly identical
tags (Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen). As a result, it
automates the recommendation process and instantly suggests those titles to the user's
 feed.
 '''