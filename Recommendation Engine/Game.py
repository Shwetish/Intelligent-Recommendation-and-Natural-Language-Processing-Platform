#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1. Business Problem Statement
#A gaming DVD store wants to increase sales by suggesting relevant games to
# customers. They have survey data showing how users rated different games.
# The store needs a recommendation engine to predict which DVDs a customer 
#will likely buy.

#2. Business Objective
#Increase DVD sales by recommending top-selling / highly relevant games to
# each customer, improving customer satisfaction and purchase frequency.

#3. Motivation
#Customers often feel overwhelmed by too many choices.
#Personalized recommendations lead to higher sales and better customer 
#experience.
#Without recommendations, customers may miss popular or relevant games.

#4. Constraints
#Only survey rating data is available (no explicit customer purchase history).
#Limited dataset size (rank 4967 to 5001 shown, but full dataset may have sparse ratings).
#Cold start problem for new games or new customers.
#Needs to work with User-Based Collaborative Filtering (UBCF).

#5. Success Criteria
#5.1 Business Success Criteria
#Increase in sales of recommended game DVDs by at least 10% in 3 months.
#Higher customer repeat purchase rate for game DVDs.
#Positive customer feedback on recommendations.

#5.2 ML Success Criteria
#Achieve good precision@k and recall@k on test data.
#RMSE / MAE between predicted and actual ratings < 1.0 (on 1–5 scale).
#Recommendation engine produces top 5–10 relevant games per user with high 
#overlap to actual top choices.

#-----------------------------------------------------------------------------
#Data Understanding
#-----------------------------------------------------------------------------
'''
Name of Feature      Description                      Type         Relevance
UserID          unique ID no of each user        Numeric,Discrete       High
                who rated a game
Game      Name of the video game being rated     Categorical,Nominal    High 
Rating      Score given by user to a Game         Numeric,Continuous    High
'''
#=================================================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
#=================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load the dataset
df = pd.read_csv("C:/12_Collaborative_Recommendations/Game.csv")
df.columns = ['userId', 'game', 'rating']

#-------------------------------------------------------------------------------------------------
# 1.1 Summary Analysis
#-------------------------------------------------------------------------------------------------
print("=== 1.1 DATASET SUMMARY ===")
print(f"Total rows in raw survey file : {df.shape[0]}")
print(f"Number of unique customers     : {df['userId'].nunique()}")
print(f"Number of unique gaming DVDs   : {df['game'].nunique()}")
print("\nBasic Columns & Missing Types Info:")
print(df.info())
'''
Inference: This gives us the baseline size of our project. We find out exactly how
 many total reviews were collected, how many unique customers participated, and how 
 many different game titles are available in our store's database.
'''
#-------------------------------------------------------------------------------------------------
# 1.2 Univariate Analysis (Checking the Ratings distribution)
#-------------------------------------------------------------------------------------------------
print("\n=== 1.2 UNIVARIATE MEASURES (RATINGS) ===")
print(f"Mean (Average Score)      : {df['rating'].mean():.2f}")
print(f"Median (Middle Value)     : {df['rating'].median():.2f}")
print(f"Mode (Most Common Rating) : {df['rating'].mode()[0]}")
print(f"Standard Deviation        : {df['rating'].std():.2f}")
print(f"Skewness                  : {skew(df['rating'].dropna()):.2f}")
print(f"Kurtosis                  : {kurtosis(df['rating'].dropna()):.2f}")

# Plotting the visual charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram
# Plotting the visual charts cleanly without deprecation warnings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram with smooth density line
sns.histplot(df["rating"], bins=10, kde=True, ax=axes[0], color='royalblue')
axes[0].set_title("Distribution of Ratings")
axes[0].set_xlabel("Rating Value")
axes[0].set_ylabel("Count")

# Plot 2: Frequency Count Plot (Updated syntax to prevent warnings)
sns.countplot(x='rating', data=df, hue='rating', palette='viridis', legend=False, ax=axes[1])
axes[1].set_title("Frequency Distribution of Game Ratings")
axes[1].set_xlabel("Rating Value")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
'''
Inference: This tells us how our shoppers naturally score games. If the average score
 is high (like 4 out of 5), it proves our customers are generally enthusiastic. The 
 charts visually show which exact rating (like 4.0 or 5.0) is handed out most often by
 gamers.
'''
#-------------------------------------------------------------------------------------------------
# 1.3 Bivariate Analysis (User Activity & Game Popularity)
#-------------------------------------------------------------------------------------------------

# Grouping titles to see review velocity vs average scores
popular_games = df.groupby('game').agg(
    review_count=('rating', 'count'),
    avg_rating=('rating', 'mean')
).sort_values(by='review_count', ascending=False).head(5)

print("Top 5 Most Reviewed Games & Their Average Ratings:")
print(popular_games)
'''
Inference: This allows us to instantly spot the store's "blockbuster" games. It 
separates the highly popular games that get a lot of reviews from the obscure games, 
showing us which titles are the driving trendsetters in the store.
'''


#=================================================================================================
# 4. DATA PRE-PROCESSING & DATA MINING
#=================================================================================================

# 1. Strip absolute duplicates
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Removed {duplicate_count} duplicate rows from the data.")

# 2. Reshape raw logs into a User-Item Interaction Matrix
user_item_matrix = df.pivot_table(index='userId', columns='game', values='rating')

# 3. Quantify Grid Sparsity
total_possible_cells = user_item_matrix.size
actual_reviews_count = df.shape[0]
sparsity_percentage = (1 - (actual_reviews_count / total_possible_cells)) * 100

print(f"Reshaped Grid Structure : {user_item_matrix.shape[0]} Users x {user_item_matrix.shape[1]} Games")
print(f"Matrix Sparsity Profile : {sparsity_percentage:.2f}% empty cells")
'''
Inference: Cleaning duplicates ensures our data is completely accurate and fair. The 
new grid layout is mandatory for collaborative filtering. The high Sparsity percentage
shows us that most cells are blank because individual customers only play a few games. Our model's job is to fill in these blank spots with smart guesses.
'''
#=================================================================================================
# 5. MODEL BUILDING (USER-BASED COLLABORATIVE FILTERING)
#=================================================================================================
from sklearn.metrics.pairwise import cosine_similarity

# 1. Mean-Centering to eliminate personal scoring bias
matrix_normalized = user_item_matrix.apply(lambda row: row - row.mean(), axis=1).fillna(0)

# 2. Calculate the structural proximity matrix between profiles using Cosine Similarity
user_similarity_array = cosine_similarity(matrix_normalized)
user_sim_df = pd.DataFrame(user_similarity_array, index=user_item_matrix.index, columns=user_item_matrix.index)

# 3. Complete Recommendation Algorithm Pipeline (MUST BE DEFINED BEFORE RUNNING)
def get_ubcf_dvd_recommendations(target_user, raw_matrix, similarity_df, top_n=5):
    # Cold Start Handler: If user profile doesn't exist, fall back to global popularity hits
    if target_user not in raw_matrix.index:
        return df.groupby('game')['rating'].count().sort_values(ascending=False).head(top_n).index.tolist()
    
    # Extract titles the user hasn't seen/rated yet
    user_history = raw_matrix.loc[target_user]
    unrated_games = user_history[user_history.isnull()].index
    
    # Sort closest taste-match neighbors for this user
    nearest_neighbors = similarity_df[target_user].drop(target_user).sort_values(ascending=False)
    
    predicted_scores = {}
    
    # Calculate scores for games the user has not seen yet
    for game in unrated_games:
        other_user_scores = raw_matrix[game]
        people_who_played = other_user_scores[other_user_scores.notnull()].index
        
        # Isolate the top 5 closest neighbors who have played this specific game
        voter_neighbors = nearest_neighbors.index.intersection(people_who_played)[:5]
        
        if len(voter_neighbors) == 0:
            continue
            
        # Combine neighbor ratings using similarity values as mathematical weights
        neighbor_similarities = nearest_neighbors.loc[voter_neighbors]
        neighbor_ratings = raw_matrix.loc[voter_neighbors, game]
        
        sum_of_weights = neighbor_similarities.abs().sum()
        if sum_of_weights > 0:
            predicted_scores[game] = np.dot(neighbor_ratings, neighbor_similarities) / sum_of_weights

    # Extract top matching items
    top_recommended_games = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [game[0] for game in top_recommended_games]

#-------------------------------------------------------------------------------------------------
# Execution of the engine using a sample consumer profile
#-------------------------------------------------------------------------------------------------

sample_user_id = user_item_matrix.index[0]
recommended_dvds = get_ubcf_dvd_recommendations(sample_user_id, user_item_matrix, user_sim_df, top_n=5)

print(f"\nTop 5 Gaming DVD Recommendations for User [{sample_user_id}]:")
for i, game in enumerate(recommended_dvds, 1):
    print(f" {i}. {game}")
'''
Inference: This is the final engine working. By calculating a weighted average based 
on what similar users liked, it cuts through the empty cells and outputs a personalized
Top 5 list of gaming DVDs that this specific customer is mathematically most likely to
buy.
'''
#======================================================================================
#6. Business Benefits & Impact
#======================================================================================
Boosts Sales Conversions (Higher Revenue): Instead of forcing customers to manually
search through thousands of DVDs, the store automatically displays games tailored to 
their exact preferences. This hyper-personalization turns casual browsers into buyers, directly increasing sales.

Clears Out Warehouse Stock ("Long-Tail" Titles): Standard marketing usually only 
focuses on the top-selling blockbuster games. This algorithm uncovers hidden patterns,
 allowing the store to automatically show niche or older DVD stock to the exact 
 sub-groups who will love them, reducing wasted inventory costs.

Improves Customer Retention & Loyalty: Gamers love shopping at storefronts that 
understand what they like. Providing immediate, accurate suggestions simplifies the 
shopping process and keeps customers coming back to this store instead of moving to 
online competitors.