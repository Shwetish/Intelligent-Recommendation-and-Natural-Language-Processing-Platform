#==============================================================
#Business Understanding
#1. Business Problem Statement:-Online streaming platforms and
# digital video downloads are dominating the entertainment
# market. Physical video rental stores, theaters, or retail 
#media shops are facing sharp revenue declines and loss of 
#customer visits. They need to find a data-driven way to keep 
#customers engaged and purchasing.

#2. Business Objective:-
#Use Association Rule Mining (Market Basket Analysis) on past
# movie transaction records to find out which movies or genres
# people frequently watch or buy together. The marketing team
# can use these patterns to build high-converting movie bundless, 
#design targeted "Recommended for You" sections, and set up 
#attractive store displays.

#3. Motivation (The "Why")Smart Recommendations: If the data
# shows that a customer who buys Gladiator almost always buys 
#Patriot, the system can automatically suggest Patriot at 
#check-out.Themed Movie Marathons: The business can create 
#pre-packaged movie sets (like a "Historical Action Pack" or "
#Fantasy Adventure Collection") to sell multiple titles at 
#once and increase the average order amount.

#4. Constraints:-
#No Movie Timing or Order Details: The data only shows if 
#items were bought together in one basket. It does not show 
#which movie was watched first or the time gap between them.
#Cold Start for New Releases: This model relies entirely on 
#past data, so it cannot predict patterns for a brand-new 
#movie until people start buying it.

#===========================================================
#Data Understanding:-
#===========================================================
'''
Name of Feature   Description               Type    Relevance
Sixth Sense      Tracks if the movie 
                 "The Sixth Sense" was  Quantitative,
                 purchased.              Binary         High
Gladiator        Tracks if the movie 
                 "Gladiator" was purchased.Quantitative,
                                            Binary       High
LOTR1            Tracks if "The Lord of the 
                 Rings: The Fellowship     Quantitative,
                 of the Ring" was purchased Binary       High
Harry Potter1     Tracks if the first 
                 "Harry Potter" movie was  Quantitative,
                  purchased.                 Binary      High
Patriot           Tracks if the movie      Quantitative,
                 "The Patriot" was purchased. Binary     High
LOTR2             Tracks if "The Lord of the 
                  Rings: The Two Towers" was Quantitative,
                  purchased.                  Binary     High
Harry Potter2    Tracks if the second 
                 "Harry Potter" movie was    Quantitative,
                  purchased.                 Binary      High
LOTR             Tracks if other "Lord of 
                 the Rings" franchise media  Quantitative,
                  was purchased.             Binary      High
Braveheart       Tracks if the movie         Quantitative,
                 "Braveheart" was purchased. Binary      High
Green            MileTracks if the movie     Quantitative,
              "The Green Mile" was purchased. Binary     High
'''
#----------------------------------------------------------------------------
# 1. Import Core Analytics & Association Rule Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Configure clean visualization aesthetics
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Load the movie binary transaction spreadsheet
# Note: Using the 'r' prefix eliminates Windows file path backslash errors
file_path = r"C:/11_Association Rules/my_movies.csv"
df = pd.read_csv(file_path)

#============================================================================
#Data Cleaning & Structural Refinement
#============================================================================
# Drop any accidental Excel unnamed index columns if present in the sheet
df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Automated Integrity Firewall Check
print("--- Pre-Processing: Data Structural Inventory ---")
print(f"Total Watch History Logs: {df_cleaned.shape[0]} | Unique Movie Titles: {df_cleaned.shape[1]}")
print(f"Total Null Values Identified: {df_cleaned.isnull().sum().sum()}")

# Enforce strict bitwise integer representations for faster Apriori execution
df_encoded = df_cleaned.astype(int)
'''
Inference:
Ready-to-Use Layout: The integrity check proves the dataset has a 
zero-null, clean binary structure. No text-to-numeric encoding or 
complex parsing is needed.
Optimized Execution: Casting variables to standard integer
 representations guarantees compatibility and speed when parsing 
 frequent combinations through the algorithm.
'''
#====================================================================
#2. Model Building
#Application of Apriori Algorithm
#============================================================================
#Generate Most Frequent Movie Itemsets (Support Threshold = 10%)
#============================================================================
#Convert DataFrame to boolean type to ensure performance and prevent
# Deprecation Warning
df_encoded_bool = df_encoded.astype(bool)

# Isolate single movies or movie pairs that appear in at least 10% of streaming baskets
frequent_itemsets = apriori(df_encoded_bool, min_support=0.10, use_colnames=True)

# Sort the itemsets based on absolute watch frequency
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Most Popular Movie Itemsets (Highest Support) ===")
print(frequent_itemsets.head(10).round(4))
'''
Inference:
Filtering Noise: A baseline minimum support of 10% filters out extreme
niche genres or rare titles, allowing your team to focus strictly on
mainstream, dependable consumer watch habits.
Core Viewership Hubs: The resulting itemsets quickly pinpoint your 
platform's absolute highest-performing titles (like Gladiator or The 
Sixth Sense) before calculating cross-genre combinations.
'''
#===================================================================
#Build Frequent Itemsets & Extract Association Rules
#============================================================================
#Extract Strong Movie Association Rules (Confidence Threshold = 70%)
#============================================================================
# Build conditional viewership paths mapping the likelihood of co-watching
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.70)

# Sort rules by Lift to expose true cross-genre behavioral clusters
rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Strongest Movie Association Rules (Sorted by Lift) ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).round(4))

#----------------------------------------------------------------------------
# 4.2.1 Visualization: Scatter Plot of Rules (Support vs Confidence vs Lift)
#----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules_sorted['support'], rules_sorted['confidence'], 
                      c=rules_sorted['lift'], cmap='coolwarm', alpha=0.85, s=110)
plt.colorbar(scatter, label='Lift Rule Affinity Strength')
plt.title('Film Distribution Association Rules Matrix Profile', fontsize=14, fontweight='bold')
plt.xlabel('Support (Rule Popularity)', fontsize=12)
plt.ylabel('Confidence (Conditional Watch Certainty)', fontsize=12)
plt.axhline(y=0.85, color='purple', linestyle='--', alpha=0.6, label='High-Affinity Threshold (85%)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
'''
Inference:
The "Lift" Connection: A lift score higher than 1.0 confirms a strong
 psychological link between film categories rather than a coincidence.
 For example, if a rule reveals that watching [LOTR1] strongly leads 
 to watching [LOTR2], it quantifies franchise loyalty. If it connects 
 independent titles across genres (e.g., action fans renting specific 
 thrillers), it highlights cross-selling opportunities.
Rule Distribution Profile: The scatter plot visually charts your 
patterns. Data points grouped high up on the y-axis (Confidence) 
reflect viewing pathways you can target for user recommendations.
'''

#======================================================================
#3. Deployment Solution
#============================================================================
#Real-Time Streaming Recommendation Engine Prototype
#============================================================================
def stream_recommend_next_movie(current_watch_history, rules_dataframe, max_recommendations=2):
    """
    Simulates a live user profile landing page or watch next queue on a streaming
    platform that automatically suggests movies matching historical preferences.
    """
    history_set = set(current_watch_history)
    recommended_queue = []
    
    for _, row in rules_dataframe.iterrows():
        # If the user's active view history contains the rule's antecedent
        if set(row['antecedents']).issubset(history_set):
            for movie in row['consequents']:
                if movie not in history_set and movie not in recommended_queue:
                    recommended_queue.append(movie)
                    if len(recommended_queue) >= max_recommendations:
                        return recommended_queue
    return recommended_queue

# Operational Test: User just finished watching 'Gladiator'
active_user_profile = ['Gladiator']
next_up_suggestions = stream_recommend_next_movie(active_user_profile, rules_sorted, max_recommendations=2)

print(f"\n[STREAMING DASHBOARD] User Finished Watching: {active_user_profile}")
print(f"[STREAMING DASHBOARD] Dynamically Injecting into 'Watch Next' Bar: {next_up_suggestions}")
'''
 Inference:
Actionable Automation: This deployment function takes the static 
historical analysis and transforms it into a live software 
recommendation system.

Algorithmic Engagement: 
The moment a movie finishes or is added to an online cart, the engine
updates to show related content, maximizing digital interactions 
without manual overhead.
'''
#=====================================================================
#4. Business Benefits & Solution Impact
#=====================================================================
Implementing this Association Rule solution offers clear commercial 
and marketing advantages for the film distribution company:

1. High-Engagement Automated Recommendation Engines
The Action: Power your digital catalog's "Because You Watched..." or 
"Watch Next" sections with high-lift association rules.

The Benefit: Automatically presenting highly relevant recommendations
 keeps users engaged on the platform longer, driving up subscription 
 retention rates and active view times.

2. High-Converting Curated Bundles and Theme Packages
The Action: Use strong itemsets to build custom digital purchase packs
 or themed marathons (like an "Epic Action Bundle" or "Mystery Night
Collection").

The Benefit: Packaging related films together encourages users to buy
 collections instead of single titles, increasing the platform's 
 average transaction size.

3. Optimized Content Acquisition & Licensing Strategies
The Action: Rely on clear rule paths to guide film distribution 
licensing budgets. If the data shows specific secondary films are 
critical entry points to broader movie marathons, prioritize licensing
 those titles.

The Benefit: This eliminates guesswork when purchasing film rights, 
allowing you to focus your acquisition budget on content that actively
 drives user engagement.