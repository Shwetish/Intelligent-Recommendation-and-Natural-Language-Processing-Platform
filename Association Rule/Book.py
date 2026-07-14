#----------------------------------------------------------------------
#Business Understanding:-
#----------------------------------------------------------------------

#1. Business Problem Statement:-
#"Kitabi Duniya, a legendary heritage bookstore, is losing its annual growth, customer
#footfall, and revenue because of fierce competition from online e-commerce platforms 
#and widespread internet reading options."

#2. Business Objective:-
#Group frequently co-purchased genres and book 
#categories using Association Rule Mining (Market Basket Analysis) to
#design optimized physical store layouts, bundle offers, and personalized
#checkout recommendations that attract customers back to the physical 
#store.

#3. Motivation (The "Why")Smart Cross-Selling:-
#If a data pattern reveals that customers who buy CookBks (Cookbooks) 
#almost always pick up ItalCook (Italian Cooking), placing them right 
#next to each other or offering them as a combined "Culinary Bundle" 
#instantly raises the average receipt value.The Heritage Experience: 
#Unlike scrolling online, a curated physical layout encourages impulse
# buying and recreates the nostalgic joy of book browsing.

#4.Constraints :-
#Physical Space Limitations: A physical store has fixed shelf space,
#meaning we cannot display every single book combination 
#together.No Customer Timestamps: The dataset is a simple transactional
#binary matrix (0 or 1). It does not contain customer names, age, time
#of purchase, or price details.

#5. Business Success Criteria:-
#Increase customer physical footfall and boost total sales volume by at least 12%
# over the next two quarters.Measurably increase the cross-category 
#purchase rate (customers buying more than one genre per visit).

#6. ML Success Criteria:-
#Extract meaningful, high-confidence association
# rules using algorithms like Apriori or FP-Growth.Generate rules that
# meet a minimum benchmark threshold (e.g., Support $\ge$ 5%, 
#Confidence $\ge$ 60%, and Lift $>$ 1) to eliminate random noise.

#----------------------------------------------------------------------
#Data Understanding:-
#----------------------------------------------------------------------
'''
Name of Feature	   Simple Description	        Type	      Relevance
ChildBks	Children's Books	            Quantitative,Binary	  High
YouthBks	Youth / Young Adult Novels	    Quantitative,Binary	  High
CookBks	    Cookbooks and Recipe Guides	    Quantitative,Binary   High
DoItYBks	Do-It-Yourself (DIY) 	        Quantitative,Binary   High
             & Craft Books
RefBks	    Reference Books                 Quantitative,Binary   High
            (Dictionaries, Encyclopedias)	
ArtBks	    Art, Photography, 	            Quantitative,Binary   High
            and Design Books
GeogBks	    Geography, Travel Guides, 	    Quantitative,Binary   High
            and Atlases
ItalCook	Specialty Italian Cookbooks	    Quantitative,Binary   High
ItalAtlas	Specialty Italian 	            Quantitative,Binary Medium
            Atlases / Maps
ItalArt	    Specialty Italian Art 	        Quantitative,Binary   High
            and History Books
'''
#---------------------------------------------------------------------
#1. Data Pre-processing & Feature Engineering
#---------------------------------------------------------------------
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
# Load the transaction matrix file
file_path = "C:/11_Association Rules/book.csv"
df = pd.read_csv(file_path)

#============================================================================
#Data Cleaning & Integrity Check
#============================================================================
# Explicitly drop structural indices or blank tracking columns if present
df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Automated Verification Firewall
print("--- Pre-Processing: Data Structural Inventory ---")
print(f"Total Transactions: {df_cleaned.shape[0]} | Tracked Book Categories: {df_cleaned.shape[1]}")
print(f"Total Missing Values Found: {df_cleaned.isnull().sum().sum()}")

# Force explicit integer data types to ensure compatibility with apriori boolean math
df_encoded = df_cleaned.astype(int)
'''
 Inference:
Binary Consistency: The structural check ensures that the entire 
matrix contains zero missing records and is explicitly translated into
 a standardized binary layout (1 for purchase, 0 for no purchase).

Algorithmic Compatibility: Converting elements into tight integer 
formats prevents downstream mathematical failures within mlxtend's 
internal bitwise execution arrays.
'''
#=============================================================================
#2. Model Building
#Application of Apriori Algorithm
#================================================================================

#============================================================================
#  Generate Most Frequent Itemsets (Support Threshold = 5%)
#============================================================================
# Convert DataFrame to boolean type to prevent DeprecationWarning 
# and optimize computational performance.
df_encoded_bool = df_encoded.astype(bool)

# Calculate itemsets that appear in at least 5% of all client transactions
frequent_itemsets = apriori(df_encoded_bool, min_support=0.05, use_colnames=True)

# Sort itemsets by popularity for initial assessment
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Most Frequent Book Itemsets ===")
print(frequent_itemsets.head(10).round(4))
'''
Inference:
Core Anchor Genres: Setting a minimum support boundary of 5% 
successfully filters out isolated, rare transactions while retaining
 categories broad enough to run promotions on.

Volume Concentrations: The output systematically isolates which 
specific book genres—such as CookBks or ChildBks—act as the highest 
foundational volume drivers inside the physical bookstore.
'''
#==============================================================================
#Build Frequent Itemsets & Extract Association Rules
#==============================================================================
#============================================================================
# Extract Strong Association Rules (Confidence Threshold = 60%)
#============================================================================
# Generate operational purchase paths based on customer conditional probability
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.60)

# Sort rules by Lift to expose the strongest cross-purchase dynamics
rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Strongest Book Association Rules (Sorted by Lift) ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).round(4))

#----------------------------------------------------------------------------
# Visualization: Scatter Plot of Rules (Support vs Confidence vs Lift)
#----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules_sorted['support'], rules_sorted['confidence'], 
                      c=rules_sorted['lift'], cmap='viridis', alpha=0.8, s=100)
plt.colorbar(scatter, label='Lift Matrix Strength')
plt.title('Market Basket Association Rules Distribution Profile', fontsize=14, fontweight='bold')
plt.xlabel('Support (Rule Popularity Scale)', fontsize=12)
plt.ylabel('Confidence (Conditional Purchase Probability)', fontsize=12)
plt.axhline(y=0.75, color='crimson', linestyle='--', alpha=0.6, label='High-Certainty Threshold (75%)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

'''Inference:
The "Lift" Multiplier: Rules featuring a Lift value significantly
 greater than 1 confirm a strong statistical relationship between 
 genres. For instance, an entry showing (ItalCook) -> (CookBks) with
 high confidence confirms that a buyer browsing specialized Italian
 culinary books is highly likely to purchase a general recipe guide.

Visual Clusters: The scatter plot maps the structural balance between 
how often an itemset is bought (Support) versus how reliable the 
connection is (Confidence). Items in the top-right cluster represent 
highly reliable, frequent store display pairs.
'''
#==============================================================================
#3. Deployment Solution
#==============================================================================
#============================================================================
# Production Bundle Recommendation Engine Strategy
#============================================================================
def generate_book_recommendations(basket_items, association_rules_df, top_n=3):
    """
    Simulates a digital catalog or checkout terminal algorithm that suggests 
    complementary genres based on current items in the customer's shopping basket.
    """
    recommendations = []
    basket_set = set(basket_items)
    
    for _, row in association_rules_df.iterrows():
        # If the customer has the antecedent items in their basket
        if set(row['antecedents']).issubset(basket_set):
            # Propose the consequent item if they don't already own it
            for item in row['consequents']:
                if item not in basket_set and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= top_n:
                        return recommendations
    return recommendations

# Operational Mock Test: Customer stands at the desk with Art and Youth Books
mock_basket = ['ArtBks', 'YouthBks']
suggested_add_ons = generate_book_recommendations(mock_basket, rules_sorted, top_n=2)

print(f"\n[DEPLOYMENT TERMINAL] Active Items in Basket: {mock_basket}")
print(f"[DEPLOYMENT TERMINAL] Recommended Next Shelf Add-ons: {suggested_add_ons}")
'''
Inference:
Instant Point-of-Sale Personalization: This function transitions the 
static analysis rules into a live script that can run on cash 
registers or information kiosks.

Dynamic Suggestion Flow: By feeding active customer choices directly 
into the script, checkout counters can dynamically suggest targeted 
inventory add-ons right before payment processing, minimizing manual
 intervention.
'''
#---------------------------------------------------------------------
# 4. Business Benefits & Solution Impact
#---------------------------------------------------------------------
Implementing this Association Rule Mining solution gives Kitabi 
Duniya key operational and strategic advantages to reverse its sales 
decline and draw footfall back into the physical store:

1. High-Impact Physical Store Re-Layout
The Move: Stop organizing shelves strictly by static alphabetical
 lists. Instead, use your top association rules to pair items 
 structurally. Place high-lift cross-genre books (such as pairing
ArtBks and DoItYBks or GeogBks and ChildBks) on shared end-cap 
displays.

The Impact: This design encourages casual browsing and prompts 
impulse buys, maximizing the revenue generated from each customer's 
walking path through the store.

2. High-Conversion "Heritage Experience" Bundles
The Move: Group slower-moving, niche regional titles (ItalArt, 
Florence) into curated combo packages alongside stable, high-volume
 baseline genres (ArtBks). Sell these combinations as single, themed 
 gift packages (e.g., "The Renaissance Art & Travel Collection").

The Impact: This approach increases your average order value (AOV) 
and clears slow-moving inventory without resorting to broad, 
margin-killing store-wide discount campaigns.

3. Smart Inventory and Supply Chain Management
The Move: Tie future warehouse re-ordering schedules directly to your
 correlated demand rules. If market demand projections indicate a 
 seasonal spike in children's books (ChildBks), automatically adjust 
 stock levels for high-confidence companion categories like YouthBks
 or CookBks.

The Impact: This optimization protects operational cash flow by 
reducing overstock issues while eliminating lost revenue from 
unexpected out-of-stock situations on your most popular items.
'''