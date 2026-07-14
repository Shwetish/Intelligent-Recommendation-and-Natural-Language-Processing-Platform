#-----------------------------------------------------------
# Business Understanding: Mobile Phone Color Optimization
#-------------------------------------------------------------
#1. Business Problem Statement
#"The mobile phone manufacturing company is launching three new 
#smartphones but risks wasting its marketing budget and missing 
#sales targets by using a one-size-fits-all traditional marketing
# campaign, instead of matching regional consumer color preferences

#2. Business Objective
#Apply Association Rule Mining (Market Basket Analysis) to your
# historical binary color sales matrix to discover which phone
# colors are frequently purchased together, helping the marketing
# team design high-converting color bundles, targeted display 
#setups, and regional stock allocations.

#3. Motivation (The "Why")
#Curated Color Bundles: If data shows that customers purchasing
# a red phone are highly likely to purchase a white phone in the
# same transaction (perhaps for family members or matching 
#accessories), the company can instantly launch a discounted 
#"Dual-Color Couple Pack" to drive higher order volumes.

#Smart Store Display & Cross-Marketing: Instead of scattering
# phone variants randomly, high-confidence associated color units
# can be displayed side-by-side on store banners and retail
# kiosks to stimulate impulse buying.

#4. Constraints
#Limited Launch Portfolio: The upcoming launch is strictly 
#limited to three phone models, meaning the team must carefully
# pick the absolute best color combinations to avoid stock
# overhead.

#Sparse Matrix / Zero Context: The historical dataset consists 
#only of a simple binary transaction tracker (1 = purchased, 
#0 = not purchased). It lacks customer age, buying power, 
#specific storage variants, or time-stamps.

#5. Business Success Criteria
#Boost new model launch sales volume by at least 15% in the first
# quarter compared to previous traditional rollouts.

#Maximize initial campaign ROI by reducing the accumulation of
# unpopular, slow-moving color inventories in retail stores.

#6. ML Success Criteria
#Extract strong, clear association rules using the Apriori 
#algorithm with a defined baseline (e.g., Support ≥ 10% and
# Confidence ≥ 60%).

#Filter out random purchasing noise by isolating color pairs that
# achieve a Lift score greater than 1.0, proving a real 
#statistical relationship exists between those color choices.

#-----------------------------------------------------------
# Data Understanding:-
#-------------------------------------------------------------
'''
Name of Feature    Description              Type         Relevance
red              Tracks if a red mobile    Quantitative,
              phone model was purchased.   Binary          High
white         Tracks if a white mobile 
              phone model was purchased.  Quantitative, 
                                           Binary          High
green         Tracks if a green mobile 
             phone model was purchased.   Quantitative, 
                                           Binary          High
yellow       Tracks if a yellow mobile
             phone model was purchased.   Quantitative, 
                                           Binary          High
orange       Tracks if an orange mobile 
             phone model was purchased.   Quantitative, 
                                           Binary          High
blue         Tracks if a blue mobile phone
             model was purchased.         Quantitative, 
                                           Binary          High
'''
#=========================================================
#1. Data Pre-processing & Feature Engineering
#=========================================================
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
# Load the phone color transaction matrix file
# Note: Ensure you use an 'r' prefix or forward slashes to avoid path errors
file_path = r"C:/11_Association Rules/book.csv"
df = pd.read_csv(file_path)

#============================================================================
# 3.1. Data Cleaning & Integrity Check
#============================================================================
# Drop any accidental unnamed indexing columns if they exist
df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Automated Verification Firewall
print("--- Pre-Processing: Data Structural Inventory ---")
print(f"Total Sales Records: {df_cleaned.shape[0]} | Tracked Phone Colors: {df_cleaned.shape[1]}")
print(f"Total Missing Values Found: {df_cleaned.isnull().sum().sum()}")

# Force explicit integer data types to ensure strict bitwise Apriori execution
df_encoded = df_cleaned.astype(int)
''' Inference:Clean Boolean Structure: The data structural step
 proves that your dataset is completely clean, zero-null, and 
 perfectly formatted as a binary transaction layout 
 ($1$ = color sold, $0$ = color not sold).Memory 
 Optimization: Casting the values to integer type prepares the
 raw matrix for fast bitwise comparisons when evaluating 
 combinations inside the mlxtend algorithm.
 '''
#===============================================================
#2.Model Building4.1 Application of Apriori Algorithm
#Generate Most Frequent Itemsets (Support Threshold = 10%)
#============================================================================
# FIX: Convert DataFrame to boolean type to ensure performance and prevent DeprecationWarning
df_encoded_bool = df_encoded.astype(bool)

# Find color or color combinations that appear in at least 10% of sales invoices
frequent_itemsets = apriori(df_encoded_bool, min_support=0.10, use_colnames=True)

# Sort the itemsets by overall popularity
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Most Frequent Phone Color Combinations ===")
print(frequent_itemsets.head(10).round(4))
'''
Inference:
    Isolating Core Color Drivers: By locking down a 
minimum support threshold of $10\%$, we eliminate sporadic or 
rare custom-order colors and immediately highlight the baseline 
color engines driving volume (e.g., standard colors like white 
or red).Volume Concentrations: This step directly tells the 
manufacturing team which colors are universally loved on their
 own before evaluating mixed pairs.
 '''
#================================================================
#Build Frequent Itemsets & Extract Association Rules.
#Extract Strong Association Rules (Confidence Threshold = 60%)
#============================================================================
# Calculate conditional probability pathways for co-purchased colors
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.60)

# Sort the generated rules by Lift to expose the strongest cross-purchase bonds
rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Strongest Phone Color Association Rules (Sorted by Lift) ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).round(4))

#----------------------------------------------------------------------------
# 4.2.1 Visualization: Scatter Plot of Rules (Support vs Confidence vs Lift)
#----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules_sorted['support'], rules_sorted['confidence'], 
                      c=rules_sorted['lift'], cmap='plasma', alpha=0.8, s=120)
plt.colorbar(scatter, label='Lift Matrix Strength')
plt.title('Phone Color Association Rules Distribution Profile', fontsize=14, fontweight='bold')
plt.xlabel('Support (Rule Popularity Scale)', fontsize=12)
plt.ylabel('Confidence (Conditional Purchase Probability)', fontsize=12)
plt.axhline(y=0.70, color='darkgreen', linestyle='--', alpha=0.6, label='High-Certainty Threshold (70%)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
'''
Inference:The "Lift" Multiplier Rule: Rules showing a Lift score
 substantially greater than $1.0$ mean that purchasing the first
 color heavily influences the purchase of the second color. 
 For example, if a rule shows (red) -> (white) with high lift, 
 it proves that red phone buyers are highly likely to add a 
 white phone or a matching color accessory to the same 
 transaction.Scatter Distribution: The scatter plot maps your 
 rules visually. Dots sitting in the upper-right area represent
 color patterns that are both highly popular (Support) and highly
 predictable (Confidence), making them perfect targets for active 
 marketing campaigns.
'''
#===============================================================
#3. Deployment Solution
# 5.1 Real-Time Product Bundle Recommendation Engine
#============================================================================
def suggest_phone_bundles(current_cart, rules_dataframe, top_suggestions=2):
    """
    Simulates a live e-commerce cart or retail POS checkout system that 
    suggests alternative/complementary colors to buyers based on active choices.
    """
    cart_set = set(current_cart)
    recommended_colors = []
    
    for _, row in rules_dataframe.iterrows():
        # Check if the active cart matches the antecedent color pattern
        if set(row['antecedents']).issubset(cart_set):
            for color in row['consequents']:
                if color not in cart_set and color not in recommended_colors:
                    recommended_colors.append(color)
                    if len(recommended_colors) >= top_suggestions:
                        return recommended_colors
    return recommended_colors

# Operational Production Test: Customer picks a Blue phone variant
active_customer_cart = ['blue']
recommended_add_ons = suggest_phone_bundles(active_customer_cart, rules_sorted, top_suggestions=2)

print(f"\n[RETAIL TERMINAL] Active Item Added to Cart: {active_customer_cart}")
print(f"[RETAIL TERMINAL] Dynamic Recommendations for Marketing Banner: {recommended_add_ons}")
'''
 Inference:
     Automated Upselling: This deployment block takes the 
     mathematical patterns and turns them into a live software
     tool.Point-of-Sale Actions: When a customer selects a
     specific device color, the checkout screen or retail tablet
     can instantly pop up a banner recommending companion 
     variants or matching cases, maximizing the average order
     value right before payment.
'''
#================================================================    
4. Business Benefits & Solution Impact:- 
#=====================================================================
By utilizing this Association Rule solution, the mobile phone
 manufacturer shifts away from guesswork and gains key 
 competitive advantages for its upcoming launch:
     1. High-Converting Product Bundles & "Multi-Packs"
     The Strategy: Use high-lift color rules to create 
     promotional launch packages. If blue and white show an 
     unbreakable purchase relationship, run a launch promotion 
     such as a "Family Share Pack" or "His & Hers Bundle" 
     combining these variants at a slight discount.
     The Benefit: This raises the average units per transaction 
     (UPT) and accelerates overall inventory turnover right 
     from the launch date.
     2. Regional Inventory Optimization & Less Dead Stock
     The Strategy: Guide manufacturing volumes based on verified
     combination rules. Do not print equal volumes of every color
     Use historical regional support numbers to allocate color 
     stock exactly to regions where those rules track the 
     highest.The Benefit: This saves massive capital by 
     preventing the accumulation of slow-moving, unpopular 
     colors (like yellow or orange) in warehouses while 
     eliminating out-of-stock lost revenue on high-demand variants
    .3. High-Impact Retail Displays and Digital AdsThe Strategy:
        Redesign physical store showcases, kiosks, and digital 
        landing pages to pair associated colors side-by-side. 
        Instead of grouping phones strictly by storage size, 
        group them by verified consumer color paths.
        The Benefit: This leverages natural visual associations,
        triggering impulse buys and organic interest among 
        shoppers browsing the store without increasing the 
        actual marketing budget.