#-----------------------------------------------------------------
#Business Understanding:-
#-----------------------------------------------------------------
# Departmental Store Market Basket Optimization
#1. Business Problem Statement:-
#"The departmental store sells thousands of grocery items daily,
# but products are arranged randomly on shelves without a 
#data-driven strategy. This lack of structure leads to low impulse
# buying, smaller customer shopping baskets, and missed 
#cross-selling revenue."

#2. Business Objective:-
#Apply Association Rule Mining (Market Basket Analysis) on daily
# transaction logs to discover which food items, household goods,
# or fresh produce are naturally bought together. The management
# team can use these hidden patterns to redesign shelf layouts,
# place companion items side-by-side, and create high-converting
# promotional bundles.

#3. Motivation :-
#Smart Placement: If data shows that customers buying whole milk
# or yogurt also frequently pick up rolls/buns, placing the 
#bakery section right next to the dairy aisle forces customers 
#to pass both, encouraging an extra impulse purchase.

#Profitable Combos: Slow-moving or high-margin items can be 
#strategically paired or discounted alongside everyday essentials
# (like bread or eggs) to clear stock quickly and increase the 
#store's average transaction size.

#4. Constraints:-
#Varying Transaction Lengths: Unlike clean binary matrices, some
# grocery receipts have only 1 item (e.g., just buying a bottle 
#of water), while family shopping trips might have 15+ items. 
#The algorithm must handle this massive variation without breaking.
#Perishable Lifespans: Groceries include fresh items (like milk,
# fruit, and meat) that spoil quickly. The store cannot store 
#unlimited stock of paired items just because a rule is strong; 
#inventory turnover must match item lifespans.

#5. Business Success Criteria:-
#Increase the store's Average Order Value (AOV) and boost total
# sales revenue by at least 8% within the next quarter.
#Visibly increase footfall across slower sections of the store by
# pulling customers there using highly connected companion item 
#displays.

#6. ML Success Criteria
#Use the Apriori algorithm to convert list-style transaction rows
# into structured itemsets and extract clear, reliable rules.
#Filter out accidental or random single-item purchases by
# tracking rules that meet a minimum population benchmark (e.g.,
# Support ≥ 1%, Confidence ≥ 30%, and a Lift score greater than 
#1.0).
'''
#================================================================
#Data Understanding:-
#================================================================
Name of Feature   Description               Type       Relevance
citrus            fruitFresh             Qualitative,
                   citrus fruits            Nominal      High
                (lemons, oranges)
tropical          fruitFresh 
              tropical fruits            Qualitative,
              (bananas, pineapples)         Nominal      High
whole milk      Standard whole 
                dairy milk               Qualitative, 
                                            Nominal      High
pip fruit      Seeded core fruits 
              (apples, pears, etc.)      Qualitative, 
                                           Nominal       High
other         General fresh vegetables
vegetables    and leafy greens           Qualitative,
                                           Nominal       High
rolls/buns    Freshly baked bread        Qualitative,
               rolls and buns               Nominal      High
pot plants     Decorative indoor         Qualitative,
                potted plants               Nominal     Medium
beef            Raw beef meat cuts       Qualitative, 
                                            Nominal      High
frankfurter    Pre-packed frankfurter    Qualitative,
                sausages                  Nominal        High
chicken        Raw fresh chicken meat    Qualitative, 
                                           Nominal       High
butter         Dairy butter blocks       Qualitative, 
                                           Nominal       High
fruit/vegetable Packaged fruit or        Qualitative,
 juice           vegetable juices          Nominal       High
packaged fruit/ Pre-washed or            Qualitative,
vegetables      pre-packaged produce units Nominal       High
chocolate      Sweet cocoa chocolate     Qualitative,
               bars or candy              Nominal        High
specialty      Single-serve candy        Qualitative,
 bar             or snack bars           Nominal        Medium
butter milk    Cultured dairy            Qualitative,
               buttermilk liquid         Nominal         High
bottled       Packaged plain             Qualitative,
water         drinking water bottles     Nominal         High
yogurt         Standard flavored or      Qualitative,
              plain dairy yogurt         Nominal         High
sausage       Assorted standard          Qualitative,
              processed sausages         Nominal         High
brown        Whole grain or wheat        Qualitative,
bread        bakery bread loaves         Nominal         High
soda          Carbonated soft drinks     Qualitative,
              and sweet beverages        Nominal         High
hamburger     Ground minced              Qualitative,
meat          meat patties               Nominal         High
root          Underground vegetables     Qualitative,
vegetables  (carrots, potatoes, onions)  Nominal         High
pork         Raw pork meat portions      Qualitative, 
                                         Nominal         High

'''
#====================================================================
#1. Data Pre-processing & Feature EngineeringPython
#  Import Core Analytics & Specialized Pre-processing Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Configure clean visualization aesthetics
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Raw Transaction List Dataset
#============================================================================
# Since rows have varying columns, we read row-by-row to prevent layout
# crashes
file_path = r"C:/11_Association Rules/groceries.csv"

raw_baskets = []
with open(file_path, 'r') as file:
    for line in file:
        # Split elements by comma and strip extra spaces/newlines
        basket = [item.strip() for item in line.split(',') if item.strip()]
        if basket:
            raw_baskets.append(basket)

#============================================================================
#One-Hot Matrix Feature Engineering (Crucial Step for Text Data)
#============================================================================
# Initialize the Transaction Encoder to turn items into table columns
te = TransactionEncoder()
te_array = te.fit(raw_baskets).transform(raw_baskets)

# Convert transformed boolean array into a clean binary dataframe
df_encoded = pd.DataFrame(te_array, columns=te.columns_).astype(int)

# Automated Verification Firewall
print("--- Pre-Processing: Grocery Inventory Cleaned ---")
print(f"Total Customer Baskets: {df_encoded.shape[0]} | Unique Products Tracked: {df_encoded.shape[1]}")
'''
 Inference:Successful Layout Transformation: The custom loader and
 TransactionEncoder successfully caught the unequal text lists and 
 compiled them into a uniform matrix ($1$ for item present on receipt,
 $0$ for missing). Blank spreadsheet cells were dropped completely 
 instead of corrupting the model as missing data errors.Dimensional
 Clarity: The final column footprint outlines the absolute checklist 
 of unique products sold across your retail department store.
 '''
#=====================================================================
# 2. Model Building4.1 Application of Apriori Algorithm
# Generate Most Frequent Itemsets (Support Threshold = 1%)
#============================================================================
# In daily grocery retail, 1% support means an item pair is bought hundreds of times
# Convert DataFrame to boolean type to ensure compatibility and speed
df_encoded_bool = df_encoded.astype(bool)

# Calculate itemsets that appear in at least 1% of all client transactions
frequent_itemsets = apriori(df_encoded_bool, min_support=0.01, use_colnames=True)

# Sort itemsets by sales popularity
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Most Common Grocery Itemsets ===")
print(frequent_itemsets.head(10).round(4))
'''Inference:
    The Essentials Benchmark: Using a tight $1\%$ support value allows
    the model to spot realistic food trends without letting massive
    staple volume completely hide smaller high-value pairs.Volume 
    Drivers: The initial rows isolate essential household pillars—like
    whole milk, other vegetables, and rolls/buns—acting as the core 
    grocery volume blocks.
'''
#======================================================================
# Build Frequent Itemsets & Extract Association Rules
# Extract Grocery Rules (Confidence Threshold = 30%)
#============================================================================
# Grocery lists are highly dynamic, so a 30% baseline captures strong consumer intents
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)

# Sort rules by Lift to expose reliable cross-shopping combinations
rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Strongest Grocery Association Rules (Sorted by Lift) ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).round(4))

#----------------------------------------------------------------------------
# 4.2.1 Visualization: Scatter Plot of Grocery Rules
#----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules_sorted['support'], rules_sorted['confidence'], 
                      c=rules_sorted['lift'], cmap='YlOrRd', alpha=0.75, s=100)
plt.colorbar(scatter, label='Lift Correlation Strength')
plt.title('Grocery Market Basket Rule Connectivity Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Support (How often pair is bought total)', fontsize=12)
plt.ylabel('Confidence (Likelihood of buying companion item)', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference:
Complementary Food Pairings: Rules recording a high Lift score 
(e.g., (root vegetables) -> (other vegetables)) represent high-priority
 cross-shopping habits. This proves that buying item A dramatically 
 shifts the likelihood of adding item B to the cart.Chart Clustering:
 The visual scatter markers map out rule trends. High-intensity dots 
 floating high on the confidence scale give the store clear directions
 on which items are highly interdependent during daily shopping runs.
 '''
#======================================================================
#3. Deployment Solution
#Real-Time Grocery Checkout Recommendation System
#============================================================================
def live_grocery_recommender(current_cart, association_rules_df, limit=2):
    """
    Simulates a live self-checkout kiosk or digital app basket that proposes 
    smart add-ons to grocery shoppers based on current ingredients in cart.
    """
    cart_set = set(current_cart)
    suggested_items = []
    
    for _, row in association_rules_df.iterrows():
        if set(row['antecedents']).issubset(cart_set):
            for product in row['consequents']:
                if product not in cart_set and product not in suggested_items:
                    suggested_items.append(product)
                    if len(suggested_items) >= limit:
                        return suggested_items
    return suggested_items

# Live Cashier Simulation: Customer scans Root Vegetables and Yogurt
active_shopping_cart = ['root vegetables', 'yogurt']
checkout_suggestions = live_grocery_recommender(active_shopping_cart, rules_sorted, limit=2)

print(f"\n[CHECKOUT KIOSK] Items Scanned: {active_shopping_cart}")
print(f"[CHECKOUT KIOSK] Smart Screen Promotion Recommendation: {checkout_suggestions}")
'''
Inference:
    Point-of-Sale Actions: This code transforms raw mathematical 
    patterns into a live assistant tool for checkout screens.Instant
    Digital Promos: By matching items scanned at self-checkout 
    terminals with your rules, the store can automatically show quick 
    discount prompts for companion items before payment processing 
    wraps up.
'''
#=========================================================
4. Business Benefits & Solution Impact
#===========================================================
Implementing this Market Basket Analysis pipeline helps a departmental
store turn raw checkout data into a strategic business asset:
1. High-Margin Shelf Layout OptimizationThe Strategy: Do not keep all
dairy items or fresh vegetables isolated away in deep separate corners.
Use high-lift cross-category patterns to redesign store walking paths.
For example, place high-lift bakery goods (rolls/buns) or recipe
ingredients right along the path to the high-volume milk coolers.
The Benefit: This intentional layout exposes everyday shoppers to
complementary items, prompting impulse purchases and increasing the 
stores overall profitability.
2. High-Performance Promotional Combo BundlingThe Strategy: Avoid 
running generic store-wide discounts that eat into profit margins. 
Instead, build specific ingredient bundles based on strong itemset 
rules (e.g., grouping soup vegetables together or pairing baking 
items with fresh dairy essentials).The Benefit: This strategic 
pricing encourages shoppers to buy multiple items at once, effectively
 clearing inventory for perishable goods before they spoil.
3. Smart Inventory Management and Reduced SpoilageThe Strategy: 
Connect the stock replenishment system with your association rules.
 When the store system orders extra stock of high-volume anchor items
like whole milk or root vegetables, automatically update purchasing 
alerts for highly correlated companion items.The Benefit: This 
synchronizes inventory levels with actual consumer behaviors, 
preventing stockouts on popular items while minimizing costly 
spoilage for fresh groceries.
