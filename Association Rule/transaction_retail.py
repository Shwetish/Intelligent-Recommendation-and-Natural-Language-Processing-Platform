#==================================================================================== 
#Business Understanding:-
#====================================================================================
#Indian Retail Shelf Placement Optimization
#1. Business Problem Statement:-
#"The store sells a variety of home decor items, gifts, and household goods daily, 
#but items are currently arranged on shelves based on intuition rather than customer
# buying data. Because related items are spread across different aisles, customers 
#often leave without seeing them, resulting in smaller basket sizes and lower overall
# sales.

#2. Business Objective:-
#Analyze the daily transaction logs using Association Rule Mining (Market Basket 
#Analysis) to find out exactly which retail products (like matching themes, colored 
#alarm clocks, or gift items) are naturally bought together. The manager can then 
#place these companion products side-by-side on the shelves to encourage impulse
# purchases and increase customer footfall across slower areas of the store.

#3. Motivation :-
#Convenience & Impulse Sales: If data reveals that a
# customer purchasing an ALARM clock also highly tends to buy a matching WHITE or 
#HEART decor item, placing them on the same shelf layout ensures the customer notices
# both immediately, sparking an easy extra sale.Cross-Promotion: The store can bundle
# slower-moving decorative stock directly next to highly popular household essentials
# to clear inventory faster without running margin-killing storewide discounts.

#4. Constraints:-
#Inconsistent Basket Lengths: Some shoppers buy a single small gift item, while 
#others buy large sets of household goods, leaving a lot of blank spaces (NA) in the
# raw transaction records. The analytics pipeline must handle these varying lengths 
#cleanly.Physical Shelf Space Limits: The store has limited physical shelf space. 
#The manager cannot pair every single item together; only the highest-performing, 
#most profitable rules should be given prime eye-level shelf space.

#5. Business Success CriteriaIncrease the store's Average Transaction Value (ATV) by
# getting customers to add at least one extra item to their basket per visit.Drive 
#up total store sales revenue by 10% over the next quarter by improving the physical
# product layout.6. ML Success CriteriaExtract strong, reliable purchasing patterns
# using the Apriori algorithm from the unstacked raw text transaction sheets.Filter
# out accidental combinations by focusing only on rules that meet strict statistical
# benchmarks: a minimum Support $\ge$ 2%, Confidence $\ge$ 50%, and a Lift score 
#greater than 1.0.
#===============================================================================
##Data Understanding:-
#===============================================================================
'''
Name of Feature                  Description                     Type       Relevance
ALARM                Various styles of retro or          Qualitative,
                     colorful alarm clocks.              Nominal             High
WHITE                White-themed home accessories       Qualitative
                     and paint-finish gifts.             Nominal             High
HEART                Love-themed decor items,            Qualitative,
                    heart boxes, or hanging hearts.      Nominal             High
HANGING             Wall-mounted hooks,hanging           Qualitative,
                   lanterns, or hanging mirrors.         Nominal             High
HOLDER              Kitchen or bathroom utility holders  Qualitative,
                     (like T-light holders).             Nominal             High
BOX                Assorted decorative storage           Qualitative,
                   boxes and lunchboxes.                 Nominal             High
BOTTLE              Water bottles, glass jars,           Qualitative,
                     or decorative infusers.             Nominal             High
CHILDREN           Kids' cutlery sets, space-themed      Qualitative,
                   items, or school toys.                Nominal             Medium
APRON              Fabric kitchen aprons                 Qualitative,
                   for kids or adults.                   Nominal             Medium
CAKE               Baking tins, cake stands, and         Qualitative, 
                    decorating tools.                    Nominal             Medium
JAM                Glass storage jars specifically       Qualitative,
                   for homemade jams or preserves.       Nominal             Medium
NA                 Blank spaces used to pad empty        Text Placeholder    Irrelevant
                   cells on short receipts.               
'''
#==============================================================================
#1. Data Pre-processing & Feature Engineering
#==============================================================================
#----------------------------------------------------------------------------
# 1. Import Core Analytics & Association Rule Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
# FIX: Import TransactionEncoder from the correct preprocessing module
from mlxtend.preprocessing import TransactionEncoder

# Configure clean visualization aesthetics
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load and Transform Retail Transaction Lists
#============================================================================
file_path = r"C:/11_Association Rules/transactions_retail1.csv"

raw_retail_baskets = []
with open(file_path, 'r') as f:
    for line in f:
        # Split by comma, remove quotes, strip whitespace, and filter out 'NA' markers
        items = [
            item.replace("'", "").replace('"', '').strip() 
            for item in line.split(',') 
            if item.strip() and item.strip() != 'NA'
        ]
        if items:
            raw_retail_baskets.append(items)

#============================================================================
# Convert Text Items to a Binary Structured Matrix
#============================================================================
te = TransactionEncoder()
te_matrix = te.fit(raw_retail_baskets).transform(raw_retail_baskets)

# Reconstruct into a optimized boolean dataframe
df_encoded = pd.DataFrame(te_matrix, columns=te.columns_)

# Automated Integrity Verification Firewall
print("--- Pre-Processing: Data Structural Inventory ---")
print(f"Total Transactions Parsed: {df_encoded.shape[0]} | Unique Inventory Items: {df_encoded.shape[1]}")
print(f"Total Null/NA Columns Fixed: Handled during stream input split")
''' Inference:
Structural Reshaping Success: The preprocessing script successfully drops all 
meaningless NA text strings from the shorter transactions and maps out each
individual gift, home decor, or hardware item into its own structured boolean column.
Format Ready: The output data frames are clean and fully encoded into single matrix
 arrays, resolving uneven item counts per row so that numerical support counters can
 be run.
'''
#=================================================================================
#2. Model Building
#Application of Apriori Algorithm
#===================================================================================

#============================================================================
# Generate Most Frequent Retail Itemsets (Support Threshold = 2%)
#============================================================================
# Isolate retail items or item combinations that appear in at least 2% of baskets
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# Sort the itemsets based on overall purchase frequency
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Most Popular Retail Itemsets (Highest Support) ===")
print(frequent_itemsets.head(10).round(4))
'''
Inference:
Identifying Demand Drivers: Setting the minimum support limit to 2% removes 
low-volume, single-sale custom items while keeping track of highly popular decor 
pieces (like WHITE, HEART, ALARM, or HOLDER variants).
Baseline Volume Mapping: This step tells the manager exactly which product stock 
keeps the store busy, serving as the baseline for determining product pairs.
'''
#=============================================================================
# Build Frequent Itemsets & Extract Association Rules
#============================================================================
# Extract Strong Shelf Placement Rules (Confidence Threshold = 50%)
#============================================================================
# Uncover conditional purchasing rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)

# Sort rules by Lift to bubble up the most direct product affinities
rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

print("\n=== Top 10 Strongest Product Placement Rules (Sorted by Lift) ===")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).round(4))

#----------------------------------------------------------------------------
# Visualization: Scatter Plot of Rules (Support vs Confidence vs Lift)
#----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules_sorted['support'], rules_sorted['confidence'], 
                      c=rules_sorted['lift'], cmap='plasma', alpha=0.8, s=100)
plt.colorbar(scatter, label='Lift Rule Affinity Strength')
plt.title('Retail Store Product Placement Rules Profiling', fontsize=14, fontweight='bold')
plt.xlabel('Support (Rule Popularity)', fontsize=12)
plt.ylabel('Confidence (Conditional Purchase Certainty)', fontsize=12)
plt.axhline(y=0.65, color='red', linestyle='--', alpha=0.6, label='High-Certainty Shelf Boundary (65%)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

'''
 Inference:
The "Lift" Visual Clue: A lift score higher than 1.0 indicates that buying item A 
significantly increases the likelihood that a customer will buy item B. For instance,
 if the rules show high lift values for specific product paths (like grouping colored
 alarm clocks or matching style themes together), the store manager should arrange 
 these lines near each other.
Scatter Distribution Insights: The visualization acts as a roadmap for shelf 
management. Points in the upper-left corner denote highly predictable customer 
patterns (High Confidence), providing reliable guidance for updating store floor 
layouts.
'''
#================================================================================
#3. Deployment Solution
#============================================================================
# Real-Time Smart Checkout Placement Suggestion Engine
#============================================================================
def get_shelf_placement_recommendations(current_customer_basket, rules_dataframe, max_items=2):
    """
    Simulates a smart billing counter or digital inventory tablet that 
    suggests complementary items to display next to items already in the basket.
    """
    basket_set = set(current_customer_basket)
    placement_suggestions = []
    
    for _, row in rules_dataframe.iterrows():
        # Check if the items in the basket match the antecedent rules
        if set(row['antecedents']).issubset(basket_set):
            for product in row['consequents']:
                if product not in basket_set and product not in placement_suggestions:
                    placement_suggestions.append(product)
                    if len(placement_suggestions) >= max_items:
                        return placement_suggestions
    return placement_suggestions

# Operational Store Test: Customer brings a specific colored item line to counter
active_basket = ['ALARM']
suggested_displays = get_shelf_placement_recommendations(active_basket, rules_sorted, max_items=2)

print(f"\n[POS TERMINAL] Scanned Product Line: {active_basket}")
print(f" [POS TERMINAL] Recommended Nearby Impulse-Rack Products: {suggested_displays}")
''' Inference:
Operationalizing Analytics: This deployment engine transforms historical patterns 
into an actionable tool for checkout terminals or handheld inventory tablets.

Dynamic Shelf Adjustments: Floor staff can use these automated suggestions to set up
 secondary point-of-sale displays or optimize front-of-counter endcaps, capitalizing
 on high-interest items right before checkout.
'''
#====================================================================================
#4. Business Benefits & Solution Impact
#====================================================================================
By leveraging these insights, the retail store manager can implement layout
improvements that directly benefit business operations:
1.Data-Driven Shelf Layouts to Increase Average Basket Size
The Strategy: Place high-lift associated items next to each other on shelves. If the 
data shows ALARM clocks of different styles or colors are bought together, display 
them as a single, coordinated set rather than splitting them up.

The Benefit: This encourages customers to buy multi-piece sets or matching collections
 helping to expand cross-category sales and grow the stores average transaction 
value.

2.High-Impact Checkout Displays That Boost Footfall
The Strategy: Use high-confidence item pairings to set up endcap displays and focal
 points along primary walkways, drawing shoppers toward less-visited sections of the
 store.

The Benefit: This layout encourages natural exploration through different aisles, 
extending shopping trips and increasing exposure to seasonal or high-margin product
 lines.

3.Clearer Stock Rotations & Lower Holding Costs
The Strategy: Align seasonal decor and inventory bundles with verified purchase 
patterns instead of relying on seasonal guesswork.

The Benefit: This approach helps minimize over-ordering on slow-moving inventory,
 quickens inventory turnaround times, and ensures high-demand companion items are
 consistently stocked together.

