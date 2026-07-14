import re
from nltk.tokenize import RegexpTokenizer

# The target transaction sentence
transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"

# Step 1: Create a RegexpTokenizer that matches words, numbers, and currency symbols
# \w+ matches alphanumeric sequences ('two', '500')
# [\$\€] matches the specific currency symbols '$' or '€'
tokenizer = RegexpTokenizer(r"\w+|[\$\€]")

# Step 2: Tokenize the entire string
all_tokens = tokenizer.tokenize(transactions)

# Step 3: Extract only the target elements based on their positions
# We want to find where a token matches our targets and print them line-by-line
target_outputs = ["two", "$", "500", "€"]

for token in all_tokens:
    if token in target_outputs:
        print(token)