import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = [
    [str(item).strip() for item in row if pd.notna(item) and str(item).strip()]
    for row in df.values
]

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=encoder.columns_)
frequent = apriori(basket, min_support=0.01, use_colnames=True)

for confidence in [0.2, 0.4, 0.6]:
    rules = association_rules(frequent, metric="confidence", min_threshold=confidence)
    print("\nMinimum Confidence:", confidence)
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

rules = association_rules(frequent, metric="confidence", min_threshold=0.2)
if not rules.empty:
    plt.scatter(rules["support"], rules["confidence"], s=rules["lift"] * 20)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()