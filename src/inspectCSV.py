import pandas as pd

df = pd.read_csv("data/extracted/separable_verbs_extracted.csv")

rows = df[df["interfering_prefix_forms"].notna()]

print(rows)
