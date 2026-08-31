#!/usr/bin/env python3
import re
import pandas as pd

df = pd.read_csv("reports/final_submission_source_of_truth/canonical_references.csv")
valid_keys = set(df["CitationKey"])

with open("manuscript.tex") as f:
    text = f.read()

# Find all citations
cited_keys = set()
for match in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", text):
    for k in match.group(1).split(","):
        cited_keys.add(k.strip())

print(f"Total unique cited keys in manuscript.tex: {len(cited_keys)}")
print("Cited keys:", sorted(list(cited_keys)))

missing_in_bib = cited_keys - valid_keys
print(f"Missing in canonical references: {missing_in_bib}")

missing_in_text = valid_keys - cited_keys
print(f"In canonical references but not cited in text: {missing_in_text}")

# Also extract the bibliography keys in the \begin{thebibliography} block
bib_keys = set()
for match in re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", text):
    bib_keys.add(match.group(1).strip())

print(f"Total keys in thebibliography environment: {len(bib_keys)}")
orphaned_bib_keys = bib_keys - cited_keys
print(f"Orphaned keys in thebibliography (not cited in text): {orphaned_bib_keys}")

uncited_in_bib = cited_keys - bib_keys
print(f"Cited in text but missing from thebibliography: {uncited_in_bib}")

