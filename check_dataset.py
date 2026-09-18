import pandas as pd


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/phishing_features.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(data))


# ==========================================
# 2. CHECK DUPLICATE URLs
# ==========================================

duplicate_urls = data["url"].duplicated().sum()

print("\n========== DUPLICATE URL CHECK ==========")
print("Duplicate URL rows:", duplicate_urls)

if duplicate_urls == 0:
    print("No duplicate URLs found.")
else:
    print("Duplicate URLs found.")


# ==========================================
# 3. CHECK DUPLICATE COMPLETE ROWS
# ==========================================

duplicate_rows = data.duplicated().sum()

print("\n========== DUPLICATE ROW CHECK ==========")
print("Duplicate complete rows:", duplicate_rows)

if duplicate_rows == 0:
    print("No completely duplicate rows found.")
else:
    print("Duplicate complete rows found.")


# ==========================================
# 4. CHECK SAME URL WITH DIFFERENT LABELS
# ==========================================

url_label_counts = data.groupby("url")["label"].nunique()

conflicting_urls = url_label_counts[url_label_counts > 1]

print("\n========== LABEL CONFLICT CHECK ==========")
print("URLs having different labels:", len(conflicting_urls))

if len(conflicting_urls) == 0:
    print("No URL has conflicting labels.")
else:
    print("WARNING: Some URLs have different labels.")


# ==========================================
# 5. CHECK LABEL DISTRIBUTION
# ==========================================

print("\n========== LABEL DISTRIBUTION ==========")

print(data["label"].value_counts())

print("\nLabel percentages:")

print(
    (data["label"].value_counts(normalize=True) * 100)
    .round(2)
)


# ==========================================
# 6. CHECK EXACT DUPLICATES
# ==========================================

unique_urls = data["url"].nunique()

print("\n========== UNIQUE URL CHECK ==========")

print("Total rows:", len(data))
print("Unique URLs:", unique_urls)

if unique_urls == len(data):
    print("Every URL is unique.")
else:
    print("Some URLs occur more than once.")


# ==========================================
# 7. SUMMARY
# ==========================================

print("\n========== DATASET SUMMARY ==========")

print("Total rows:", len(data))
print("Unique URLs:", unique_urls)
print("Duplicate URL rows:", duplicate_urls)
print("Duplicate complete rows:", duplicate_rows)
print("Conflicting URL labels:", len(conflicting_urls))

print("\nDataset check completed.")