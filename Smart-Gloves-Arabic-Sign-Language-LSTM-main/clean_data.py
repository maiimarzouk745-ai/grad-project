import pandas as pd
import numpy as np

print("Loading original raw dataset...")
df = pd.read_csv("sign_languagedataset.csv")
df.columns = df.columns.str.strip()
total_rows_before = len(df)

columns_to_clean = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pitch', 'roll']

print(f"Applying Boxplot cleaning on IMU & Angles only. Skipping all 5 flex sensors!")

good_rows_mask = np.ones(len(df), dtype=bool)

for col in columns_to_clean:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        good_rows_mask = good_rows_mask & column_mask

df_clean = df[good_rows_mask]
df_clean.to_csv("sign_languagedataset.csv", index=False)

print(f"Rows Before: {total_rows_before:,} | Rows After: {len(df_clean):,}")
print(" Done! Custom cleaning complete.")
