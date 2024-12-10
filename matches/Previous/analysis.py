import pandas as pd

# Read the Excel file
excel_file = './predictions.xlsx'
df = pd.read_excel(excel_file)

# Filter matches based on the specified conditions
filtered_matches = df[
    ((df['1x2_h'].between(0.5, 0.55)) | (df['1x2_a'].between(0.5, 0.55))) &
    (df['o_2.5'] > 0.6)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
result_df = filtered_matches[columns_to_keep]

# Sort the DataFrame by 'o_2.5' in descending order
result_df = result_df.sort_values(by='o_2.5', ascending=False)

# Format probability columns as percentages with one decimal place
for col in ['1x2_h', '1x2_a', 'o_2.5']:
    result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to a CSV file
csv_file = 'Analysis_matches.xlsx'
result_df.to_excel(csv_file, index=False)

print(f"Extracted {len(result_df)} matches and saved to {csv_file}")
