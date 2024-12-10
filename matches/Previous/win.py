import pandas as pd

# Read the Excel file
excel_file = './predictions.xlsx'
df = pd.read_excel(excel_file)

# Filter matches where either team has >70% probability of under 2.5 goals
high_probability_matches = df[
    (df['1x2_a'] > 0.7)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', '1x2_a']
result_df = high_probability_matches[columns_to_keep]

# Sort the DataFrame by 'u_2.5' in descending order
result_df = result_df.sort_values(by='1x2_a', ascending=False)

# Format 'u_2.5' as percentage with one decimal place
result_df.loc[:, '1x2_a'] = result_df['1x2_a'].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to a CSV file
csv_file = 'Home_matches.csv'
result_df.to_csv(csv_file, index=False)

print(f"Extracted {len(result_df)} matches and saved to {csv_file}")

