import pandas as pd

# Read the Excel file
excel_file = './predictions.xlsx'
df = pd.read_excel(excel_file)

# Filter matches where either team has >70% probability of under 2.5 goals
high_probability_matches = df[
    (df['u_2.5'] > 0.7) &
    (df['u_3.5'] > 0.75) &
    (df['1x2_d'] > 0.33)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', 'u_2.5']
result_df = high_probability_matches[columns_to_keep]

# Sort the DataFrame by 'u_2.5' in descending order
result_df = result_df.sort_values(by='u_2.5', ascending=False)

# Format 'u_2.5' as percentage with one decimal place
result_df.loc[:, 'u_2.5'] = result_df['u_2.5'].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to a CSV file
csv_file = 'Under_2.5_matches.csv'
result_df.to_csv(csv_file, index=False)

print(f"Extracted {len(result_df)} matches and saved to {csv_file}")

