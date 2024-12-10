import pandas as pd
from datetime import datetime

# Read the Excel file
excel_file = './predictions.xlsx'
df = pd.read_excel(excel_file)

# Get today's date
today = datetime.now().date()

# Convert 'date' column to datetime, keeping both date and time
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Filter matches for today's date
today_matches = df[df['date'].dt.date == today]

# Filter matches where either team has >70% probability of under 2.5 goals
high_probability_matches = today_matches[
    (today_matches['o_3.5'] > 0.6)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', 'o_3.5']
match_df = high_probability_matches[columns_to_keep]

# Sort the DataFrame by 'u_2.5' in descending order
match_df = match_df.sort_values(by='o_3.5', ascending=False)

match_df['date'] = match_df['date'].dt.strftime('%d.%m.%Y %H:%M')

# Format 'u_2.5' as percentage with one decimal place
match_df.loc[:, 'o_3.5'] = match_df['o_3.5'].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to a CSV file
csv_file = 'Over_3.5_matches.csv'
match_df.to_csv(csv_file, index=False)

print(f"Extracted {len(match_df)} matches and saved to {csv_file}")
