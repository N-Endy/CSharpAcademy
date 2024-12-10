# # import pandas as pd

# # # Read the Excel file
# # excel_file = './predictions.xlsx'
# # df = pd.read_excel(excel_file)

# # # Filter matches based on the specified conditions
# # filtered_matches = df[
# #     ((df['1x2_h'].between(0.6, 0.7)) | (df['1x2_a'].between(0.6, 0.7))) &
# #     (df['o_2.5'] > 0.6)
# # ].copy()

# # # Select relevant columns
# # columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
# # result_df = filtered_matches[columns_to_keep]

# # # Sort the DataFrame by 'o_2.5' in descending order
# # result_df = result_df.sort_values(by='o_2.5', ascending=False)

# # # Format probability columns as percentages with one decimal place
# # for col in ['1x2_h', '1x2_a', 'o_2.5']:
# #     result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# # # Save the result to an Excel file
# # excel_output_file = 'Excel_matches.xlsx'
# # result_df.to_excel(excel_output_file, index=False)

# # print(f"Extracted {len(result_df)} matches and saved to {excel_output_file}")




# import pandas as pd
# from datetime import datetime

# # Read the Excel file
# excel_file = './predictions.xlsx'
# df = pd.read_excel(excel_file)

# # Get today's date
# today = datetime.now().date()

# # Convert 'date' column to datetime, keeping both date and time
# df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# # Filter matches for today's date
# today_matches = df[df['date'].dt.date == today]

# # Filter matches based on the specified conditions
# filtered_matches = today_matches[
#     (
#     ((today_matches['1x2_h'].between(0.6, 0.7)) | (today_matches['1x2_a'].between(0.6, 0.7))) &
#     (today_matches['o_2.5'] > 0.6) &
#     (today_matches['o_3.5'] > 0.5)
#     ) |
#     (
#     ((today_matches['1x2_h'].between(0.6, 0.95)) | (today_matches['1x2_a'].between(0.6, 0.95))) &
#     (today_matches['o_3.5'] > 0.5)
#     )
#     ].copy()

# # Select relevant columns
# columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
# result_df = filtered_matches[columns_to_keep]

# # Sort the DataFrame by 'o_2.5' in descending order
# result_df = result_df.sort_values(by='date', ascending=True)

# # Format the date column to the desired string format
# result_df['date'] = result_df['date'].dt.strftime('%d.%m.%Y %H:%M')

# # Format probability columns as percentages with one decimal place
# for col in ['1x2_h', '1x2_a', 'o_2.5']:
#     result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# # Save the result to an Excel file
# excel_output_file = 'Today_Excel_matches.xlsx'
# result_df.to_excel(excel_output_file, index=False)

# print(f"Extracted {len(result_df)} matches for today and saved to {excel_output_file}")


# import pandas as pd

# # Read the Excel file
# excel_file = './predictions.xlsx'
# df = pd.read_excel(excel_file)

# # Filter matches based on the specified conditions
# filtered_matches = df[
#     ((df['1x2_h'].between(0.6, 0.7)) | (df['1x2_a'].between(0.6, 0.7))) &
#     (df['o_2.5'] > 0.6)
# ].copy()

# # Select relevant columns
# columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
# result_df = filtered_matches[columns_to_keep]

# # Sort the DataFrame by 'o_2.5' in descending order
# result_df = result_df.sort_values(by='o_2.5', ascending=False)

# # Format probability columns as percentages with one decimal place
# for col in ['1x2_h', '1x2_a', 'o_2.5']:
#     result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# # Save the result to an Excel file
# excel_output_file = 'Excel_matches.xlsx'
# result_df.to_excel(excel_output_file, index=False)

# print(f"Extracted {len(result_df)} matches and saved to {excel_output_file}")




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

# Filter matches based on the specified conditions
filtered_matches = today_matches[
    ((today_matches['1x2_h'].between(0.55, 0.65)) | (today_matches['1x2_a'].between(0.55, 0.65))) &
    (today_matches['o_2.5'] > 0.6) &
    (today_matches['o_3.5'] > 0.5)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
result_df = filtered_matches[columns_to_keep]

# Sort the DataFrame by 'o_2.5' in descending order
result_df = result_df.sort_values(by='date', ascending=True)

# Format the date column to the desired string format
result_df['date'] = result_df['date'].dt.strftime('%d.%m.%Y %H:%M')

# Format probability columns as percentages with one decimal place
for col in ['1x2_h', '1x2_a', 'o_2.5']:
    result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to an Excel file
excel_output_file = 'Today_Excel_matches.xlsx'
result_df.to_excel(excel_output_file, index=False)

print(f"Extracted {len(result_df)} matches for today and saved to {excel_output_file}")