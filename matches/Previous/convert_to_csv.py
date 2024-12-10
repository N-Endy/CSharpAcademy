# import pandas as pd

# # Load the Excel file
# file_path = '/home/nelson/Desktop/matches/predictions.xlsx'

# # Read the Excel file
# excel_data = pd.read_excel(file_path)

# # Filter only the required columns
# filtered_data = excel_data[['date', 'league', 'home', 'away', '1x2_h', '1x2_d', '1x2_a', 
#                             'o_1.5', 'o_2.5', 'o_3.5', 'o_4', 'u_1.5', 'u_2.5', 'u_3.5', 'u_4']]

# # Save the filtered data to a CSV file
# csv_file_path = './predictions_filtered.csv'
# filtered_data.to_csv(csv_file_path, index=False)

# print(f"Filtered CSV file saved to {csv_file_path}")


import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('/home/nelson/Desktop/matches/predictions_filtered.csv')

# Function to determine the outcome based on probabilities
def determine_outcome(row):
    probs = [row['1x2_h'], row['1x2_d'], row['1x2_a']]
    return np.argmax(probs)

# Add the ActualResult column
df['ActualResult'] = df.apply(determine_outcome, axis=1)

# Save the updated CSV
df.to_csv('/home/nelson/Desktop/myMLApp/ML/predictions_with_actual.csv', index=False)
