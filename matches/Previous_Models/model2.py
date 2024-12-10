# # INITIAL ONE


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Load the data
# df = pd.read_csv('Data/EPL.csv')

# # Convert 'Date' to datetime
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# # Create dictionaries for team and location encoding
# def create_encoding_dict(column):
#     unique_values = df[column].unique()
#     return {value: i for i, value in enumerate(unique_values)}

# team_encoding = create_encoding_dict('Home Team')
# location_encoding = create_encoding_dict('Location')

# # Encoding function that handles new values
# def encode_column(value, encoding_dict):
#     if value not in encoding_dict:
#         encoding_dict[value] = len(encoding_dict)
#     return encoding_dict[value]

# # Encode teams and locations
# df['Home Team Encoded'] = df['Home Team'].apply(lambda x: encode_column(x, team_encoding))
# df['Away Team Encoded'] = df['Away Team'].apply(lambda x: encode_column(x, team_encoding))
# df['Location Encoded'] = df['Location'].apply(lambda x: encode_column(x, location_encoding))

# # Extract features from 'Date'
# df['DayOfWeek'] = df['Date'].dt.dayofweek
# df['Month'] = df['Date'].dt.month
# df['Hour'] = df['Date'].dt.hour

# # Calculate goal difference
# def calculate_goal_difference(result):
#     if isinstance(result, str) and ' - ' in result:
#         home_goals, away_goals = result.split(' - ')
#         return int(home_goals) - int(away_goals)
#     else:
#         return None

# df['GoalDifference'] = df['Result'].apply(calculate_goal_difference)

# # Remove rows with None values in GoalDifference
# df = df.dropna(subset=['GoalDifference'])

# # Add new features: team's average goal difference in last 3 matches
# def team_performance(team, n=3):
#     team_matches = df[(df['Home Team'] == team) | (df['Away Team'] == team)].sort_values('Date')
#     goal_diffs = []
#     for _, match in team_matches.iterrows():
#         if match['Home Team'] == team:
#             goal_diffs.append(match['GoalDifference'])
#         else:
#             goal_diffs.append(-match['GoalDifference'])
#     return np.mean(goal_diffs[-n:]) if goal_diffs else 0

# df['Home Team Performance'] = df['Home Team'].apply(team_performance)
# df['Away Team Performance'] = df['Away Team'].apply(team_performance)

# # Prepare the features (X) and target variable (y)
# features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded', 
#             'Location Encoded', 'DayOfWeek', 'Month', 'Hour', 
#             'Home Team Performance', 'Away Team Performance']
# X = df[features]
# y = df['GoalDifference']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train multiple models
# models = {
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
# }

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
    
#     print(f"\n{name} Results:")
#     print(f"Root Mean Squared Error: {rmse}")
#     print(f"R-squared Score: {r2}")
    
#     # Print feature importances
#     importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
#     for feature, importance in zip(features, importances):
#         print(f"{feature}: {importance}")

# # Use the best performing model for predictions
# best_model = max(models.items(), key=lambda x: r2_score(y_test, x[1].predict(X_test)))[1]

# # Prepare data for the next round of matches
# next_round = df['Round Number'].max() + 1
# next_matches = df[df['Round Number'] == next_round].copy()

# # If next_matches is empty, use the last available round
# if next_matches.empty:
#     next_round = df['Round Number'].max()
#     next_matches = df[df['Round Number'] == next_round].copy()

# # Prepare features for prediction
# X_next = next_matches[features]

# # Make predictions
# predictions = best_model.predict(X_next)

# # Add predictions to the next_matches DataFrame
# next_matches['Predicted_Goal_Difference'] = predictions

# # Calculate predicted scores (this is a simplification)
# next_matches['Predicted_Home_Goals'] = np.round((predictions + 2) / 2).astype(int)
# next_matches['Predicted_Away_Goals'] = np.round((2 - predictions) / 2).astype(int)

# # Ensure non-negative scores
# next_matches['Predicted_Home_Goals'] = next_matches['Predicted_Home_Goals'].clip(lower=0)
# next_matches['Predicted_Away_Goals'] = next_matches['Predicted_Away_Goals'].clip(lower=0)

# # Create reverse mappings for decoding
# team_decoding = {v: k for k, v in team_encoding.items()}
# location_decoding = {v: k for k, v in location_encoding.items()}

# # Display predictions
# print(f"\nPredictions for Round {next_round}:")
# for _, match in next_matches.iterrows():
#     home_team_name = team_decoding[match['Home Team Encoded']]
#     away_team_name = team_decoding[match['Away Team Encoded']]
#     print(f"{home_team_name} vs {away_team_name}: {match['Predicted_Home_Goals']}-{match['Predicted_Away_Goals']}")

# # Optionally, save predictions to a CSV file
# next_matches[['Home Team', 'Away Team', 'Predicted_Home_Goals', 'Predicted_Away_Goals']].to_csv(f'predictions_round_{next_round}.csv', index=False)





# NEXT MATCH PREDICTION FOR FIRST ONE


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the data
df = pd.read_csv('/home/nelson/Desktop/matches/Data/LaLiga.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# Get the latest round with actual results
latest_round_with_results = df.dropna(subset=['Result'])['Round Number'].max()
print(f"Latest round with results: {latest_round_with_results}")

# Create dictionaries for team and location encoding
def create_encoding_dict(column):
    unique_values = df[column].unique()
    return {value: i for i, value in enumerate(unique_values)}

team_encoding = create_encoding_dict('Home Team')
location_encoding = create_encoding_dict('Location')

# Encoding function that handles new values
def encode_column(value, encoding_dict):
    if value not in encoding_dict:
        encoding_dict[value] = len(encoding_dict)
    return encoding_dict[value]

# Encode teams and locations
df['Home Team Encoded'] = df['Home Team'].apply(lambda x: encode_column(x, team_encoding))
df['Away Team Encoded'] = df['Away Team'].apply(lambda x: encode_column(x, team_encoding))
df['Location Encoded'] = df['Location'].apply(lambda x: encode_column(x, location_encoding))

# Extract features from 'Date'
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour

# Calculate goal difference
def calculate_goal_difference(result):
    if isinstance(result, str) and ' - ' in result:
        home_goals, away_goals = result.split(' - ')
        return int(home_goals) - int(away_goals)
    return None

df['GoalDifference'] = df['Result'].apply(calculate_goal_difference)

# Filter data for training to include only matches with results
df_training = df.dropna(subset=['GoalDifference'])

# Add new features: team's average goal difference in last 3 matches
def team_performance(team, n=3):
    team_matches = df_training[(df_training['Home Team'] == team) | (df_training['Away Team'] == team)].sort_values('Date')
    goal_diffs = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            goal_diffs.append(match['GoalDifference'])
        else:
            goal_diffs.append(-match['GoalDifference'])
    return np.mean(goal_diffs[-n:]) if goal_diffs else 0

df_training['Home Team Performance'] = df_training['Home Team'].apply(team_performance)
df_training['Away Team Performance'] = df_training['Away Team'].apply(team_performance)

# Prepare features
features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded', 
            'Location Encoded', 'DayOfWeek', 'Month', 'Hour', 
            'Home Team Performance', 'Away Team Performance']

X = df_training[features]
y = df_training['GoalDifference']

# Train the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")

# Select best model
best_model = max(models.items(), key=lambda x: r2_score(y_test, x[1].predict(X_test)))[1]

# Get next round matches
next_round = latest_round_with_results + 1
next_matches = df[df['Round Number'] == next_round].copy()

if next_matches.empty:
    print(f"No fixture data available for Round {next_round}")
else:
    # Calculate performance metrics for next round matches
    next_matches['Home Team Performance'] = next_matches['Home Team'].apply(team_performance)
    next_matches['Away Team Performance'] = next_matches['Away Team'].apply(team_performance)
    
    # Prepare features for prediction
    X_next = next_matches[features]
    
    # Make predictions
    predictions = best_model.predict(X_next)
    
    # Calculate predicted scores
    next_matches['Predicted_Goal_Difference'] = predictions
    next_matches['Predicted_Home_Goals'] = np.round((predictions + 2) / 2).astype(int)
    next_matches['Predicted_Away_Goals'] = np.round((2 - predictions) / 2).astype(int)
    
    # Ensure non-negative scores
    next_matches['Predicted_Home_Goals'] = next_matches['Predicted_Home_Goals'].clip(lower=0)
    next_matches['Predicted_Away_Goals'] = next_matches['Predicted_Away_Goals'].clip(lower=0)
    
    # Display predictions
    print(f"\nPredictions for Round {next_round}:")
    for _, match in next_matches.iterrows():
        print(f"{match['Home Team']} vs {match['Away Team']}: {match['Predicted_Home_Goals']}-{match['Predicted_Away_Goals']}")
    
    # Save predictions
    output_file = f'predictions_round_{next_round}.csv'
    next_matches[['Home Team', 'Away Team', 'Predicted_Home_Goals', 'Predicted_Away_Goals']].to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")











# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Load the data
# df = pd.read_csv('Data/EPL.csv')

# # Convert 'Date' to datetime
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# # Create dictionaries for team and location encoding
# def create_encoding_dict(column):
#     unique_values = df[column].unique()
#     return {value: i for i, value in enumerate(unique_values)}

# team_encoding = create_encoding_dict('Home Team')
# location_encoding = create_encoding_dict('Location')

# # Encoding function that handles new values
# def encode_column(value, encoding_dict):
#     if value not in encoding_dict:
#         encoding_dict[value] = len(encoding_dict)
#     return encoding_dict[value]

# # Encode teams and locations
# df['Home Team Encoded'] = df['Home Team'].apply(lambda x: encode_column(x, team_encoding))
# df['Away Team Encoded'] = df['Away Team'].apply(lambda x: encode_column(x, team_encoding))
# df['Location Encoded'] = df['Location'].apply(lambda x: encode_column(x, location_encoding))

# # Extract features from 'Date'
# df['DayOfWeek'] = df['Date'].dt.dayofweek
# df['Month'] = df['Date'].dt.month
# df['Hour'] = df['Date'].dt.hour

# # Calculate goal difference
# def calculate_goal_difference(result):
#     if isinstance(result, str) and ' - ' in result:
#         home_goals, away_goals = result.split(' - ')
#         return int(home_goals) - int(away_goals)
#     return None

# df['GoalDifference'] = df['Result'].apply(calculate_goal_difference)

# # Get the latest round with actual results
# latest_round_with_results = df.dropna(subset=['Result'])['Round Number'].max()
# print(f"Latest round with results: {latest_round_with_results}")

# # Filter data for training to include only matches with results
# df_training = df.dropna(subset=['GoalDifference'])

# # Add new features: team's average goal difference in last 3 matches
# def team_performance(team, n=3):
#     team_matches = df_training[(df_training['Home Team'] == team) | (df_training['Away Team'] == team)].sort_values('Date')
#     goal_diffs = []
#     for _, match in team_matches.iterrows():
#         if match['Home Team'] == team:
#             goal_diffs.append(match['GoalDifference'])
#         else:
#             goal_diffs.append(-match['GoalDifference'])
#     return np.mean(goal_diffs[-n:]) if goal_diffs else 0

# df_training['Home Team Performance'] = df_training['Home Team'].apply(team_performance)
# df_training['Away Team Performance'] = df_training['Away Team'].apply(team_performance)

# # Prepare features
# features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded', 
#             'Location Encoded', 'DayOfWeek', 'Month', 'Hour', 
#             'Home Team Performance', 'Away Team Performance']

# X = df_training[features]
# y = df_training['GoalDifference']

# # Train the models
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models = {
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
# }

# # Train and evaluate models
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
    
#     print(f"\n{name} Results:")
#     print(f"Root Mean Squared Error: {rmse}")
#     print(f"R-squared Score: {r2}")

# # Select best model
# best_model = max(models.items(), key=lambda x: r2_score(y_test, x[1].predict(X_test)))[1]

# # Get next round matches
# next_round = latest_round_with_results + 1
# next_matches = df[df['Round Number'] == next_round].copy()

# if next_matches.empty:
#     print(f"No fixture data available for Round {next_round}")
# else:
#     # Calculate performance metrics for next round matches
#     next_matches['Home Team Performance'] = next_matches['Home Team'].apply(team_performance)
#     next_matches['Away Team Performance'] = next_matches['Away Team'].apply(team_performance)
    
#     # Prepare features for prediction
#     X_next = next_matches[features]
    
#     # Make predictions
#     predictions = best_model.predict(X_next)
    
#     # Calculate predicted scores
#     next_matches['Predicted_Goal_Difference'] = predictions
#     next_matches['Predicted_Home_Goals'] = np.round((predictions + 2) / 2).astype(int)
#     next_matches['Predicted_Away_Goals'] = np.round((2 - predictions) / 2).astype(int)
    
#     # Ensure non-negative scores
#     next_matches['Predicted_Home_Goals'] = next_matches['Predicted_Home_Goals'].clip(lower=0)
#     next_matches['Predicted_Away_Goals'] = next_matches['Predicted_Away_Goals'].clip(lower=0)
    
#     # Display predictions
#     print(f"\nPredictions for Round {next_round}:")
#     for _, match in next_matches.iterrows():
#         print(f"{match['Home Team']} vs {match['Away Team']}: {match['Predicted_Home_Goals']}-{match['Predicted_Away_Goals']}")
    
#     # Save predictions
#     output_file = f'predictions_round_{next_round}.csv'
#     next_matches[['Home Team', 'Away Team', 'Predicted_Home_Goals', 'Predicted_Away_Goals']].to_csv(output_file, index=False)
#     print(f"\nPredictions saved to {output_file}")

