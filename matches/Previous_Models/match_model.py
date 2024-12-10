# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Load the data
# df = pd.read_csv('Data/UCL.csv')

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

# # Prepare the features (X) and target variable (y)
# features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded', 'Location Encoded', 'DayOfWeek', 'Month', 'Hour']
# X = df[features]

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
# X = df[features]
# y = df['GoalDifference']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Print feature importances
# for feature, importance in zip(features, model.coef_):
#     print(f"{feature}: {importance}")

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
# predictions = model.predict(X_next)

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




# # SECOND MODEL

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
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

# # Create and train the model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Print feature importances
# for feature, importance in zip(features, model.feature_importances_):
#     print(f"{feature}: {importance}")

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
# predictions = model.predict(X_next)

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
















# # THIRD MODEL

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import numpy as np

# # Load the data
# df = pd.read_csv('Data/LaLiga.csv')

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
#     'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
#     'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
#     'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
#     'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
#     'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
#     'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
#     'Support Vector Machine': SVR(kernel='rbf')
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
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         importances = model.coef_
#     elif hasattr(model, 'coefs_'):
#         importances = model.coefs_[0]
#     else:
#         importances = np.zeros(len(features))  # Default if no importance available

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





# FOURTH MODEL

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from lightgbm import LGBMRegressor
# import datetime

# # Load and preprocess data
# df = pd.read_csv('EPL.csv')

# # Convert date strings to datetime objects
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# # Extract additional features
# df['DayOfWeek'] = df['Date'].dt.dayofweek
# df['Month'] = df['Date'].dt.month
# df['Hour'] = df['Date'].dt.hour

# # Encode categorical variables
# le = LabelEncoder()
# df['Home Team Encoded'] = le.fit_transform(df['Home Team'])
# df['Away Team Encoded'] = le.fit_transform(df['Away Team'])
# df['Location Encoded'] = le.fit_transform(df['Location'])

# # Calculate team performance metrics
# def calculate_team_performance(team, is_home=True):
#     team_matches = df[df['Home Team' if is_home else 'Away Team'] == team]
#     if len(team_matches) == 0:
#         return 0
    
#     team_matches = team_matches[team_matches['Result'].notna()]
    
#     if len(team_matches) == 0:
#         return 0
        
#     results = team_matches['Result'].str.split(' - ', expand=True).astype(int)
#     goals_scored = results[0] if is_home else results[1]
#     return goals_scored.mean()

# # Add team performance features
# teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
# for team in teams:
#     print(f"Processing team: {team}")
#     mask_home = df['Home Team'] == team
#     mask_away = df['Away Team'] == team
#     df.loc[mask_home, 'Home Team Performance'] = df[mask_home].apply(
#         lambda x: calculate_team_performance(team, True), axis=1
#     )
#     df.loc[mask_away, 'Away Team Performance'] = df[mask_away].apply(
#         lambda x: calculate_team_performance(team, False), axis=1
#     )

# # Filter matches with results for training
# df_with_results = df[df['Result'].notna()].copy()

# # Prepare features and target
# features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded',
#            'Location Encoded', 'DayOfWeek', 'Month', 'Hour',
#            'Home Team Performance', 'Away Team Performance']

# X = df_with_results[features]
# y_home = df_with_results['Result'].str.split(' - ', expand=True)[0].astype(int)
# y_away = df_with_results['Result'].str.split(' - ', expand=True)[1].astype(int)

# # Split the data
# X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42)
# _, _, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42)

# # Focused parameter grids
# rf_params = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# lgb_params = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.05, 0.1],
#     'max_depth': [5, 10],
#     'num_leaves': [31, 50],
#     'min_child_samples': [20, 30]
# }

# # Train models for home goals
# print("Training models for home goals...")
# rf_grid_home = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# rf_grid_home.fit(X_train, y_train_home)

# lgb_grid_home = GridSearchCV(LGBMRegressor(), lgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# lgb_grid_home.fit(X_train, y_train_home)

# # Train models for away goals
# print("\nTraining models for away goals...")
# rf_grid_away = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# rf_grid_away.fit(X_train, y_train_away)

# lgb_grid_away = GridSearchCV(LGBMRegressor(), lgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# lgb_grid_away.fit(X_train, y_train_away)

# # Create optimized models
# models_home = {
#     'Random Forest': RandomForestRegressor(**rf_grid_home.best_params_),
#     'LightGBM': LGBMRegressor(**lgb_grid_home.best_params_)
# }

# models_away = {
#     'Random Forest': RandomForestRegressor(**rf_grid_away.best_params_),
#     'LightGBM': LGBMRegressor(**lgb_grid_away.best_params_)
# }

# # Train final models
# for name, model in models_home.items():
#     print(f"\nTraining {name} for home goals...")
#     model.fit(X_train, y_train_home)
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test_home, y_pred))
#     r2 = r2_score(y_test_home, y_pred)
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R2 Score: {r2:.4f}")

# for name, model in models_away.items():
#     print(f"\nTraining {name} for away goals...")
#     model.fit(X_train, y_train_away)
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test_away, y_pred))
#     r2 = r2_score(y_test_away, y_pred)
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R2 Score: {r2:.4f}")

# # Find the last completed round
# last_completed_round = df[df['Result'].notna()]['Round Number'].max()

# # Get matches for the next round
# next_round_matches = df[df['Round Number'] == last_completed_round + 1]

# print(f"\nPredictions for Round {last_completed_round + 1}:")
# for _, match in next_round_matches.iterrows():
#     X_pred = match[features].values.reshape(1, -1)
#     print(f"\n{match['Home Team']} vs {match['Away Team']}:")
#     for name in models_home.keys():
#         home_goals = int(round(models_home[name].predict(X_pred)[0]))
#         away_goals = int(round(models_away[name].predict(X_pred)[0]))
#         print(f"{name} predicts: {home_goals} - {away_goals}")








# INITIAL ONE


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the data
df = pd.read_csv('Data/EPL.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

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
    else:
        return None

df['GoalDifference'] = df['Result'].apply(calculate_goal_difference)

# Remove rows with None values in GoalDifference
df = df.dropna(subset=['GoalDifference'])

# Add new features: team's average goal difference in last 3 matches
def team_performance(team, n=3):
    team_matches = df[(df['Home Team'] == team) | (df['Away Team'] == team)].sort_values('Date')
    goal_diffs = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            goal_diffs.append(match['GoalDifference'])
        else:
            goal_diffs.append(-match['GoalDifference'])
    return np.mean(goal_diffs[-n:]) if goal_diffs else 0

df['Home Team Performance'] = df['Home Team'].apply(team_performance)
df['Away Team Performance'] = df['Away Team'].apply(team_performance)

# Prepare the features (X) and target variable (y)
features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded', 
            'Location Encoded', 'DayOfWeek', 'Month', 'Hour', 
            'Home Team Performance', 'Away Team Performance']
X = df[features]
y = df['GoalDifference']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")
    
    # Print feature importances
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
    for feature, importance in zip(features, importances):
        print(f"{feature}: {importance}")

# Use the best performing model for predictions
best_model = max(models.items(), key=lambda x: r2_score(y_test, x[1].predict(X_test)))[1]

# Prepare data for the next round of matches
next_round = df['Round Number'].max() + 1
next_matches = df[df['Round Number'] == next_round].copy()

# If next_matches is empty, use the last available round
if next_matches.empty:
    next_round = df['Round Number'].max()
    next_matches = df[df['Round Number'] == next_round].copy()

# Prepare features for prediction
X_next = next_matches[features]

# Make predictions
predictions = best_model.predict(X_next)

# Add predictions to the next_matches DataFrame
next_matches['Predicted_Goal_Difference'] = predictions

# Calculate predicted scores (this is a simplification)
next_matches['Predicted_Home_Goals'] = np.round((predictions + 2) / 2).astype(int)
next_matches['Predicted_Away_Goals'] = np.round((2 - predictions) / 2).astype(int)

# Ensure non-negative scores
next_matches['Predicted_Home_Goals'] = next_matches['Predicted_Home_Goals'].clip(lower=0)
next_matches['Predicted_Away_Goals'] = next_matches['Predicted_Away_Goals'].clip(lower=0)

# Create reverse mappings for decoding
team_decoding = {v: k for k, v in team_encoding.items()}
location_decoding = {v: k for k, v in location_encoding.items()}

# Display predictions
print(f"\nPredictions for Round {next_round}:")
for _, match in next_matches.iterrows():
    home_team_name = team_decoding[match['Home Team Encoded']]
    away_team_name = team_decoding[match['Away Team Encoded']]
    print(f"{home_team_name} vs {away_team_name}: {match['Predicted_Home_Goals']}-{match['Predicted_Away_Goals']}")

# Optionally, save predictions to a CSV file
next_matches[['Home Team', 'Away Team', 'Predicted_Home_Goals', 'Predicted_Away_Goals']].to_csv(f'predictions_round_{next_round}.csv', index=False)
