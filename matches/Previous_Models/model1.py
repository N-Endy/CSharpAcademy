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
# df = pd.read_csv('Data/EPL.csv')

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
#             'Location Encoded', 'DayOfWeek', 'Month', 'Hour',
#             'Home Team Performance', 'Away Team Performance']

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
#     X_pred = pd.DataFrame([match[features]], columns=features)
#     print(f"\n{match['Home Team']} vs {match['Away Team']}:")
#     for name in models_home.keys():
#         home_goals = int(round(models_home[name].predict(X_pred)[0]))
#         away_goals = int(round(models_away[name].predict(X_pred)[0]))
#         print(f"{name} predicts: {home_goals} - {away_goals}")









# FOURTH MODEL: RESULTS ARE TOO PERFECT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import datetime

# Load and preprocess data
df = pd.read_csv('/home/nelson/Desktop/matches/Data/LaLiga.csv')

# Convert date strings to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# Extract additional features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour

# Encode categorical variables
le = LabelEncoder()
df['Home Team Encoded'] = le.fit_transform(df['Home Team'])
df['Away Team Encoded'] = le.fit_transform(df['Away Team'])
df['Location Encoded'] = le.fit_transform(df['Location'])

# Calculate team performance metrics
def calculate_team_performance(team, is_home=True):
    team_matches = df[df['Home Team' if is_home else 'Away Team'] == team]
    if len(team_matches) == 0:
        return 0
    
    team_matches = team_matches[team_matches['Result'].notna()]
    
    if len(team_matches) == 0:
        return 0
        
    results = team_matches['Result'].str.split(' - ', expand=True).astype(int)
    goals_scored = results[0] if is_home else results[1]
    return goals_scored.mean()

# Add team performance features
teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
for team in teams:
    print(f"Processing team: {team}")
    mask_home = df['Home Team'] == team
    mask_away = df['Away Team'] == team
    df.loc[mask_home, 'Home Team Performance'] = df[mask_home].apply(
        lambda x: calculate_team_performance(team, True), axis=1
    )
    df.loc[mask_away, 'Away Team Performance'] = df[mask_away].apply(
        lambda x: calculate_team_performance(team, False), axis=1
    )

# Filter matches with results for training
df_with_results = df[df['Result'].notna()].copy()

# Prepare features and target
features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded',
            'Location Encoded', 'DayOfWeek', 'Month', 'Hour',
            'Home Team Performance', 'Away Team Performance']

X = df_with_results[features]
y_home = df_with_results['Result'].str.split(' - ', expand=True)[0].astype(int)
y_away = df_with_results['Result'].str.split(' - ', expand=True)[1].astype(int)

# Split the data
X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Define separate parameter grids for each model type
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# lgb_params = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.05, 0.1],
#     'max_depth': [5, 10],
#     'num_leaves': [31, 50],
#     'min_child_samples': [20, 30]
# }

ada_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
}

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

cat_params = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],
    'depth': [5, 10]
}

et_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Train models for home goals
print("Training models for home goals...")
rf_grid_home = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
rf_grid_home.fit(X_train, y_train_home)

# lgb_grid_home = GridSearchCV(LGBMRegressor(), lgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# lgb_grid_home.fit(X_train, y_train_home)

cat_grid_home = GridSearchCV(CatBoostRegressor(), cat_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
cat_grid_home.fit(X_train, y_train_home)

xgb_grid_home = GridSearchCV(XGBRegressor(), xgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
xgb_grid_home.fit(X_train, y_train_home)

ada_grid_home = GridSearchCV(AdaBoostRegressor(), ada_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
ada_grid_home.fit(X_train, y_train_home)

et_grid_home = GridSearchCV(ExtraTreesRegressor(), et_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
et_grid_home.fit(X_train, y_train_home)

# Train models for away goals
print("\nTraining models for away goals...")
rf_grid_away = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
rf_grid_away.fit(X_train, y_train_away)

# lgb_grid_away = GridSearchCV(LGBMRegressor(), lgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# lgb_grid_away.fit(X_train, y_train_away)

cat_grid_away = GridSearchCV(CatBoostRegressor(), cat_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
cat_grid_away.fit(X_train, y_train_away)

xgb_grid_away = GridSearchCV(XGBRegressor(), xgb_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
xgb_grid_away.fit(X_train, y_train_away)

ada_grid_away = GridSearchCV(AdaBoostRegressor(), ada_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
ada_grid_away.fit(X_train, y_train_away)

et_grid_away = GridSearchCV(ExtraTreesRegressor(), et_params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
et_grid_away.fit(X_train, y_train_away)

# Create optimized models with correct best parameters
models_home = {
    'Random Forest': RandomForestRegressor(**rf_grid_home.best_params_),
    # 'LightGBM': LGBMRegressor(**lgb_grid_home.best_params_),
    'Cat Boost': CatBoostRegressor(**cat_grid_home.best_params_),
    'XGBoost': XGBRegressor(**xgb_grid_home.best_params_),
    'Ada Boost': AdaBoostRegressor(**ada_grid_home.best_params_),
    'Extra Trees': ExtraTreesRegressor(**et_grid_home.best_params_)
}

models_away = {
    'Random Forest': RandomForestRegressor(**rf_grid_away.best_params_),
    # 'LightGBM': LGBMRegressor(**lgb_grid_away.best_params_),
    'Cat Boost': CatBoostRegressor(**cat_grid_away.best_params_),
    'XGBoost': XGBRegressor(**xgb_grid_away.best_params_),
    'Ada Boost': AdaBoostRegressor(**ada_grid_away.best_params_),
    'Extra Trees': ExtraTreesRegressor(**et_grid_away.best_params_)
}

# Train final models
for name, model in models_home.items():
    print(f"\nTraining {name} for home goals...")
    model.fit(X_train, y_train_home)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_home, y_pred))
    r2 = r2_score(y_test_home, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

for name, model in models_away.items():
    print(f"\nTraining {name} for away goals...")
    model.fit(X_train, y_train_away)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_away, y_pred))
    r2 = r2_score(y_test_away, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Find the last completed round
last_completed_round = df[df['Result'].notna()]['Round Number'].max()

# Get matches for the next round
next_round_matches = df[df['Round Number'] == last_completed_round + 1]

print(f"\nPredictions for Round {last_completed_round + 1}:")
for _, match in next_round_matches.iterrows():
    X_pred = pd.DataFrame([match[features]], columns=features)
    print(f"\n{match['Home Team']} vs {match['Away Team']}:")
    for name in models_home.keys():
        home_goals = int(round(models_home[name].predict(X_pred)[0]))
        away_goals = int(round(models_away[name].predict(X_pred)[0]))
        print(f"{name} predicts: {home_goals} - {away_goals}")









# EXPRESSIVE MODEL: LOW ACCURACY

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
# from catboost import CatBoostRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from lightgbm import LGBMRegressor
# import datetime

# # Load and preprocess data
# df = pd.read_csv('Data/Ligue1.csv')
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# # Extract time-based features
# df['DayOfWeek'] = df['Date'].dt.dayofweek
# df['Month'] = df['Date'].dt.month
# df['Hour'] = df['Date'].dt.hour

# # Encode categorical variables
# le = LabelEncoder()
# df['Home Team Encoded'] = le.fit_transform(df['Home Team'])
# df['Away Team Encoded'] = le.fit_transform(df['Away Team'])
# df['Location Encoded'] = le.fit_transform(df['Location'])

# # Enhanced team performance calculation with temporal aspect
# def calculate_team_performance(team, date, is_home=True):
#     historical_matches = df[
#         (df['Date'] < date) & 
#         (df[('Home Team' if is_home else 'Away Team')] == team) &
#         (df['Result'].notna())
#     ]
#     if len(historical_matches) == 0:
#         return 0
        
#     results = historical_matches['Result'].str.split(' - ', expand=True).astype(int)
#     goals_scored = results[0] if is_home else results[1]
    
#     # Weight recent matches more heavily
#     days_diff = (date - historical_matches['Date']).dt.days
#     weights = np.exp(-days_diff / 365)  # Exponential decay over a year
#     return np.average(goals_scored, weights=weights)

# # Calculate performance features
# teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
# for team in teams:
#     print(f"Processing team: {team}")
#     for idx, row in df.iterrows():
#         if row['Home Team'] == team:
#             df.loc[idx, 'Home Team Performance'] = calculate_team_performance(team, row['Date'], True)
#         if row['Away Team'] == team:
#             df.loc[idx, 'Away Team Performance'] = calculate_team_performance(team, row['Date'], False)

# # Prepare features and targets
# features = [
#     'Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded',
#     'Location Encoded', 'DayOfWeek', 'Month', 'Hour',
#     'Home Team Performance', 'Away Team Performance'
# ]

# df_with_results = df[df['Result'].notna()].copy()
# df_with_results = df_with_results.sort_values('Date')

# # Time-based split
# train_size = int(len(df_with_results) * 0.8)
# X_train = df_with_results[features][:train_size]
# X_test = df_with_results[features][train_size:]
# y_train_home = df_with_results['Result'].str.split(' - ', expand=True)[0].astype(int)[:train_size]
# y_test_home = df_with_results['Result'].str.split(' - ', expand=True)[0].astype(int)[train_size:]
# y_train_away = df_with_results['Result'].str.split(' - ', expand=True)[1].astype(int)[:train_size]
# y_test_away = df_with_results['Result'].str.split(' - ', expand=True)[1].astype(int)[train_size:]

# # Enhanced parameter grids with regularization
# rf_params = {
#     'n_estimators': [200, 300],
#     'max_depth': [10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['sqrt', 'log2'],
#     'bootstrap': [True]
# }

# # lgb_params = {
# #     'n_estimators': [200, 300],
# #     'learning_rate': [0.05, 0.1],
# #     'max_depth': [5, 10],
# #     'num_leaves': [31, 50],
# #     'min_child_samples': [20, 30],
# #     'reg_alpha': [0, 0.1],
# #     'reg_lambda': [0, 1.0]
# # }

# ada_params = {
#     'n_estimators': [50, 100],
#     'learning_rate': [0.05, 0.1],
#     'loss': ['linear', 'square', 'exponential']
# }

# xgb_params = {
#     'n_estimators': [200, 300],
#     'max_depth': [3, 5],
#     'learning_rate': [0.05, 0.1],
#     'reg_alpha': [0, 0.1],
#     'reg_lambda': [0, 1.0],
#     'subsample': [0.8, 1.0]
# }

# cat_params = {
#     'iterations': [200, 300],
#     'learning_rate': [0.05, 0.1],
#     'depth': [5, 10],
#     'l2_leaf_reg': [1, 3],
#     'bootstrap_type': ['Bayesian'],
#     'random_strength': [1, 3]
# }

# et_params = {
#     'n_estimators': [200, 300],
#     'max_depth': [10, 20],
#     'min_samples_split': [2, 5],
#     'max_features': ['sqrt', 'log2'],
#     'bootstrap': [True]
# }

# # Time series cross-validation
# tscv = TimeSeriesSplit(n_splits=5)

# # Train models with time series cross-validation
# models_home = {}
# models_away = {}

# for name, (model, params) in [
#     ('Random Forest', (RandomForestRegressor(), rf_params)),
#     # ('LightGBM', (LGBMRegressor(), lgb_params)),
#     ('CatBoost', (CatBoostRegressor(), cat_params)),
#     ('XGBoost', (XGBRegressor(), xgb_params)),
#     ('AdaBoost', (AdaBoostRegressor(), ada_params)),
#     ('Extra Trees', (ExtraTreesRegressor(), et_params))
# ]:
#     print(f"\nTraining {name}...")
    
#     # Home goals
#     grid_search_home = GridSearchCV(
#         model, params, cv=tscv, scoring='neg_mean_squared_error',
#         verbose=1, n_jobs=-1
#     )
#     grid_search_home.fit(X_train, y_train_home)
#     models_home[name] = grid_search_home.best_estimator_
    
#     # Away goals
#     grid_search_away = GridSearchCV(
#         model, params, cv=tscv, scoring='neg_mean_squared_error',
#         verbose=1, n_jobs=-1
#     )
#     grid_search_away.fit(X_train, y_train_away)
#     models_away[name] = grid_search_away.best_estimator_

# # Evaluate models
# for name in models_home.keys():
#     print(f"\nEvaluating {name}:")
    
#     # Home goals
#     y_pred_home = models_home[name].predict(X_test)
#     rmse_home = np.sqrt(mean_squared_error(y_test_home, y_pred_home))
#     r2_home = r2_score(y_test_home, y_pred_home)
#     print(f"Home Goals - RMSE: {rmse_home:.4f}, R2: {r2_home:.4f}")
    
#     # Away goals
#     y_pred_away = models_away[name].predict(X_test)
#     rmse_away = np.sqrt(mean_squared_error(y_test_away, y_pred_away))
#     r2_away = r2_score(y_test_away, y_pred_away)
#     print(f"Away Goals - RMSE: {rmse_away:.4f}, R2: {r2_away:.4f}")

# # Predict next round
# last_completed_round = df[df['Result'].notna()]['Round Number'].max()
# next_round_matches = df[df['Round Number'] == last_completed_round + 1]

# print(f"\nPredictions for Round {last_completed_round + 1}:")
# for _, match in next_round_matches.iterrows():
#     X_pred = pd.DataFrame([match[features]], columns=features)
#     print(f"\n{match['Home Team']} vs {match['Away Team']}:")
#     for name in models_home.keys():
#         home_goals = int(round(models_home[name].predict(X_pred)[0]))
#         away_goals = int(round(models_away[name].predict(X_pred)[0]))
#         print(f"{name} predicts: {home_goals} - {away_goals}")
