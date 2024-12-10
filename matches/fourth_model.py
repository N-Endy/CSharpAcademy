import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv('Data/Ligue1.csv')

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
    'Cat Boost': CatBoostRegressor(**cat_grid_home.best_params_),
    'XGBoost': XGBRegressor(**xgb_grid_home.best_params_),
    'Ada Boost': AdaBoostRegressor(**ada_grid_home.best_params_),
    'Extra Trees': ExtraTreesRegressor(**et_grid_home.best_params_)
}

models_away = {
    'Random Forest': RandomForestRegressor(**rf_grid_away.best_params_),
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
























# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from skopt import BayesSearchCV
# import joblib
# import logging
# import warnings
# import os

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def calculate_head_to_head(home_team, away_team, df_history):
#     matches = df_history[
#         ((df_history['Home Team'] == home_team) & (df_history['Away Team'] == away_team)) |
#         ((df_history['Home Team'] == away_team) & (df_history['Away Team'] == home_team))
#     ]
#     if len(matches) == 0:
#         return 0
    
#     results = matches['Result'].str.split(' - ', expand=True).astype(float)
#     return results.mean().mean()

# def calculate_recent_form(team, df_history, n_matches=5):
#     team_matches = df_history[
#         (df_history['Home Team'] == team) | 
#         (df_history['Away Team'] == team)
#     ].tail(n_matches)
    
#     if len(team_matches) == 0:
#         return 0
        
#     total_score = 0
#     match_count = 0
    
#     for _, match in team_matches.iterrows():
#         if pd.isna(match['Result']):
#             continue
            
#         home_goals, away_goals = map(int, match['Result'].split(' - '))
#         if match['Home Team'] == team:
#             total_score += home_goals
#         else:
#             total_score += away_goals
#         match_count += 1
    
#     return total_score / match_count if match_count > 0 else 0


# def calculate_elo_rating(team, matches, k_factor=32, initial_rating=1500):
#     if len(matches) == 0:
#         return initial_rating
    
#     rating = initial_rating
#     for _, match in matches.iterrows():
#         if match['Result'] is not None and isinstance(match['Result'], str):
#             if match['Home Team'] == team:
#                 home_goals, away_goals = map(int, match['Result'].split(' - '))
#                 result = 1 if home_goals > away_goals else 0 if home_goals < away_goals else 0.5
#                 rating += k_factor * result
#             elif match['Away Team'] == team:
#                 home_goals, away_goals = map(int, match['Result'].split(' - '))
#                 result = 1 if away_goals > home_goals else 0 if away_goals < home_goals else 0.5
#                 rating += k_factor * result
#     return rating

# def create_stacked_model():
#     estimators = [
#         ('rf', RandomForestRegressor(n_estimators=200, max_depth=20)),
#         ('xgb', XGBRegressor(n_estimators=200, max_depth=5)),
#         ('cat', CatBoostRegressor(iterations=200, depth=10, verbose=False))
#     ]
#     return StackingRegressor(
#         estimators=estimators,
#         final_estimator=RandomForestRegressor(n_estimators=100)
#     )

# def create_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, input_shape=input_shape, return_sequences=True),
#         LSTM(50),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def evaluate_model(y_true, y_pred):
#     return {
#         'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
#         'MAE': mean_absolute_error(y_true, y_pred),
#         'R2': r2_score(y_true, y_pred)
#     }



# def save_model(model, filename):
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(model, f'models/{filename}.pkl')

# def load_model(filename):
#     return joblib.load(f'models/{filename}.pkl')

# # Main execution
# if __name__ == "__main__":
#     # Load data
#     df = pd.read_csv('Data/NBA.csv')
    
#     # Convert date and create temporal features
#     df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
#     df['DayOfWeek'] = df['Date'].dt.dayofweek
#     df['Month'] = df['Date'].dt.month
#     df['Hour'] = df['Date'].dt.hour

#     # Encode categorical variables
#     le = LabelEncoder()
#     df['Home Team Encoded'] = le.fit_transform(df['Home Team'])
#     df['Away Team Encoded'] = le.fit_transform(df['Away Team'])
#     df['Location Encoded'] = le.fit_transform(df['Location'])

#     # Add enhanced features
#     for idx, row in df.iterrows():
#         df_history = df[df.index < idx]
#         df.at[idx, 'H2H_History'] = calculate_head_to_head(
#             row['Home Team'], 
#             row['Away Team'], 
#             df_history
#         )
#         df.at[idx, 'Home_Recent_Form'] = calculate_recent_form(
#             row['Home Team'], 
#             df_history
#         )
#         df.at[idx, 'Away_Recent_Form'] = calculate_recent_form(
#             row['Away Team'], 
#             df_history
#         )
#         df.at[idx, 'Home_ELO'] = calculate_elo_rating(
#             row['Home Team'], 
#             df_history
#         )
#         df.at[idx, 'Away_ELO'] = calculate_elo_rating(
#             row['Away Team'], 
#             df_history
#         )

#     # Prepare features
#     features = ['Match Number', 'Round Number', 'Home Team Encoded', 'Away Team Encoded',
#                 'Location Encoded', 'DayOfWeek', 'Month', 'Hour',
#                 'H2H_History', 'Home_Recent_Form', 'Away_Recent_Form',
#                 'Home_ELO', 'Away_ELO']

#     # Filter matches with results
#     df_with_results = df[df['Result'].notna()].copy()
    
#     # Prepare X and y
#     X = df_with_results[features]
#     y_home = df_with_results['Result'].str.split(' - ', expand=True)[0].astype(int)
#     y_away = df_with_results['Result'].str.split(' - ', expand=True)[1].astype(int)

#     # Split data
#     X_train, X_test, y_train_home, y_test_home = train_test_split(
#         X, y_home, test_size=0.2, random_state=42
#     )
#     _, _, y_train_away, y_test_away = train_test_split(
#         X, y_away, test_size=0.2, random_state=42
#     )

#     # Train stacked model
#     stacked_model_home = create_stacked_model()
#     stacked_model_away = create_stacked_model()

#     # Fit models
#     logger.info("Training home goals model...")
#     stacked_model_home.fit(X_train, y_train_home)
    
#     logger.info("Training away goals model...")
#     stacked_model_away.fit(X_train, y_train_away)

#     # Save models
#     save_model(stacked_model_home, 'stacked_model_home')
#     save_model(stacked_model_away, 'stacked_model_away')

#     # Make predictions
#     y_pred_home = stacked_model_home.predict(X_test)
#     y_pred_away = stacked_model_away.predict(X_test)

#     # Evaluate models
#     logger.info("Home Goals Model Performance:")
#     logger.info(evaluate_model(y_test_home, y_pred_home))
    
#     logger.info("Away Goals Model Performance:")
#     logger.info(evaluate_model(y_test_away, y_pred_away))

#     # Predict next matches
#     last_completed_round = df[df['Result'].notna()]['Round Number'].max()
#     next_round_matches = df[df['Round Number'] == last_completed_round + 1]

#     logger.info(f"\nPredictions for Round {last_completed_round + 1}:")
#     for _, match in next_round_matches.iterrows():
#         X_pred = pd.DataFrame([match[features]], columns=features)
#         home_goals = int(round(stacked_model_home.predict(X_pred)[0]))
#         away_goals = int(round(stacked_model_away.predict(X_pred)[0]))
#         logger.info(f"{match['Home Team']} vs {match['Away Team']}: {home_goals} - {away_goals}")