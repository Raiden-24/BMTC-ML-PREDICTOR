import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor, Pool
import json
from math import radians, sin, cos, atan2, sqrt

# --- Haversine Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

# --- Data Processing Function ---
def process_data(df_raw):
    df_raw.rename(columns={
        'departure_from_origin': 'Start_time_raw',
        'arrival_at_destination': 'Reach_time_raw',
        'departure_from_destination': 'Return_trip_start_time_raw',
        'arrival_at_origin': 'Return_trip_reach_time_raw'
    }, inplace=True)

    expanded_data = []
    for _, row in df_raw.iterrows():
        route_no = row['route_no']
        distance_km = float(row['distance'].replace(" KM", "").strip())
        origin = row['origin']
        destination = row['destination']
        try:
            stops = json.loads(row['map_json_content'])
            num_stops = len(stops)
            recalculated_length = sum(
                haversine_distance(
                    float(stops[i]['latlons'][0]), float(stops[i]['latlons'][1]),
                    float(stops[i+1]['latlons'][0]), float(stops[i+1]['latlons'][1])
                ) for i in range(num_stops - 1)
            ) if num_stops > 1 else 0
            avg_stop_dist = recalculated_length / (num_stops - 1) if num_stops > 1 else 0
        except:
            num_stops = recalculated_length = avg_stop_dist = np.nan

        start_times = row['Start_time_raw'].split(',')
        reach_times = row['Reach_time_raw'].split(',')

        for i in range(min(len(start_times), len(reach_times))):
            expanded_data.append({
                'route_no': route_no,
                'distance_km': distance_km,
                'origin': origin,
                'destination': destination,
                'Start_time': start_times[i].strip(),
                'Reach_time': reach_times[i].strip(),
                'number_of_bus_stops': num_stops,
                'recalculated_route_length': recalculated_length,
                'average_distance_between_stops': avg_stop_dist
            })

    df = pd.DataFrame(expanded_data)
    df['Start_time'] = pd.to_datetime(df['Start_time'], format="%H:%M", errors='coerce')
    df['Reach_time'] = pd.to_datetime(df['Reach_time'], format="%H:%M", errors='coerce')
    df['Journey_Duration_Minutes'] = (df['Reach_time'] - df['Start_time']).dt.total_seconds() / 60
    df.loc[df['Journey_Duration_Minutes'] < 0, 'Journey_Duration_Minutes'] += 1440

    df['Start_hour'] = df['Start_time'].dt.hour
    df['minute_of_day'] = df['Start_time'].dt.hour * 60 + df['Start_time'].dt.minute
    df['day_of_week'] = df['Start_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['Start_hour_sin'] = np.sin(2 * np.pi * df['Start_hour'] / 24)
    df['Start_hour_cos'] = np.cos(2 * np.pi * df['Start_hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['time_bin'] = df['Start_hour'].apply(lambda h: (
        'Morning Rush' if 5 <= h < 10 else
        'Midday' if 10 <= h < 16 else
        'Evening Rush' if 16 <= h < 20 else
        'Off Hours'))

    required_cols = ['distance_km', 'number_of_bus_stops', 'recalculated_route_length',
                     'average_distance_between_stops', 'minute_of_day', 'Start_hour_sin',
                     'Start_hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                     'route_no', 'origin', 'destination', 'time_bin', 'is_weekend']
    df = df.dropna(subset=required_cols)

    df['Journey_Duration_Minutes_Original'] = df['Journey_Duration_Minutes']
    df['Journey_Duration_Minutes'] = np.log1p(df['Journey_Duration_Minutes'])
    return df

# --- Load and Process Data ---
df_raw = pd.read_csv("C://Users//Amruth//Desktop//ML model project final//bmtc_own_dataset.csv")
df = process_data(df_raw.copy())

# --- Define Features ---
numerical_features = ['distance_km', 'number_of_bus_stops', 'recalculated_route_length',
                      'average_distance_between_stops', 'minute_of_day',
                      'Start_hour_sin', 'Start_hour_cos', 'day_of_week_sin', 'day_of_week_cos']
categorical_features = ['route_no', 'origin', 'destination', 'time_bin', 'is_weekend']

X = df[numerical_features + categorical_features]
y = df['Journey_Duration_Minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Categorical Indices for CatBoost ---
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]
# Before train_test_split

X = df[numerical_features + categorical_features]

# Encode categorical variables into numeric using one-hot
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Then proceed with model training

# --- Models Dictionary ---
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0)
}

# --- Model Evaluation ---
results = {'Model': [], 'R2_Score': [], 'MAE': [], 'MSE': [], 'RMSE': []}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predictions = np.expm1(predictions)
    actuals = np.expm1(y_test)

    results['Model'].append(name)
    results['R2_Score'].append(r2_score(actuals, predictions))
    results['MAE'].append(mean_absolute_error(actuals, predictions))
    results['MSE'].append(mean_squared_error(actuals, predictions))
    results['RMSE'].append(np.sqrt(mean_squared_error(actuals, predictions)))

# --- Results DataFrame ---
results_df = pd.DataFrame(results)
print(results_df)

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.bar(results_df['Model'], results_df['R2_Score'], color='skyblue')
plt.title('Model Accuracy (R2 Score)')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(results_df['Model'], results_df['MAE'], color='orange')
plt.title('Model Precision (MAE - Lower is Better)')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
