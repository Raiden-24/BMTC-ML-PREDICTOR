import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from math import radians, sin, cos, atan2, sqrt
from catboost import CatBoostRegressor  # New import for CatBoost

# --- 1. Advanced Feature Engineering & Data Expansion ---

# Helper function to calculate Haversine distance between two lat/lon points
# This is used to re-calculate route lengths from bus stop coordinates
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to process the raw data, expand trips, and engineer features
def process_data(df_raw):
    # Rename columns for clarity and consistency
    df_raw.rename(columns={
        'departure_from_origin': 'Start_time_raw',
        'arrival_at_destination': 'Reach_time_raw',
        'departure_from_destination': 'Return_trip_start_time_raw',
        'arrival_at_origin': 'Return_trip_reach_time_raw'
    }, inplace=True)

    expanded_data = []  # Initialize the list to hold expanded data
    # Iterate through each row of the raw DataFrame (each route)
    for index, row in df_raw.iterrows():
        route_no = row['route_no']
        # Clean 'distance' column and convert to float
        distance = float(row['distance'].replace(" KM", "").strip())
        origin = row['origin']
        destination = row['destination']
        map_json_content = row['map_json_content']

        # Parse map_json_content to extract geospatial features [4]
        number_of_bus_stops = np.nan
        recalculated_route_length = np.nan
        average_distance_between_stops = np.nan

        try:
            bus_stops_list = json.loads(map_json_content)
            number_of_bus_stops = len(bus_stops_list)  # Number of bus stops [5, 6]
            
            recalculated_route_length = 0
            if number_of_bus_stops > 1:
                # Calculate total route length by summing Haversine distances between consecutive stops
                for i in range(number_of_bus_stops - 1):
                    # Ensure latlons are accessed as list elements and converted to float
                    lat1, lon1 = float(bus_stops_list[i]['latlons'][0]), float(bus_stops_list[i]['latlons'][1])
                    lat2, lon2 = float(bus_stops_list[i+1]['latlons'][0]), float(bus_stops_list[i+1]['latlons'][1])
                    recalculated_route_length += haversine_distance(lat1, lon1, lat2, lon2)
            
            # Calculate average distance between stops [4]
            average_distance_between_stops = recalculated_route_length / (number_of_bus_stops - 1) if number_of_bus_stops > 1 else 0

        except (json.JSONDecodeError, TypeError, IndexError):  # Added IndexError for safer latlons access
            # Handle cases where map_json_content might be malformed or missing
            number_of_bus_stops = np.nan
            recalculated_route_length = np.nan
            average_distance_between_stops = np.nan

        # Split comma-separated time strings into lists of individual trip times [4]
        start_times = row['Start_time_raw'].split(',')
        reach_times = row['Reach_time_raw'].split(',')

        # Expand the DataFrame: create a new row for each scheduled trip
        for i in range(min(len(start_times), len(reach_times))):
            expanded_data.append({
                'route_no': route_no,
                'distance': distance,
                'origin': origin,
                'destination': destination,
                'Start_time': start_times[i].strip(),  #.strip() to remove leading/trailing spaces
                'Reach_time': reach_times[i].strip(),  #.strip() to remove leading/trailing spaces
                'number_of_bus_stops': number_of_bus_stops,
                'recalculated_route_length': recalculated_route_length,
                'average_distance_between_stops': average_distance_between_stops
            })

    # Create a new DataFrame from the expanded data
    df_expanded = pd.DataFrame(expanded_data)

    # Convert time columns to datetime objects
    df_expanded['Start_time'] = pd.to_datetime(df_expanded['Start_time'], format="%H:%M", errors="coerce")
    df_expanded['Reach_time'] = pd.to_datetime(df_expanded['Reach_time'], format="%H:%M", errors="coerce")

    # Calculate Journey_Duration_Minutes, adjusting for overnight trips
    df_expanded['Journey_Duration_Minutes'] = (df_expanded['Reach_time'] - df_expanded['Start_time']).dt.total_seconds() / 60
    df_expanded.loc[df_expanded['Journey_Duration_Minutes'] < 0, "Journey_Duration_Minutes"] += 24 * 60  # Add 24 hours for overnight trips

    # Extract more granular temporal features [1, 7, 8]
    df_expanded['Start_hour'] = df_expanded['Start_time'].dt.hour  # Explicitly add Start_hour
    df_expanded['minute_of_day'] = df_expanded['Start_time'].dt.hour * 60 + df_expanded['Start_time'].dt.minute
    df_expanded['day_of_week'] = df_expanded['Start_time'].dt.dayofweek  # Monday=0, Sunday=6
    df_expanded['is_weekend'] = df_expanded['day_of_week'].isin([5, 6]).astype(int)  # Binary indicator for weekend

    # Apply cyclical encoding to 'Start_hour' and 'day_of_week' [9, 2, 3, 10]
    # This preserves the cyclical nature (e.g., 23:00 is close to 00:00)
    df_expanded['Start_hour_sin'] = np.sin(2 * np.pi * df_expanded['Start_hour'] / 24)
    df_expanded['Start_hour_cos'] = np.cos(2 * np.pi * df_expanded['Start_hour'] / 24)
    df_expanded['day_of_week_sin'] = np.sin(2 * np.pi * df_expanded['day_of_week'] / 7)
    df_expanded['day_of_week_cos'] = np.cos(2 * np.pi * df_expanded['day_of_week'] / 7)

    # Define time bins (can be used as a categorical feature)
    def time_bin(hour):
        if 5 <= hour < 10:
            return "Morning Rush"
        elif 10 <= hour < 16:
            return "Midday"
        elif 16 <= hour < 20:
            return "Evening Rush"
        else:
            return "Off Hours"
    df_expanded["time_bin"] = df_expanded['Start_hour'].apply(time_bin)

    # Define all features that should not be NaN for model training
    all_model_features = ['distance', 'number_of_bus_stops', 'recalculated_route_length', 'average_distance_between_stops', 'minute_of_day', 'Start_hour_sin', 'Start_hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'route_no', 'origin', 'destination', 'time_bin', 'is_weekend']
    
    # Drop rows with any NaN values in critical columns after all processing
    df_expanded = df_expanded.dropna(subset=all_model_features)

    return df_expanded

# Load and preprocess data using the new function
df_raw = pd.read_csv("C://Users//Amruth//Desktop//ML model project final//bmtc_own_dataset.csv")
df = process_data(df_raw.copy())  # Use.copy() to ensure original df_raw is not modified

# --- 2. Robust Outlier Handling for Target Variable ---
df['Journey_Duration_Minutes_Original'] = df['Journey_Duration_Minutes']
df['Journey_Duration_Minutes'] = np.log1p(df['Journey_Duration_Minutes'])

# --- 3. Optimized Model Selection (CatBoost Regressor) ---

numerical_features = ['distance', 'number_of_bus_stops', 'recalculated_route_length', 'average_distance_between_stops', 'minute_of_day', 'Start_hour_sin', 'Start_hour_cos', 'day_of_week_sin', 'day_of_week_cos']
categorical_features = ['route_no', 'origin', 'destination', 'time_bin', 'is_weekend']

X = df[numerical_features + categorical_features]
y = df['Journey_Duration_Minutes']  # Use the log-transformed target

# Identify categorical feature indices for CatBoost
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters like iterations, learning_rate, depth, l2_leaf_reg can be further tuned for optimal performance.
model = CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.05, 
    depth=8, 
    l2_leaf_reg=3, 
    loss_function='RMSE', 
    eval_metric='RMSE', 
    random_seed=42,
    verbose=0, 
    cat_features=cat_features_indices 
)
model.fit(X_train, y_train)

# Predict on test set and inverse transform predictions to original scale for evaluation
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Inverse transform using expm1 (e^x - 1)

# Calculate evaluation metrics on the original scale for interpretability
r2 = r2_score(np.expm1(y_test), y_pred)  # Inverse transform y_test for comparison
mae = mean_absolute_error(np.expm1(y_test), y_pred)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
print("--- Model Evaluation ---")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print("------------------------")

# Dash app setup
app = Dash(__name__)
app.title = "BMTC Journey Dashboard with Enhanced Regression Model"

# Get unique values for dropdowns from the processed DataFrame
unique_routes = sorted(df["route_no"].unique())
unique_origins = sorted(df["origin"].unique())
unique_destinations = sorted(df["destination"].unique())
# No need for unique_days_of_week as it's 0-6 numerical input

# Layout
app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "auto",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "padding": "20px",
        "backgroundColor": "#f5f7fa"
    },
    children=[
        html.H2("BMTC Journey Dashboard", style={
            "textAlign": "center",
            "color": "#0074D9",
            "marginBottom": "30px"
        }),

        dcc.Dropdown(
            id="route-dropdown",
            options=[{"label": route, "value": route} for route in unique_routes],
            value=unique_routes[0],
            clearable=False,
            style={
                "width": "100%",
                "padding": "10px",
                "fontSize": "16px",
                "borderRadius": "5px",
                "marginBottom": "20px"
            }
        ),
        html.Div(style={"width": "48%", "backgroundColor": "#f9f9f9", "padding": "15px", "borderRadius": "8px"}, children=[
            html.H4("Model Evaluation Metrics", style={"marginTop": "0"}),
            html.P(f"R² Score: {r2:.3f}"),
            html.P(f"Mean Absolute Error (MAE): {mae:.2f} minutes"),
            html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes"),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr",
            "gap": "20px",
            "backgroundColor": "#ffffff",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)",
            "marginBottom": "30px"
        }, children=[
            dcc.Graph(id="duration-graph"),
            dcc.Graph(id="distance-histogram"),
            dcc.Graph(id="origin-bar"),
            dcc.Graph(id="destination-pie"),
        ]),

        html.Div(style={
            "padding": "25px",
            "border": "2px solid #0074D9",
            "borderRadius": "12px",
            "backgroundColor": "#E8F4FD",
            "boxShadow": "0 4px 10px rgba(0, 116, 217, 0.2)"
        }, children=[
            html.H3("Predict Journey Duration", style={
                "textAlign": "center",
                "color": "#003366",
                "marginBottom": "20px"
            }),

            dcc.Dropdown(
                id="input-route-no",
                options=[{"label": route, "value": route} for route in unique_routes],
                placeholder="Select route number",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "15px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc"
                }
            ),
            dcc.Input(
                id="input-distance",
                type="number",
                placeholder="Distance (KM)",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "15px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc"
                }
            ),
            dcc.Input(
                id="input-start-hour",
                type="number",
                placeholder="Start Hour (0-23)",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "15px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc"
                }
            ),
            dcc.Input(
                id="input-day-of-week",
                type="number",
                placeholder="Day of Week (0-6)",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "15px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc"
                }
            ),
            dcc.Dropdown(
                id="input-origin",
                options=[{"label": origin, "value": origin} for origin in unique_origins],
                placeholder="Select origin",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "15px",
                    "fontSize": "16px",
                    "borderRadius": "5px"
                }
            ),
            dcc.Dropdown(
                id="input-destination",
                options=[{"label": destination, "value": destination} for destination in unique_destinations],
                placeholder="Select destination",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "20px",
                    "fontSize": "16px",
                    "borderRadius": "5px"
                }
            ),
            html.Button(
                "Predict",
                id="predict-button",
                n_clicks=0,
                style={
                    "width": "100%",
                    "padding": "12px",
                    "backgroundColor": "#0074D9",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "fontSize": "16px",
                    "cursor": "pointer"
                }
            ),
            html.Div(
                id="prediction-output",
                style={
                    "marginTop": "20px",
                    "fontWeight": "bold",
                    "fontSize": "18px",
                    "color": "#003366",
                    "textAlign": "center"
                }
            ),
        ]),
    ]
)

# Callbacks for graphs (remain mostly the same, operating on the new 'df')
@app.callback(
    Output("duration-graph", "figure"),
    Output("distance-histogram", "figure"),
    Output("origin-bar", "figure"),
    Output("destination-pie", "figure"),
    Input("route-dropdown", "value"),
)
def update_graphs(selected_route):
    filtered_df = df[df["route_no"] == selected_route]

    # Journey duration over time (Start_time)
    # Use 'Journey_Duration_Minutes_Original' for plotting to show actual durations
    duration_fig = px.bar(
        filtered_df,
        x="Start_time",
        y="Journey_Duration_Minutes_Original",
        title=f"Journey Duration over Time for Route {selected_route}",
        labels={"Start_time": "Start Time", "Journey_Duration_Minutes_Original": "Duration (min)"}
    )
    duration_fig.update_layout(xaxis_title="Start Time", yaxis_title="Journey Duration (minutes)")

    # Distance histogram (all routes)
    dist_fig = px.histogram(
        df,
        x="distance",
        nbins=20,
        title="Distribution of Route Distances",
        labels={"distance": "Distance (KM)"}
    )

    # Top 10 origins by count
    origin_counts = df["origin"].value_counts().head(10)
    origin_fig = px.bar(
        x=origin_counts.index,
        y=origin_counts.values,
        labels={"x": "Origin", "y": "Count"},
        title="Top 10 Origins"
    )

    # Least 10 destinations by count
    dest_counts = df["destination"].value_counts().tail(10)
    dest_fig = px.pie(
        names=dest_counts.index,
        values=dest_counts.values,
        title="Least 10 Destinations"
    )

    return duration_fig, dist_fig, origin_fig, dest_fig

# Callback for prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("input-route-no", "value"), 
    State("input-distance", "value"),
    State("input-start-hour", "value"),
    State("input-day-of-week", "value"), 
    State("input-origin", "value"),
    State("input-destination", "value"),
)
def predict_duration(n_clicks, route_no, distance, start_hour, day_of_week, origin, destination):
    if n_clicks == 0:
        return ""

    # Validate all required inputs are filled
    if route_no is None or distance is None or start_hour is None or day_of_week is None or origin is None or destination is None:
        return "Please fill in all input fields to predict."

    # Validate numerical ranges
    if not (0 <= start_hour <= 23):
        return "Start hour must be between 0 and 23."
    if not (0 <= day_of_week <= 6):
        return "Day of week must be between 0 (Monday) and 6 (Sunday)."

    # Retrieve route-specific features (number_of_bus_stops, recalculated_route_length, average_distance_between_stops)
    # from the pre-processed global DataFrame 'df' based on the selected route_no.
    # This assumes these values are consistent for a given route_no.
    route_info = df[df['route_no'] == route_no]
    if route_info.empty:
        return f"No data found for route {route_no}. Please select a valid route."
    
    # Take the first entry for the route, as these features are constant per route
    route_info = route_info.iloc[0]
    number_of_bus_stops = route_info['number_of_bus_stops']
    recalculated_route_length = route_info['recalculated_route_length']
    average_distance_between_stops = route_info['average_distance_between_stops']

    # Derive additional temporal features for prediction input
    minute_of_day = start_hour * 60  # For simplicity, assuming minute is 0 for prediction input
    is_weekend = 1 if day_of_week in [5, 6] else 0
    start_hour_sin = np.sin(2 * np.pi * start_hour / 24)
    start_hour_cos = np.cos(2 * np.pi * start_hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Determine time bin for prediction input
    def time_bin_predict(hour):
        if 5 <= hour < 10:
            return "Morning Rush"
        elif 10 <= hour < 16:
            return "Midday"
        elif 16 <= hour < 20:
            return "Evening Rush"
        else:
            return "Off Hours"
    time_bin_val = time_bin_predict(start_hour)

    # Prepare feature vector for CatBoost prediction
    # Ensure the feature columns match the training data (X)
    input_data = pd.DataFrame([[
        distance,
        number_of_bus_stops,
        recalculated_route_length,
        average_distance_between_stops,
        minute_of_day,
        start_hour_sin,
        start_hour_cos,
        day_of_week_sin,
        day_of_week_cos,
        route_no,
        origin,
        destination,
        time_bin_val,
        is_weekend
    ]], columns=X.columns)

    # Predict log-transformed duration and inverse transform to original scale
    predicted_duration_log = model.predict(input_data)
    predicted_duration = np.expm1(predicted_duration_log)  # expm1 = e^x - 1

    return f"Predicted Journey Duration: {predicted_duration[0]:.2f} minutes"

if __name__ == "__main__":
    app.run(debug=True)
