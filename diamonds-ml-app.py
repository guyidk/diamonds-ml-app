import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.ensemble import IsolationForest

# Load and preprocess the diamond dataset
@st.cache
def load_data():
    df = pd.read_csv('diamonds.csv')

    # Remove entries where 'x', 'y', or 'z' are below 0.1
    df = df.loc[df['x'] >= 0.1]
    df = df.loc[df['y'] >= 0.1]
    df = df.loc[df['z'] >= 0.1]

    # One-hot encoding of categorical columns
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

    # Remove outliers using z-score
    from scipy.stats import zscore
    df = df[(np.abs(zscore(df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])) < 3).all(axis=1)]

    # Apply IsolationForest for outlier detection
    model = IsolationForest(contamination=0.05)
    outliers = model.fit_predict(df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])
    df = df[outliers == 1]

    return df

# Train the model
@st.cache
def train_model(df):
    X = df.drop('price', axis=1).to_numpy()
    y = df['price'].to_numpy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(
        n_estimators=1000, 
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=9,
        max_features=0.1,
        loss="huber",
        random_state=7
    )
    gb_model.fit(X_train, y_train)

    # Test the model
    y_pred = gb_model.predict(X_test)

    # Print evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    st.write(f"**Model Evaluation:**")
    st.write(f"- Mean Squared Error: {mse:.2f}")
    st.write(f"- Mean Absolute Error: {mae:.2f}")
    st.write(f"- R-squared: {r2:.2f}")
    st.write(f"- Explained Variance Score: {evs:.2f}")

    return gb_model, X_train, y_train

# Streamlit UI for user input
def user_input_features():
    st.sidebar.header('Enter Diamond Features')

    carat = st.sidebar.slider('Carat', 0.2, 5.0, 1.0)
    cut = st.sidebar.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.sidebar.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.sidebar.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.sidebar.slider('Depth', 43.0, 79.0, 61.0)
    table = st.sidebar.slider('Table', 43.0, 95.0, 58.0)
    x = st.sidebar.slider('X (length)', 0.0, 10.0, 6.0)
    y = st.sidebar.slider('Y (width)', 0.0, 10.0, 6.0)
    z = st.sidebar.slider('Z (depth)', 0.0, 10.0, 3.0)

    features = pd.DataFrame({
        'carat': [carat],
        'cut_Fair': [1 if cut == 'Fair' else 0],
        'cut_Good': [1 if cut == 'Good' else 0],
        'cut_Very Good': [1 if cut == 'Very Good' else 0],
        'cut_Premium': [1 if cut == 'Premium' else 0],
        'cut_Ideal': [1 if cut == 'Ideal' else 0],
        'color_D': [1 if color == 'D' else 0],
        'color_E': [1 if color == 'E' else 0],
        'color_F': [1 if color == 'F' else 0],
        'color_G': [1 if color == 'G' else 0],
        'color_H': [1 if color == 'H' else 0],
        'color_I': [1 if color == 'I' else 0],
        'color_J': [1 if color == 'J' else 0],
        'clarity_I1': [1 if clarity == 'I1' else 0],
        'clarity_SI2': [1 if clarity == 'SI2' else 0],
        'clarity_SI1': [1 if clarity == 'SI1' else 0],
        'clarity_VS2': [1 if clarity == 'VS2' else 0],
        'clarity_VS1': [1 if clarity == 'VS1' else 0],
        'clarity_VVS2': [1 if clarity == 'VVS2' else 0],
        'clarity_VVS1': [1 if clarity == 'VVS1' else 0],
        'clarity_IF': [1 if clarity == 'IF' else 0],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    return features

# Streamlit app layout
st.title("Diamond Price Prediction App")
st.write("""
This app predicts the **price of a diamond** based on its features!
""")

# Load and train the model
df = load_data()
gb_model, X_train, y_train = train_model(df)

# Get user input
user_input = user_input_features()

st.subheader("User Input Features")
st.write(user_input)

# Predict the price
price_prediction = gb_model.predict(user_input)
st.subheader('Predicted Price')
st.write(f"${price_prediction[0]:,.2f}")
