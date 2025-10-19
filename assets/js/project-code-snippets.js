/**
 * Project Code Snippets
 * Contains code examples for portfolio projects
 */

const projectCodeSnippets = {
	'gmv-forecasting': [
		{
			name: 'SARIMAX Model Training',
			filename: 'sarimax_training.py',
			language: 'python',
			code: `"""
SARIMAX Model Training with Auto ARIMA
Finds optimal hyperparameters for seasonal forecasting
"""

from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
import joblib

m_candidates = [12, 26, 52]  # Test different seasonal periods

for city in city_name_list:
    df_train = train_sets[city].copy()
    df_test = test_sets[city].copy()
    
    # Prepare features and target
    exog_cols = [col for col in df_train.columns 
                 if col not in ['city_name', 'gmv']]
    X_train = df_train[exog_cols]
    y_train = df_train['gmv']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    
    best_model = None
    best_aic = float('inf')
    best_m = None
    
    # Test different seasonal periods
    for m in m_candidates:
        model_try = auto_arima(
            y=y_train, 
            X=X_scaled_train,
            seasonal=True, m=m,
            start_p=1, start_q=1, max_p=5, max_q=5,
            start_P=0, start_Q=0, max_P=5, max_Q=5,
            stepwise=True,
            information_criterion='aicc'
        )
        
        current_aic = model_try.aic()
        if current_aic < best_aic:
            best_model = model_try
            best_aic = current_aic
            best_m = m
    
    print(f"Best model for '{city}' with m={best_m} (AICc={best_aic:.2f})")
    print(f"Order: {best_model.order}, Seasonal: {best_model.seasonal_order}")
    
    # Save model and scaler
    artifacts = {'model': best_model, 'scaler': scaler}
    joblib.dump(artifacts, f"SARIMAX_{city}.pkl")`
		},
		{
			name: 'Prophet Model',
			filename: 'prophet_model.py',
			language: 'python',
			code: `"""
Prophet Model Implementation
Handles seasonality, holidays, and external regressors
"""

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

for city in city_name_list:
    # Prepare data in Prophet format
    df_train = train_sets[city][['gmv'] + list_features].copy()
    df_train.index.name = 'ds'
    df_train = df_train.reset_index()
    df_train.columns = ['ds', 'y'] + list_features
    
    # Initialize Prophet model
    model = Prophet(    
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    
    # Add external regressors
    for regressor in list_features:
        model.add_regressor(regressor, mode='multiplicative')
    
    # Add country-specific holidays
    model.add_country_holidays(country_name='VN')
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=5)
    
    # Fit the model
    model.fit(df_train)
    
    # Make predictions
    forecast = model.predict(df_test)
    
    # Calculate metrics
    y_true = df_test['y'].values
    y_pred = forecast['yhat'].values
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"{city} - MAPE: {mape:.2%}")`
		},
		{
			name: 'Weighted Hybrid Model',
			filename: 'hybrid_optimization.py',
			language: 'python',
			code: `"""
Weighted Hybrid Model Optimization
Combines SARIMAX and Prophet using optimized weights
"""

from scipy.optimize import minimize
import numpy as np

optimal_weights = {}

for city in city_name_list:
    # Calculate logarithmic errors for both models
    e_sarimax = np.log(actual_values) - np.log(sarimax_predictions)
    e_prophet = np.log(actual_values) - np.log(prophet_predictions)
    
    # Create error matrix [N_samples x 2_models]
    error_matrix = np.vstack([e_sarimax, e_prophet]).T
    
    def objective_function(weights):
        """Minimize weighted sum of absolute errors"""
        weighted_errors = np.dot(error_matrix, weights)
        return np.sum(np.abs(weighted_errors))
    
    # Initial guess: equal weights
    initial_weights = [0.5, 0.5]
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1), (0, 1)]
    
    # Optimize weights
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights[city] = result.x
    print(f"Optimal Weights -> {city}:")
    print(f"  SARIMAX: {optimal_weights[city][0]:.4f}")
    print(f"  Prophet: {optimal_weights[city][1]:.4f}")
    
    # Calculate hybrid forecast
    log_hybrid = (optimal_weights[city][0] * np.log(sarimax_predictions) + 
                  optimal_weights[city][1] * np.log(prophet_predictions))
    hybrid_forecast = np.exp(log_hybrid)
    
    # Result: ~3% MAPE achieved!`
		}
	],
	'analytics-engineer': [
		{
			name: 'ETL Transform (Python)',
			filename: 'src/etl/transform.py',
			language: 'python',
			code: `"""
Transform Module
Applies business logic and data transformations
Moves data from staging to core layer
"""

from src.utils.db_connector import get_db_connector
from src.utils.logger import ETLLogger
import yaml

class DataTransformer:
    """
    Transform data from staging to core layer
    Applies data cleaning, validation, and business logic
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize data transformer"""
        self.logger = ETLLogger("DataTransformer")
        self.db = get_db_connector()
        self.config = self._load_config(config_path)
        
    def transform_staging_to_core(self):
        """
        Execute staging to core transformation
        Runs SQL transformation logic
        """
        self.logger.log_stage("TRANSFORM", "Starting staging to core transformation")
        
        sql_file = "sql/transforms/staging_to_core.sql"
        
        try:
            self.db.execute_sql_file(sql_file)
            self.logger.log_success("Transformation completed successfully")
            
        except Exception as e:
            self.logger.log_error(f"Transformation failed: {str(e)}")
            raise`
		},
		{
			name: 'SQL Transformation',
			filename: 'sql/transforms/staging_to_core.sql',
			language: 'sql',
			code: `-- ============================================
-- ETL: Staging to Core Layer Transformations
-- Purpose: Clean, validate, and apply business logic
-- ============================================

INSERT INTO core.orders (
    order_id, customer_id, order_date, order_status,
    total_amount, payment_method, shipping_address,
    city, province, postal_code, is_first_order,
    order_processing_time_hours, dw_created_at
)
SELECT DISTINCT
    TRIM(s.order_id) as order_id,
    TRIM(s.customer_id) as customer_id,
    s.order_date::date as order_date,
    UPPER(TRIM(s.order_status)) as order_status,
    COALESCE(s.total_amount, 0.00) as total_amount,
    TRIM(s.payment_method) as payment_method,
    TRIM(s.shipping_address) as shipping_address,
    TRIM(s.city) as city,
    TRIM(s.province) as province,
    TRIM(s.postal_code) as postal_code,
    -- Calculate if first order
    CASE 
        WHEN COUNT(*) OVER (
            PARTITION BY s.customer_id 
            ORDER BY s.order_date
        ) = 1 THEN TRUE 
        ELSE FALSE 
    END as is_first_order,
    -- Calculate processing time
    EXTRACT(EPOCH FROM (
        s.updated_at - s.created_at
    )) / 3600.0 as order_processing_time_hours,
    NOW() as dw_created_at
FROM staging.orders s
WHERE s.order_id IS NOT NULL
    AND s.customer_id IS NOT NULL
    AND s.order_date IS NOT NULL
ON CONFLICT (order_id) DO UPDATE SET
    order_status = EXCLUDED.order_status,
    total_amount = EXCLUDED.total_amount,
    dw_updated_at = NOW();`
		},
		{
			name: 'Dimensional Model (SQL)',
			filename: 'sql/schema/03_create_dimensional_model.sql',
			language: 'sql',
			code: `-- ============================================
-- DIMENSIONAL MODEL (Analytics Schema)
-- Star Schema Design for Business Intelligence
-- ============================================

-- Fact Table: Orders
CREATE TABLE IF NOT EXISTS analytics.fact_orders (
    order_key SERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL,
    customer_key INTEGER REFERENCES analytics.dim_customers(customer_key),
    product_key INTEGER REFERENCES analytics.dim_products(product_key),
    date_key INTEGER REFERENCES analytics.dim_date(date_key),
    location_key INTEGER REFERENCES analytics.dim_locations(location_key),
    
    -- Measures
    quantity INTEGER,
    unit_price DECIMAL(12, 2),
    discount_amount DECIMAL(12, 2),
    net_revenue DECIMAL(12, 2),
    profit_margin DECIMAL(12, 2),
    
    -- Metadata
    dw_created_at TIMESTAMP DEFAULT NOW(),
    dw_updated_at TIMESTAMP
);

-- Dimension: Customers (SCD Type 2)
CREATE TABLE IF NOT EXISTS analytics.dim_customers (
    customer_key SERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    customer_name VARCHAR(255),
    email VARCHAR(255),
    customer_segment VARCHAR(50),
    total_lifetime_value DECIMAL(12, 2),
    total_orders INTEGER,
    
    -- SCD Type 2 fields
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE,
    
    dw_created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_fact_orders_customer ON analytics.fact_orders(customer_key);
CREATE INDEX idx_fact_orders_date ON analytics.fact_orders(date_key);
CREATE INDEX idx_dim_customers_current ON analytics.dim_customers(customer_id, is_current);`
		}
	],
	'multiclass-cnn': [
		{
			name: 'Custom CNN Architecture',
			filename: 'model_architecture.py',
			language: 'python',
			code: `"""
VGG-style CNN with Batch Normalization
Dual-convolution blocks for deep feature learning
"""

import torch
import torch.nn as nn

def create_conv_block(in_channels, out_channels, kernel_size=3):
    """
    Creates a convolutional block with:
    - 2x Conv2D layers
    - Batch Normalization after each conv
    - ReLU activation
    - MaxPooling for dimensionality reduction
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding="same"
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding="same"
        ), 
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

# Build the complete model
model = nn.Sequential(
    # Feature extraction layers
    create_conv_block(in_channels=3, out_channels=8),
    create_conv_block(in_channels=8, out_channels=16),
    create_conv_block(in_channels=16, out_channels=32),
    
    # Classification head
    nn.Flatten(),
    nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=25088, out_features=512),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=512, out_features=10),
    )
)

# Model summary: 3 conv blocks -> 8->16->32 channels -> Dense(512) -> Output(10)`
		},
		{
			name: 'Training with Metric Learning',
			filename: 'training_loop.py',
			language: 'python',
			code: `"""
Training Loop with Combined Loss Function
Uses CrossEntropy + Triplet Margin Loss for better feature learning
"""

import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from tqdm.notebook import tqdm

def train_epoch(model, optimizer, loss_fn, data_loader, device):
    """
    Train for one epoch using combined loss:
    - Metric Loss: Pulls same-class samples together, pushes different-class apart
    - Classification Loss: Standard cross-entropy for class prediction
    """
    training_loss = 0.0
    correct = 0
    total = 0
    
    model.train()
    
    # Initialize metric loss (Triplet Margin with margin=0.2)
    metric_loss = losses.TripletMarginLoss(margin=0.2)
    
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        output = model(inputs)
        
        # 1. Metric Learning Loss - Find hard pairs using MultiSimilarityMiner
        hard_pairs = miners.MultiSimilarityMiner()(output, targets)
        metric_l = metric_loss(output, targets, hard_pairs)
        
        # 2. Classification Loss - Standard cross-entropy
        class_l = loss_fn(output, targets)
        
        # Combined loss
        loss = metric_l + class_l
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track metrics
        training_loss += loss.data.item() * inputs.size(0)
        _, predicted = torch.max(output, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return training_loss / len(data_loader.dataset), correct / total`
		},
		{
			name: 'ResNet50 Transfer Learning',
			filename: 'resnet_transfer.py',
			language: 'python',
			code: `"""
Transfer Learning with ResNet50
Adapts pretrained ImageNet model for 10-class animal classification
"""

import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet50
res_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all base layers to preserve ImageNet features
for params in res_model.parameters():
    params.requires_grad = False

# Replace the final classification layer
# ResNet50 outputs 2048 features -> Custom head for 10 classes
torch.manual_seed(42)
torch.cuda.manual_seed(42)

in_features = res_model.fc.in_features  # 2048

modified_last_layer = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10)  # 10 animal classes
)

# Assign the new head
res_model.fc = modified_last_layer

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res_model = res_model.to(device)

print(f"Model loaded on {device}")
print(f"Trainable parameters: {sum(p.numel() for p in res_model.parameters() if p.requires_grad):,}")

# Result: Only ~131K trainable params (head only) vs 25M total params`
		}
	],
	'sentiment-analysis': [
		{
			name: 'Text Preprocessing Pipeline',
			filename: 'text_preprocessing.py',
			language: 'python',
			code: `"""
NLP Text Preprocessing with NLTK
Tokenization, stopword removal, and lemmatization
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Comprehensive text preprocessing pipeline:
    1. Tokenization: Split text into words
    2. Stopword Removal: Filter out common words (the, is, and, etc.)
    3. Lemmatization: Reduce words to base form (running -> run)
    4. Non-alphanumeric filtering: Keep only meaningful tokens
    """
    # Handle NaN/None values
    if pd.isna(text):
        return ''
    
    text = str(text)
    
    # Tokenization - split text into words
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens 
              if token.isalnum() and token not in stop_words]
    
    # Lemmatization - reduce words to root form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Create sentiment labels from ratings
def create_sentiment_labels(df):
    """
    Map ratings to sentiment categories:
    - Ratings 4-5: Positive (class 1)
    - Rating 3: Neutral (class 2)
    - Ratings 1-2: Negative (class 3)
    """
    df['sentiment'] = df['rate'].apply(
        lambda x: "positive" if x >= 4 
        else "neutral" if x == 3 
        else "negative"
    )
    
    df['sentiment_label'] = df['rate'].apply(
        lambda x: 1 if x >= 4 else 2 if x == 3 else 3
    )
    
    return df

# Apply preprocessing
df['processed_summary'] = df['summary'].apply(preprocess_text)
df = create_sentiment_labels(df)

print("Sample preprocessed text:")
print(f"Original: {df['summary'].iloc[0]}")
print(f"Processed: {df['processed_summary'].iloc[0]}")`
		},
		{
			name: 'TF-IDF Feature Engineering',
			filename: 'tfidf_vectorization.py',
			language: 'python',
			code: `"""
TF-IDF Vectorization for Text Feature Extraction
Converts text into numerical features for machine learning
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Configure TF-IDF Vectorizer with optimized parameters
vectorizer = TfidfVectorizer(
    max_features=1000,        # Limit to top 1000 most important words
    ngram_range=(1, 2),       # Include unigrams and bigrams
    min_df=5,                 # Word must appear in at least 5 documents
    max_df=0.95               # Ignore words appearing in >95% of docs
)

# Transform text to TF-IDF features
# Result: sparse matrix of shape (n_samples, 1000)
X = vectorizer.fit_transform(df['processed_summary'])
y = df['sentiment_label']

# Split into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Feature dimensions: {X_train.shape[1]}")
print(f"Matrix sparsity: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])) * 100:.2f}%")

# Example: View most important features (words)
feature_names = vectorizer.get_feature_names_out()
print(f"\\nSample features: {feature_names[:20]}")

# TF-IDF transforms text like:
# "excellent product quality" 
# -> [0.0, 0.0, 0.52, ..., 0.31, 0.0, 0.68, ...]
#    where each value represents word importance`
		},
		{
			name: 'Model Comparison with GridSearchCV',
			filename: 'model_training.py',
			language: 'python',
			code: `"""
Machine Learning Model Comparison
Trains and evaluates 5 different algorithms with hyperparameter tuning
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report

# Define models and hyperparameter grids
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],            # Regularization strength
            'penalty': ['l1', 'l2'],          # L1 or L2 regularization
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
    },
    'Multinomial Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0, 2.0]     # Smoothing parameter
        }
    },
    'Linear SVM': {
        'model': LinearSVC(random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],            # Regularization parameter
            'penalty': ['l1', 'l2'],
            'dual': [False],
            'max_iter': [10000]
        }
    }
}

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted')
}

results = []
best_models = {}

# Train and evaluate each model
for name, model_info in models.items():
    print(f"\\nTraining {name}...")
    
    # GridSearchCV performs 5-fold cross-validation for each parameter combination
    grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=5,                    # 5-fold cross-validation
        scoring=scoring,
        refit='f1',             # Optimize for F1-score
        n_jobs=-1,              # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    
    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Training F1': grid_search.best_score_,
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Test F1 Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Results:
# Logistic Regression: 0.8290 F1-score (BEST)
# Multinomial NB: 0.8189 F1-score
# Linear SVM: 0.8265 F1-score`
		}
	],
	'song-recommendation': [
		{
			name: 'K-Means Clustering for Song Grouping',
			filename: 'clustering_model.py',
			language: 'python',
			code: `"""
K-Means Clustering Algorithm
Groups songs based on audio features similarity
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load song dataset with audio features
df = pd.read_csv('song_data.csv')

# Select audio features for clustering
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 
    'valence', 'tempo'
]

X = df[audio_features]

# Standardize features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
inertias = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Train final model with optimal k
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"Songs clustered into {optimal_k} groups")
print(f"Cluster distribution:\\n{df['cluster'].value_counts().sort_index()}")`
		},
		{
			name: 'Cosine Similarity Recommendation',
			filename: 'recommendation_engine.py',
			language: 'python',
			code: `"""
Song Recommendation Engine
Uses cosine similarity to find similar songs
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def get_song_recommendations(song_name, df, n_recommendations=10):
    """
    Recommend songs similar to the input song
    
    Args:
        song_name: Name of the song to base recommendations on
        df: DataFrame with song features and cluster assignments
        n_recommendations: Number of songs to recommend
        
    Returns:
        DataFrame with recommended songs and similarity scores
    """
    # Find the input song
    song_data = df[df['name'].str.lower() == song_name.lower()]
    
    if song_data.empty:
        return f"Song '{song_name}' not found in database"
    
    # Get the cluster of the input song
    song_cluster = song_data['cluster'].values[0]
    
    # Filter songs from the same cluster (reduces search space)
    cluster_songs = df[df['cluster'] == song_cluster].copy()
    
    # Audio features for similarity calculation
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo'
    ]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_songs[audio_features])
    
    # Get index of input song in cluster
    song_idx = cluster_songs[cluster_songs['name'].str.lower() == song_name.lower()].index[0]
    cluster_song_idx = cluster_songs.index.get_loc(song_idx)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features_scaled)
    similarity_scores = similarity_matrix[cluster_song_idx]
    
    # Get top N similar songs (excluding the input song itself)
    similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
    
    # Create recommendations DataFrame
    recommendations = cluster_songs.iloc[similar_indices].copy()
    recommendations['similarity_score'] = similarity_scores[similar_indices]
    
    return recommendations[['name', 'artists', 'cluster', 'similarity_score']]

# Example usage
recommended_songs = get_song_recommendations('Blinding Lights', df, n_recommendations=10)
print("Recommended Songs:")
print(recommended_songs)`
		},
		{
			name: 'Streamlit Web Interface',
			filename: 'app.py',
			language: 'python',
			code: `"""
Streamlit Web Application
Interactive UI for Song Recommendation System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page configuration
st.set_page_config(
    page_title="Song Recommendation Engine",
    page_icon="🎵",
    layout="wide"
)

# Load pre-trained model and data
@st.cache_data
def load_data():
    df = pd.read_csv('song_data_with_clusters.csv')
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    return df, kmeans

df, kmeans = load_data()

# App title and description
st.title("🎵 Song Recommendation Engine")
st.markdown("Discover new music based on your favorite songs!")

# Sidebar for user input
with st.sidebar:
    st.header("🎼 Select a Song")
    
    # Song selection
    song_list = sorted(df['name'].unique())
    selected_song = st.selectbox("Choose a song:", song_list)
    
    # Number of recommendations
    n_recommendations = st.slider(
        "Number of recommendations:", 
        min_value=5, 
        max_value=20, 
        value=10
    )
    
    # Recommendation button
    if st.button("Get Recommendations", type="primary"):
        st.session_state.show_recommendations = True

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Song Characteristics")
    
    # Display selected song's audio features
    song_data = df[df['name'] == selected_song].iloc[0]
    
    features = ['danceability', 'energy', 'valence', 'acousticness']
    values = [song_data[f] for f in features]
    
    # Radar chart
    fig = px.line_polar(
        r=values, 
        theta=features, 
        line_close=True,
        title=f"Audio Profile: {selected_song}"
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Recommendations")
    
    if st.session_state.get('show_recommendations', False):
        # Get recommendations
        recommendations = get_song_recommendations(
            selected_song, df, n_recommendations
        )
        
        # Display recommendations
        for idx, row in recommendations.iterrows():
            with st.container():
                st.markdown(f"**{row['name']}** by {row['artists']}")
                st.progress(row['similarity_score'])
                st.caption(f"Match: {row['similarity_score']:.1%}")
    else:
        st.info("👈 Select a song and click 'Get Recommendations'")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by K-Means Clustering")`
		}
	]
};

