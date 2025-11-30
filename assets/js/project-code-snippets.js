/**
 * Project Code Snippets
 * Contains code examples for portfolio projects
 */

const projectCodeSnippets = {
	'gmv-forecasting': [
		{
			name: 'ML Pipeline with MLflow Tracking',
			filename: 'src/models/train.py',
			language: 'python',
			code: `"""
Production ML Training Pipeline with MLflow Experiment Tracking
Trains hybrid SARIMAX + Prophet model with automated logging
"""

import mlflow
import mlflow.pyfunc
from pathlib import Path
from src.data.data_loader import DataLoader, create_train_test_sets
from src.models.hybrid_model import HybridForecaster
from src.utils.config_loader import load_config
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error

def train_city_model(city, train_data, test_data, config, model_config):
    """Train hybrid model for a specific city"""
    # Prepare data
    target_col = config['data']['target_column']
    exog_features = config['data']['exogenous_features']
    
    y_train = train_data[target_col].values
    X_train = train_data[exog_features]
    dates_train = train_data.index
    
    y_test = test_data[target_col].values
    X_test = test_data[exog_features]
    dates_test = test_data.index
    
    # Get model configurations
    sarimax_config = config['training']['sarimax']
    prophet_config = config['training']['prophet']
    hybrid_config = config['training']['hybrid']
    
    # Initialize and train hybrid model
    model = HybridForecaster(
        sarimax_config=sarimax_config,
        prophet_config=prophet_config,
        hybrid_config=hybrid_config
    )
    
    # Train with validation data for weight optimization
    model.fit(
        y_train=y_train,
        X_train=X_train,
        dates_train=dates_train,
        y_val=y_test,
        X_val=X_test,
        dates_val=dates_test
    )
    
    return model

# Main training loop
config = load_config('config/config.yaml')
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Load and split data
loader = DataLoader(...)
df = loader.preprocess_data()
train_sets, test_sets = create_train_test_sets(df, cities, split_date)

for city in cities:
    with mlflow.start_run(run_name=f"hybrid_model_{city}"):
        # Log parameters
        mlflow.log_params({
            'city': city,
            'model_type': 'hybrid_sarimax_prophet'
        })
        
        # Train model
        model = train_city_model(city, train_sets[city], 
                                test_sets[city], config, model_config)
        
        # Evaluate
        y_pred, _ = model.predict(len(y_test), X_test, dates_test)
        metrics = {
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': root_mean_squared_error(y_test, y_pred)
        }
        
        # Log to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(model_path))
        
        print(f"✅ {city}: MAPE={metrics['mape']:.2%}")`
		},
		{
			name: 'Hybrid Model with Weight Optimization',
			filename: 'src/models/hybrid_model.py',
			language: 'python',
			code: `"""
Weighted Hybrid Model: SARIMAX + Prophet
Optimizes ensemble weights using validation data
"""

import numpy as np
from scipy.optimize import minimize
from src.models.sarimax_model import SARIMAXForecaster
from src.models.prophet_model import ProphetForecaster

class HybridForecaster:
    """Weighted hybrid model combining SARIMAX and Prophet"""
    
    def __init__(self, sarimax_config, prophet_config, hybrid_config):
        self.sarimax_model = SARIMAXForecaster(sarimax_config)
        self.prophet_model = ProphetForecaster(prophet_config)
        self.hybrid_config = hybrid_config
        self.weights = np.array([0.5, 0.5])  # Initial weights
        self.is_fitted = False
    
    def fit(self, y_train, X_train, dates_train, 
            y_val=None, X_val=None, dates_val=None):
        """Fit hybrid model and optimize weights"""
        # Train SARIMAX component
        self.sarimax_model.fit(y_train, X_train.values)
        
        # Train Prophet component (with fallback for Windows)
        try:
            self.prophet_model.fit(y_train, X_train, dates_train)
            prophet_available = True
        except Exception as e:
            logger.warning(f"Prophet failed: {e}. Using SARIMAX-only.")
            prophet_available = False
            self.weights = np.array([1.0, 0.0])
        
        # Optimize weights if both models available
        if prophet_available and y_val is not None:
            self._optimize_weights(y_val, X_val, dates_val)
        
        self.is_fitted = True
        return self
    
    def _optimize_weights(self, y_val, X_val, dates_val):
        """Optimize ensemble weights using validation data"""
        n_periods = len(y_val)
        
        # Get predictions from both models
        sarimax_preds, _ = self.sarimax_model.predict(n_periods, X_val.values)
        prophet_preds, _ = self.prophet_model.predict(n_periods, X_val, dates_val)
        
        # Calculate logarithmic errors
        e_sarimax = np.log(y_val) - np.log(sarimax_preds)
        e_prophet = np.log(y_val) - np.log(prophet_preds)
        error_matrix = np.vstack([e_sarimax, e_prophet]).T
        
        def objective(weights):
            """Minimize sum of absolute weighted errors"""
            return np.sum(np.abs(np.dot(error_matrix, weights)))
        
        # Optimize with constraints: weights sum to 1, range [0,1]
        result = minimize(
            objective,
            x0=[0.5, 0.5],
            method='SLSQP',
            bounds=[(0, 1), (0, 1)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        if result.success:
            self.weights = result.x
            print(f"Optimal weights: SARIMAX={self.weights[0]:.4f}, "
                  f"Prophet={self.weights[1]:.4f}")
    
    def predict(self, n_periods, X, dates):
        """Make hybrid predictions using weighted average"""
        # Get predictions from both models
        sarimax_preds, _ = self.sarimax_model.predict(n_periods, X.values)
        prophet_preds, _ = self.prophet_model.predict(n_periods, X, dates)
        
        # Combine using logarithmic weighted average
        log_hybrid = (self.weights[0] * np.log(sarimax_preds) + 
                      self.weights[1] * np.log(prophet_preds))
        hybrid_preds = np.exp(log_hybrid)
        
        return hybrid_preds, {
            'sarimax': sarimax_preds,
            'prophet': prophet_preds
        }`
		},
		{
			name: 'FastAPI Production Deployment',
			filename: 'src/api/main.py',
			language: 'python',
			code: `"""
FastAPI RESTful API for Model Serving
Provides prediction endpoints with Prometheus monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, generate_latest
import mlflow.pyfunc
import pandas as pd
import logging

app = FastAPI(
    title="GMV Forecasting API",
    description="Production ML API for GMV predictions",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_request_count', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_latency', 'Request latency')
PREDICTION_COUNT = Counter('prediction_count', 'Total predictions made')

# Load models from MLflow
models = {}
CITIES = ['hanoi', 'hcmc', 'danang', 'haiphong']

for city in CITIES:
    models[city] = mlflow.pyfunc.load_model(f"models/{city}/hybrid_model")

# Pydantic schemas
class PredictionRequest(BaseModel):
    city: str = Field(..., description="City name")
    features: Dict[str, float] = Field(..., description="Feature values")
    
class PredictionResponse(BaseModel):
    city: str
    predicted_gmv: float
    model_version: str
    confidence_interval: Dict[str, float]

# API endpoints
@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/models")
def list_models():
    """List available city models"""
    return {
        "available_models": CITIES,
        "model_type": "SARIMAX + Prophet Hybrid"
    }

@app.post("/predict", response_model=PredictionResponse)
@REQUEST_LATENCY.time()
def predict(request: PredictionRequest):
    """
    Generate GMV forecast for specified city
    
    Returns:
        Predicted GMV with confidence intervals
    """
    REQUEST_COUNT.inc()
    
    if request.city not in CITIES:
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found for city: {request.city}"
        )
    
    try:
        # Prepare input data
        input_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = models[request.city].predict(input_df)
        
        PREDICTION_COUNT.inc()
        
        return PredictionResponse(
            city=request.city,
            predicted_gmv=float(prediction[0]),
            model_version="1.0.0",
            confidence_interval={
                "lower": float(prediction[0] * 0.95),
                "upper": float(prediction[0] * 1.05)
            }
        )
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest()

# Run with: uvicorn src.api.main:app --reload --port 8000`
		},
		{
			name: 'Docker Compose Orchestration',
			filename: 'deployment/docker-compose.yml',
			language: 'yaml',
			code: `# ==================================================
# Docker Compose - Full MLOps Stack
# Services: API, MLflow, PostgreSQL, Redis, Prometheus, Grafana
# ==================================================

version: '3.8'

services:
  # FastAPI Application
  api:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    container_name: gmv-api
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5001
      - DATABASE_URL=postgresql://user:password@postgres:5432/gmv_db
      - REDIS_URL=redis://redis:6379
    volumes:
      - ../models:/app/models
      - ../data:/app/data
    depends_on:
      - postgres
      - redis
      - mlflow
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    networks:
      - mlops-network

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    container_name: gmv-mlflow
    ports:
      - "5001:5001"
    environment:
      - BACKEND_STORE_URI=postgresql://user:password@postgres:5432/mlflow_db
      - DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    command: >
      mlflow server
      --backend-store-uri postgresql://user:password@postgres:5432/mlflow_db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5001
    depends_on:
      - postgres
    networks:
      - mlops-network

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: gmv-postgres
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_MULTIPLE_DATABASES=mlflow_db,gmv_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - mlops-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: gmv-redis
    ports:
      - "6379:6379"
    networks:
      - mlops-network

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: gmv-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - mlops-network

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: gmv-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - mlops-network

volumes:
  postgres-data:
  mlflow-artifacts:
  prometheus-data:
  grafana-data:

networks:
  mlops-network:
    driver: bridge

# Usage:
# docker compose -f deployment/docker-compose.yml up -d
# Access API: http://localhost:8000
# Access MLflow: http://localhost:5001
# Access Grafana: http://localhost:3000`
		},
		{
			name: 'Configuration Management',
			filename: 'config/config.yaml',
			language: 'yaml',
			code: `# ==================================================
# GMV Forecasting MLOps Configuration
# Central configuration for all pipeline components
# ==================================================

# Data Configuration
data:
  raw_path: "data/raw/gmv_data.csv"
  processed_path: "data/processed/"
  train_test_split: 0.8
  date_column: "date"
  target_column: "gmv"

# Cities to train models for
cities:
  - hanoi
  - hcmc
  - danang
  - haiphong
  - cantho
  - nhatrang
  - dalat
  - vungtau
  - halong
  - hoian

# Feature Engineering
features:
  exogenous:
    - temperature
    - rainfall
    - is_holiday
    - day_of_week
    - month
    - is_weekend
  scaling: "standard"
  selection_method: "kbest"
  max_features: 10

# SARIMAX Model Configuration
sarimax:
  seasonal: true
  m_candidates: [12, 26, 52]
  max_p: 5
  max_q: 5
  max_P: 5
  max_Q: 5
  information_criterion: "aicc"
  stepwise: true

# Prophet Model Configuration
prophet:
  seasonality_mode: "multiplicative"
  interval_width: 0.95
  country_holidays: "VN"
  custom_seasonalities:
    - name: "monthly"
      period: 30.5
      fourier_order: 5
    - name: "quarterly"
      period: 91.25
      fourier_order: 5

# Hybrid Model Configuration
hybrid:
  optimization_method: "SLSQP"
  error_metric: "mae"
  log_transform: true

# Evaluation Metrics
metrics:
  - mape
  - mae
  - rmse
  - aicc
  - r2_score

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5001"
  experiment_name: "gmv-forecasting-production"
  artifact_location: "mlruns/"

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "info"

# Model Storage
models:
  save_path: "models/"
  format: "pickle"
  versioning: true`
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
			name: 'Model Factory with Multiple Architectures',
			filename: 'src/models/model_factory.py',
			language: 'python',
			code: `"""
Model Factory for Animal Classification MLOps
Supports ResNet-50, CNNDualConv, and CNNSingleConv architectures
"""

import torch
import torch.nn as nn
from src.models.resnet_model import ResNetClassifier
from src.models.cnn_models import CNNDualConv, CNNSingleConv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_model(config: dict) -> nn.Module:
    """
    Factory function to create model based on configuration
    
    Args:
        config: Configuration dictionary with model architecture settings
        
    Returns:
        PyTorch model ready for training
    """
    architecture = config['model']['architecture']
    num_classes = config['data']['num_classes']
    dropout_rate = config['model'].get('dropout_rate', 0.5)
    pretrained = config['model'].get('pretrained', True)
    
    logger.info(f"Creating model: {architecture}")
    logger.info(f"Number of classes: {num_classes}")
    
    if architecture == 'resnet50':
        model = ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=config['model'].get('freeze_backbone', False),
            dropout_rate=dropout_rate
        )
        logger.info(f"ResNet-50 created (pretrained={pretrained})")
        
    elif architecture == 'cnn_dual':
        model = CNNDualConv(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        logger.info("CNNDualConv created (VGG-style)")
        
    elif architecture == 'cnn_single':
        model = CNNSingleConv(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        logger.info("CNNSingleConv created (Lightweight)")
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

# Example usage:
# config = load_config('configs/config.yaml')
# model = create_model(config)  # Returns ResNet-50 by default`
		},
		{
			name: 'Training Pipeline with MLflow',
			filename: 'src/training/trainer.py',
			language: 'python',
			code: `"""
Production Training Pipeline with MLflow Experiment Tracking
Supports early stopping, learning rate scheduling, and comprehensive logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from pathlib import Path
from src.training.early_stopping import EarlyStopping
from src.utils.metrics import calculate_metrics

class Trainer:
    """PyTorch trainer with MLflow integration"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=config['training']['early_stopping']['patience'],
                min_delta=config['training']['early_stopping']['min_delta'],
                mode='min'
            )
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop with MLflow tracking"""
        
        # Start MLflow run
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'architecture': self.config['model']['architecture'],
                'num_classes': self.config['data']['num_classes'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['learning_rate'],
                'epochs': epochs,
                'optimizer': self.config['training']['optimizer']['type']
            })
            
            best_val_loss = float('inf')
            history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
            
            for epoch in range(epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_acc, val_metrics = self._validate(val_loader)
                
                # Log to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall']
                }, step=epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model('best_model.pth')
                    mlflow.pytorch.log_model(self.model, "model")
                
                # Early stopping check
                if self.early_stopping:
                    self.early_stopping(val_loss)
                    if self.early_stopping.should_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            return history
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)`
		},
		{
			name: 'FastAPI Production Deployment',
			filename: 'api/main.py',
			language: 'python',
			code: `"""
FastAPI REST API for Animal Classification
Provides prediction endpoints with health checks and batch support
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import torch
from PIL import Image
import io

from src.evaluation.predictor import Predictor
from src.utils.config import load_config

app = FastAPI(
    title="Animal Classification API",
    description="Production ML API for multiclass animal image classification",
    version="1.0.0"
)

# Global variables
predictor = None
config = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    top_predictions: List[Dict[str, float]]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor, config
    
    try:
        config = load_config("configs/config.yaml")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = config["api"]["model_path"]
        
        predictor = Predictor.load_predictor(model_path, config, device)
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        predictor = None

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Animal Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "classes": "/classes"
        },
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict animal class from uploaded image
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        Prediction with confidence scores
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predictor.predict_single(image, top_k=3)
        
        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            top_predictions=result['top_predictions']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Batch prediction for multiple images"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = predictor.predict_single(image)
        results.append({
            "filename": file.filename,
            "prediction": result
        })
    
    return {"predictions": results, "count": len(results)}

@app.get("/classes")
async def get_classes():
    """Get supported animal classes"""
    if config:
        return {"classes": config['data']['class_names']}
    return {"classes": []}`
		},
		{
			name: 'Docker Compose MLOps Stack',
			filename: 'docker-compose.yml',
			language: 'yaml',
			code: `# ==================================================
# Animal Classification MLOps - Docker Compose
# Services: FastAPI + MLflow
# ==================================================

version: '3.8'

services:
  # FastAPI Application
  api:
    build: .
    container_name: animal-classification-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data/predictions:/app/data/predictions
    environment:
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - animal-classification-network

  # MLflow Tracking Server
  mlflow:
    image: python:3.9-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./mlartifacts:/mlflow/mlartifacts
    command: >
      sh -c "pip install mlflow &&
      mlflow server
      --backend-store-uri file:///mlflow/mlruns
      --default-artifact-root file:///mlflow/mlartifacts
      --host 0.0.0.0
      --port 5000"
    restart: unless-stopped
    networks:
      - animal-classification-network

networks:
  animal-classification-network:
    name: animal-classification-network
    driver: bridge

# Usage:
# docker-compose up -d              # Start all services
# docker-compose logs -f api        # View API logs
# docker-compose down               # Stop all services
#
# Access:
# - API:        http://localhost:8000
# - API Docs:   http://localhost:8000/docs
# - MLflow UI:  http://localhost:5000`
		},
		{
			name: 'Configuration Management',
			filename: 'configs/config.yaml',
			language: 'yaml',
			code: `# ==================================================
# Animal Classification MLOps Configuration
# Central configuration for training, evaluation, and deployment
# ==================================================

# Project settings
project:
  name: "animal-classification"
  description: "Multiclass animal image classification using CNNs"
  version: "1.0.0"
  random_seed: 42

# Data settings
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  dataset_name: "animal-image-dataset-90-different-animals"
  num_classes: 10  # First 10 classes
  class_names:
    - "antelope"
    - "badger"
    - "bat"
    - "bear"
    - "bee"
    - "beetle"
    - "bison"
    - "boar"
    - "butterfly"
    - "cat"
  test_size: 0.2
  val_size: 0.2
  image_size: [224, 224]
  
# Data augmentation
augmentation:
  train:
    horizontal_flip: true
    rotation_degrees: 10
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
    normalize: true
  test:
    normalize: true

# Model settings
model:
  architecture: "resnet50"  # Options: resnet50, cnn_dual, cnn_single
  pretrained: true
  freeze_backbone: false
  dropout_rate: 0.5

# Training settings
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 4
  
  # Optimizer
  optimizer:
    type: "adam"
    betas: [0.9, 0.999]
    eps: 1.0e-08
  
  # Learning rate scheduler
  scheduler:
    type: "reduce_on_plateau"
    mode: "min"
    factor: 0.5
    patience: 5
    min_lr: 1.0e-07
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
    monitor: "val_loss"

# MLflow settings
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "animal-classification"
  artifact_location: "mlartifacts"
  tags:
    project: "animal-classification-mlops"
    team: "ml-team"

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  model_path: "models/best_model.pth"
  allowed_extensions: [".jpg", ".jpeg", ".png"]
  max_image_size: 10485760  # 10MB

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_file: "logs/training.log"`
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

