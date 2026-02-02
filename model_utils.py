"""
Model Utilities for Smart-SAR Investigator
Handles synthetic data generation, model training with SMOTE, and inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb


# Feature names for reference
FEATURE_NAMES = [
    'transaction_amount',
    'merchant_category_encoded',
    'hour_of_day',
    'day_of_week',
    'distance_from_home_km',
    'is_international',
    'transaction_frequency_24h',
    'avg_transaction_amount_30d',
    'amount_deviation_ratio',
    'account_age_days',
    'is_weekend',
    'is_night_transaction',
    'merchant_risk_score',
    'velocity_score'
]

MERCHANT_CATEGORIES = [
    'Retail', 'Grocery', 'Restaurant', 'Gas Station', 'Online Shopping',
    'Travel', 'Entertainment', 'Healthcare', 'Utilities', 'Cash Withdrawal',
    'Wire Transfer', 'Gambling', 'Cryptocurrency', 'Jewelry', 'Electronics'
]

# Risk weights for merchant categories (higher = more suspicious)
MERCHANT_RISK_WEIGHTS = {
    'Retail': 0.2, 'Grocery': 0.1, 'Restaurant': 0.15, 'Gas Station': 0.2,
    'Online Shopping': 0.4, 'Travel': 0.35, 'Entertainment': 0.25,
    'Healthcare': 0.1, 'Utilities': 0.05, 'Cash Withdrawal': 0.5,
    'Wire Transfer': 0.6, 'Gambling': 0.7, 'Cryptocurrency': 0.8,
    'Jewelry': 0.55, 'Electronics': 0.45
}


def generate_synthetic_fraud_data(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate high-quality synthetic credit card fraud dataset.

    Args:
        n_samples: Total number of transactions
        fraud_ratio: Proportion of fraudulent transactions
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with transaction data and fraud labels
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud

    # Generate legitimate transactions
    legit_data = _generate_legitimate_transactions(n_legitimate, random_state)

    # Generate fraudulent transactions with distinct patterns
    fraud_data = _generate_fraudulent_transactions(n_fraud, random_state + 1)

    # Combine and shuffle
    df = pd.concat([legit_data, fraud_data], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add transaction IDs
    df['transaction_id'] = [f'TXN-{i:08d}' for i in range(len(df))]

    # Add timestamps
    base_date = datetime(2024, 1, 1)
    df['timestamp'] = [
        base_date + timedelta(
            days=np.random.randint(0, 365),
            hours=row['hour_of_day'],
            minutes=np.random.randint(0, 60)
        )
        for _, row in df.iterrows()
    ]

    # Reorder columns
    cols = ['transaction_id', 'timestamp'] + [c for c in df.columns if c not in ['transaction_id', 'timestamp']]
    df = df[cols]

    return df


def _generate_legitimate_transactions(n: int, seed: int) -> pd.DataFrame:
    """Generate legitimate transaction patterns."""
    np.random.seed(seed)

    data = {
        # Normal transaction amounts (log-normal distribution)
        'transaction_amount': np.exp(np.random.normal(4.5, 1.2, n)),

        # Merchant category (weighted towards common categories)
        'merchant_category': np.random.choice(
            MERCHANT_CATEGORIES,
            n,
            p=[0.15, 0.18, 0.12, 0.10, 0.12, 0.05, 0.08, 0.05, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01]
        ),

        # Time patterns (normal business hours weighted)
        'hour_of_day': np.clip(np.random.normal(14, 4, n), 0, 23).astype(int),
        'day_of_week': np.random.randint(0, 7, n),

        # Distance from home (most transactions are local)
        'distance_from_home_km': np.abs(np.random.exponential(15, n)),

        # International transactions are rare for legitimate
        'is_international': np.random.choice([0, 1], n, p=[0.95, 0.05]),

        # Normal transaction frequency
        'transaction_frequency_24h': np.random.poisson(3, n),

        # Average transaction amount (similar to current)
        'avg_transaction_amount_30d': np.exp(np.random.normal(4.3, 0.8, n)),

        # Account age (established accounts)
        'account_age_days': np.random.exponential(800, n) + 90,

        # Label
        'is_fraud': np.zeros(n, dtype=int)
    }

    df = pd.DataFrame(data)

    # Calculate derived features
    df = _add_derived_features(df)

    return df


def _generate_fraudulent_transactions(n: int, seed: int) -> pd.DataFrame:
    """Generate fraudulent transaction patterns with red flags."""
    np.random.seed(seed)

    data = {
        # Fraudulent amounts tend to be either very small (testing) or very large
        'transaction_amount': np.where(
            np.random.random(n) < 0.3,
            np.random.uniform(0.5, 5, n),  # Small test transactions
            np.exp(np.random.normal(6.5, 1.5, n))  # Large transactions
        ),

        # High-risk merchant categories
        'merchant_category': np.random.choice(
            MERCHANT_CATEGORIES,
            n,
            p=[0.05, 0.02, 0.03, 0.05, 0.15, 0.08, 0.05, 0.02, 0.01, 0.12, 0.15, 0.10, 0.08, 0.05, 0.04]
        ),

        # Unusual hours (late night/early morning)
        'hour_of_day': np.where(
            np.random.random(n) < 0.6,
            np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], n),
            np.random.randint(0, 24, n)
        ),
        'day_of_week': np.random.randint(0, 7, n),

        # Far from home
        'distance_from_home_km': np.abs(np.random.exponential(500, n)) + 50,

        # Higher international rate
        'is_international': np.random.choice([0, 1], n, p=[0.4, 0.6]),

        # Burst of transactions
        'transaction_frequency_24h': np.random.poisson(12, n) + 5,

        # Amount deviation from history
        'avg_transaction_amount_30d': np.exp(np.random.normal(4.0, 0.6, n)),

        # Newer or compromised accounts
        'account_age_days': np.random.exponential(200, n) + 10,

        # Label
        'is_fraud': np.ones(n, dtype=int)
    }

    df = pd.DataFrame(data)

    # Calculate derived features
    df = _add_derived_features(df)

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better fraud detection."""

    # Amount deviation ratio
    df['amount_deviation_ratio'] = df['transaction_amount'] / (df['avg_transaction_amount_30d'] + 1)

    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Is night transaction (10 PM - 6 AM)
    df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)

    # Merchant risk score
    df['merchant_risk_score'] = df['merchant_category'].map(MERCHANT_RISK_WEIGHTS)

    # Velocity score (combination of frequency and amount)
    df['velocity_score'] = (
        df['transaction_frequency_24h'] *
        np.log1p(df['transaction_amount']) /
        (np.log1p(df['account_age_days']) + 1)
    )

    return df


def prepare_features(
    df: pd.DataFrame,
    label_encoder: Optional[LabelEncoder] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    """
    Prepare features for model training/inference.

    Args:
        df: Input DataFrame
        label_encoder: Existing encoder for merchant category
        scaler: Existing scaler for numerical features
        fit: Whether to fit the transformers

    Returns:
        X, y, label_encoder, scaler
    """
    df = df.copy()

    # Encode merchant category
    if label_encoder is None:
        label_encoder = LabelEncoder()

    if fit:
        df['merchant_category_encoded'] = label_encoder.fit_transform(df['merchant_category'])
    else:
        df['merchant_category_encoded'] = label_encoder.transform(df['merchant_category'])

    # Select features
    feature_cols = FEATURE_NAMES
    X = df[feature_cols].values
    y = df['is_fraud'].values if 'is_fraud' in df.columns else None

    # Scale features
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, label_encoder, scaler


def train_fraud_model(
    df: pd.DataFrame,
    model_type: str = 'xgboost',
    use_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a fraud detection model with class imbalance handling.

    Args:
        df: Training DataFrame
        model_type: 'xgboost' or 'balanced_rf'
        use_smote: Whether to apply SMOTE for XGBoost
        test_size: Proportion of data for testing
        random_state: Random seed

    Returns:
        Dictionary containing model, metrics, and preprocessors
    """
    # Prepare features
    X, y, label_encoder, scaler = prepare_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == 'xgboost':
        if use_smote:
            # Apply SMOTE to handle class imbalance
            smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Calculate scale_pos_weight for remaining imbalance
        scale_pos_weight = len(y_train_resampled[y_train_resampled == 0]) / max(len(y_train_resampled[y_train_resampled == 1]), 1)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='auc',
            use_label_encoder=False
        )
        model.fit(X_train_resampled, y_train_resampled)

    elif model_type == 'balanced_rf':
        # Balanced Random Forest handles imbalance internally
        model = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'average_precision': average_precision_score(y_test, y_prob)
    }

    return {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'metrics': metrics,
        'feature_names': FEATURE_NAMES
    }


def save_model(model_dict: Dict[str, Any], path: str = 'models/fraud_model.joblib'):
    """Save trained model and preprocessors."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict, path)
    print(f"Model saved to {path}")


def load_model(path: str = 'models/fraud_model.joblib') -> Dict[str, Any]:
    """Load trained model and preprocessors."""
    return joblib.load(path)


def predict_fraud(
    transactions: pd.DataFrame,
    model_dict: Dict[str, Any]
) -> pd.DataFrame:
    """
    Predict fraud probability for transactions.

    Args:
        transactions: DataFrame with transaction data
        model_dict: Dictionary containing model and preprocessors

    Returns:
        DataFrame with fraud predictions and probabilities
    """
    model = model_dict['model']
    label_encoder = model_dict['label_encoder']
    scaler = model_dict['scaler']

    # Prepare features (don't fit, use existing transformers)
    X, _, _, _ = prepare_features(
        transactions,
        label_encoder=label_encoder,
        scaler=scaler,
        fit=False
    )

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Add results to DataFrame
    result = transactions.copy()
    result['fraud_prediction'] = predictions
    result['fraud_probability'] = probabilities
    result['risk_score'] = (probabilities * 100).round(1)

    # Assign risk category
    result['risk_category'] = pd.cut(
        result['risk_score'],
        bins=[0, 30, 60, 80, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    return result


def get_feature_importance(model_dict: Dict[str, Any]) -> pd.DataFrame:
    """Get feature importance from the trained model."""
    model = model_dict['model']
    feature_names = model_dict['feature_names']

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return df


def get_transaction_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract human-readable metadata from a transaction."""
    return {
        'transaction_id': row.get('transaction_id', 'N/A'),
        'timestamp': str(row.get('timestamp', 'N/A')),
        'amount': f"${row['transaction_amount']:,.2f}",
        'merchant_category': row['merchant_category'],
        'location': 'International' if row['is_international'] else 'Domestic',
        'distance_from_home': f"{row['distance_from_home_km']:.1f} km",
        'time_of_day': f"{int(row['hour_of_day']):02d}:00",
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(row['day_of_week'])],
        'transaction_frequency_24h': int(row['transaction_frequency_24h']),
        'account_age_days': int(row['account_age_days']),
        'amount_deviation_ratio': f"{row['amount_deviation_ratio']:.2f}x",
        'merchant_risk_score': f"{row['merchant_risk_score']:.2f}",
        'risk_score': row.get('risk_score', 0),
        'risk_category': row.get('risk_category', 'Unknown')
    }


if __name__ == '__main__':
    # Test data generation and model training
    print("Generating synthetic fraud data...")
    df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.02)
    print(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraudulent cases")

    print("\nTraining XGBoost model with SMOTE...")
    model_dict = train_fraud_model(df, model_type='xgboost', use_smote=True)

    print("\nModel Metrics:")
    print(f"ROC-AUC: {model_dict['metrics']['roc_auc']:.4f}")
    print(f"Average Precision: {model_dict['metrics']['average_precision']:.4f}")

    print("\nFeature Importance:")
    print(get_feature_importance(model_dict).head(10))

    save_model(model_dict)
