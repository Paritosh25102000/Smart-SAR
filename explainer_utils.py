"""
Explainability Utilities for Smart-SAR Investigator
SHAP-based model explanations for fraud detection transparency.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from typing import Dict, Any, List, Tuple, Optional

from model_utils import FEATURE_NAMES, prepare_features


# Human-readable feature name mapping
FEATURE_DISPLAY_NAMES = {
    'transaction_amount': 'Transaction Amount',
    'merchant_category_encoded': 'Merchant Category',
    'hour_of_day': 'Hour of Day',
    'day_of_week': 'Day of Week',
    'distance_from_home_km': 'Distance from Home',
    'is_international': 'International Transaction',
    'transaction_frequency_24h': 'Transaction Frequency (24h)',
    'avg_transaction_amount_30d': 'Avg Amount (30-day)',
    'amount_deviation_ratio': 'Amount Deviation Ratio',
    'account_age_days': 'Account Age (days)',
    'is_weekend': 'Weekend Transaction',
    'is_night_transaction': 'Night Transaction',
    'merchant_risk_score': 'Merchant Risk Score',
    'velocity_score': 'Velocity Score'
}


def create_shap_explainer(model_dict: Dict[str, Any], X_background: np.ndarray = None) -> shap.Explainer:
    """
    Create a SHAP explainer for the fraud detection model.

    Args:
        model_dict: Dictionary containing the trained model
        X_background: Background data for SHAP (optional)

    Returns:
        SHAP Explainer object
    """
    model = model_dict['model']

    # Use TreeExplainer for tree-based models (XGBoost, Random Forest)
    if hasattr(model, 'get_booster') or hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer for other models
        if X_background is None:
            raise ValueError("X_background required for non-tree models")
        explainer = shap.KernelExplainer(model.predict_proba, X_background)

    return explainer


def get_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
    check_additivity: bool = False
) -> np.ndarray:
    """
    Calculate SHAP values for given instances.

    Args:
        explainer: SHAP Explainer object
        X: Feature matrix for explanation
        check_additivity: Whether to check SHAP additivity

    Returns:
        SHAP values array
    """
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # For binary classification, we want the positive class (fraud)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Shape is (n_samples, n_features, n_classes) - take class 1
        shap_values = shap_values[:, :, 1]

    return shap_values


def explain_transaction(
    transaction: pd.Series,
    model_dict: Dict[str, Any],
    explainer: shap.Explainer
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for a single transaction.

    Args:
        transaction: Single transaction row
        model_dict: Model dictionary with preprocessors
        explainer: SHAP explainer

    Returns:
        Dictionary with explanation details
    """
    # Prepare single transaction
    df = pd.DataFrame([transaction])

    # Add merchant_category_encoded
    df['merchant_category_encoded'] = model_dict['label_encoder'].transform(df['merchant_category'])

    # Ensure derived features exist (in case they're missing)
    if 'amount_deviation_ratio' not in df.columns:
        df['amount_deviation_ratio'] = df['transaction_amount'] / (df['avg_transaction_amount_30d'] + 1)
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    if 'is_night_transaction' not in df.columns:
        df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
    if 'merchant_risk_score' not in df.columns:
        from model_utils import MERCHANT_RISK_WEIGHTS
        df['merchant_risk_score'] = df['merchant_category'].map(MERCHANT_RISK_WEIGHTS)
    if 'velocity_score' not in df.columns:
        df['velocity_score'] = (
            df['transaction_frequency_24h'] *
            np.log1p(df['transaction_amount']) /
            (np.log1p(df['account_age_days']) + 1)
        )

    X, _, _, _ = prepare_features(
        df,
        label_encoder=model_dict['label_encoder'],
        scaler=model_dict['scaler'],
        fit=False
    )

    # Get SHAP values
    shap_values = get_shap_values(explainer, X)

    # Get feature values (unscaled for interpretability)
    feature_values = df[FEATURE_NAMES].iloc[0].to_dict()

    # Get base value (handle different formats)
    if isinstance(explainer.expected_value, np.ndarray):
        if len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]  # Positive class
        else:
            base_value = explainer.expected_value[0]
    elif isinstance(explainer.expected_value, (list, tuple)):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = float(explainer.expected_value)

    # Create explanation dictionary
    explanation = {
        'shap_values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
        'feature_names': FEATURE_NAMES,
        'feature_values': feature_values,
        'base_value': base_value,
        'X': X[0]
    }

    return explanation


def get_top_contributing_features(
    explanation: Dict[str, Any],
    n_features: int = 5
) -> List[Dict[str, Any]]:
    """
    Get the top features contributing to the fraud prediction.

    Args:
        explanation: SHAP explanation dictionary
        n_features: Number of top features to return

    Returns:
        List of dictionaries with feature contributions
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']
    feature_values = explanation['feature_values']

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1][:n_features]

    contributions = []
    for idx in indices:
        feature_name = feature_names[idx]
        shap_value = shap_values[idx]
        raw_value = feature_values[feature_name]

        # Determine direction
        direction = 'increases' if shap_value > 0 else 'decreases'

        contributions.append({
            'feature': feature_name,
            'display_name': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
            'shap_value': float(shap_value),
            'raw_value': raw_value,
            'direction': direction,
            'impact': 'high' if abs(shap_value) > 0.5 else 'medium' if abs(shap_value) > 0.2 else 'low'
        })

    return contributions


def generate_red_flags(
    explanation: Dict[str, Any],
    transaction: pd.Series,
    threshold: float = 0.1
) -> List[Dict[str, str]]:
    """
    Generate human-readable red flags from SHAP explanation.

    Args:
        explanation: SHAP explanation dictionary
        transaction: Original transaction data
        threshold: Minimum SHAP value to consider as a red flag

    Returns:
        List of red flag dictionaries
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']

    red_flags = []

    for idx, (shap_val, feature_name) in enumerate(zip(shap_values, feature_names)):
        if shap_val > threshold:  # Positive contribution to fraud
            red_flag = _generate_red_flag_message(feature_name, shap_val, transaction)
            if red_flag:
                red_flags.append(red_flag)

    # Sort by severity
    red_flags.sort(key=lambda x: x['severity_score'], reverse=True)

    return red_flags


def _generate_red_flag_message(
    feature_name: str,
    shap_value: float,
    transaction: pd.Series
) -> Optional[Dict[str, str]]:
    """Generate a specific red flag message based on feature."""

    severity = 'HIGH' if shap_value > 0.5 else 'MEDIUM' if shap_value > 0.2 else 'LOW'
    severity_score = shap_value

    messages = {
        'transaction_amount': {
            'flag': f"Unusual Transaction Amount: ${transaction['transaction_amount']:,.2f}",
            'detail': f"This amount deviates significantly from the account's typical transaction pattern. "
                     f"The average 30-day transaction is ${transaction.get('avg_transaction_amount_30d', 0):,.2f}."
        },
        'distance_from_home_km': {
            'flag': f"Geographic Anomaly: {transaction['distance_from_home_km']:.1f} km from home",
            'detail': "Transaction originated from an unusual geographic location, "
                     "significantly distant from the cardholder's registered address."
        },
        'is_international': {
            'flag': "International Transaction Flag",
            'detail': "Cross-border transaction detected. International transactions carry elevated risk "
                     "for unauthorized access, especially without prior travel notification."
        },
        'is_night_transaction': {
            'flag': f"Off-Hours Transaction: {int(transaction['hour_of_day']):02d}:00",
            'detail': "Transaction occurred during high-risk hours (late night/early morning) "
                     "when legitimate activity is statistically less common."
        },
        'transaction_frequency_24h': {
            'flag': f"Velocity Alert: {int(transaction['transaction_frequency_24h'])} transactions in 24h",
            'detail': "Abnormally high transaction frequency detected, potentially indicating "
                     "rapid-fire fraudulent charges or card testing behavior."
        },
        'merchant_risk_score': {
            'flag': f"High-Risk Merchant Category: {transaction['merchant_category']}",
            'detail': f"Transaction at a merchant category with elevated fraud risk. "
                     f"Category risk score: {transaction['merchant_risk_score']:.2f}"
        },
        'amount_deviation_ratio': {
            'flag': f"Amount Deviation: {transaction['amount_deviation_ratio']:.1f}x typical spending",
            'detail': "Transaction amount significantly exceeds the customer's established spending pattern, "
                     "a common indicator of account compromise."
        },
        'velocity_score': {
            'flag': "High Velocity Score Detected",
            'detail': "Combined metric of transaction frequency and amounts indicates potential "
                     "rapid exploitation of compromised credentials."
        },
        'account_age_days': {
            'flag': f"New Account Risk: {int(transaction['account_age_days'])} days old",
            'detail': "Account has limited history, making behavioral analysis less reliable. "
                     "New accounts are frequently targeted in synthetic identity fraud."
        }
    }

    if feature_name in messages:
        msg = messages[feature_name]
        return {
            'flag': msg['flag'],
            'detail': msg['detail'],
            'severity': severity,
            'severity_score': severity_score,
            'feature': feature_name
        }

    return None


def create_force_plot(
    explanation: Dict[str, Any],
    max_display: int = 10
) -> str:
    """
    Create a SHAP force plot and return as base64 encoded image.

    Args:
        explanation: SHAP explanation dictionary
        max_display: Maximum features to display

    Returns:
        Base64 encoded PNG image string
    """
    plt.figure(figsize=(12, 3))

    # Create force plot
    base_value = explanation['base_value']
    shap_values = explanation['shap_values']
    feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in explanation['feature_names']]

    # Use matplotlib-based force plot
    shap.force_plot(
        base_value,
        shap_values,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def create_waterfall_plot(
    explanation: Dict[str, Any],
    max_display: int = 10
) -> str:
    """
    Create a SHAP waterfall plot and return as base64 encoded image.

    Args:
        explanation: SHAP explanation dictionary
        max_display: Maximum features to display

    Returns:
        Base64 encoded PNG image string
    """
    plt.figure(figsize=(10, 6))

    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=explanation['shap_values'],
        base_values=explanation['base_value'],
        data=explanation['X'],
        feature_names=[FEATURE_DISPLAY_NAMES.get(f, f) for f in explanation['feature_names']]
    )

    shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def create_summary_plot(
    model_dict: Dict[str, Any],
    df: pd.DataFrame,
    explainer: shap.Explainer,
    max_display: int = 10
) -> str:
    """
    Create a SHAP summary plot for all transactions.

    Args:
        model_dict: Model dictionary
        df: DataFrame with transactions
        explainer: SHAP explainer
        max_display: Maximum features to display

    Returns:
        Base64 encoded PNG image string
    """
    # Prepare features
    X, _, _, _ = prepare_features(
        df,
        label_encoder=model_dict['label_encoder'],
        scaler=model_dict['scaler'],
        fit=False
    )

    # Get SHAP values
    shap_values = get_shap_values(explainer, X)

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=[FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
        max_display=max_display,
        show=False
    )

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def create_bar_plot(
    explanation: Dict[str, Any],
    max_display: int = 10
) -> str:
    """
    Create a horizontal bar plot showing feature contributions.

    Args:
        explanation: SHAP explanation dictionary
        max_display: Maximum features to display

    Returns:
        Base64 encoded PNG image string
    """
    shap_values = explanation['shap_values']
    feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in explanation['feature_names']]

    # Sort by absolute value
    indices = np.argsort(np.abs(shap_values))[::-1][:max_display]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#ff4444' if v > 0 else '#44aa44' for v in shap_values[indices]]

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, shap_values[indices], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('SHAP Value (Impact on Fraud Probability)')
    ax.set_title('Feature Contributions to Fraud Score')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4444', alpha=0.8, label='Increases Fraud Risk'),
        Patch(facecolor='#44aa44', alpha=0.8, label='Decreases Fraud Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def get_explanation_summary(
    explanation: Dict[str, Any],
    transaction: pd.Series,
    n_features: int = 5
) -> Dict[str, Any]:
    """
    Get a comprehensive explanation summary for a transaction.

    Args:
        explanation: SHAP explanation dictionary
        transaction: Original transaction data
        n_features: Number of top features to include

    Returns:
        Dictionary with explanation summary
    """
    top_features = get_top_contributing_features(explanation, n_features)
    red_flags = generate_red_flags(explanation, transaction)

    return {
        'base_value': float(explanation['base_value']),
        'prediction_contribution': float(np.sum(explanation['shap_values'])),
        'top_features': top_features,
        'red_flags': red_flags,
        'total_red_flags': len(red_flags),
        'high_severity_count': len([rf for rf in red_flags if rf['severity'] == 'HIGH']),
        'medium_severity_count': len([rf for rf in red_flags if rf['severity'] == 'MEDIUM'])
    }


if __name__ == '__main__':
    # Test explainer utilities
    from model_utils import generate_synthetic_fraud_data, train_fraud_model, predict_fraud

    print("Generating test data...")
    df = generate_synthetic_fraud_data(n_samples=1000, fraud_ratio=0.05)

    print("Training model...")
    model_dict = train_fraud_model(df, model_type='xgboost')

    print("Creating SHAP explainer...")
    explainer = create_shap_explainer(model_dict)

    # Get a high-risk transaction
    predictions = predict_fraud(df, model_dict)
    high_risk = predictions[predictions['risk_score'] > 70].iloc[0] if len(predictions[predictions['risk_score'] > 70]) > 0 else predictions.iloc[0]

    print(f"\nExplaining transaction: {high_risk.get('transaction_id', 'N/A')}")
    print(f"Risk Score: {high_risk.get('risk_score', 0):.1f}%")

    explanation = explain_transaction(high_risk, model_dict, explainer)
    summary = get_explanation_summary(explanation, high_risk)

    print(f"\nRed Flags Detected: {summary['total_red_flags']}")
    for rf in summary['red_flags'][:3]:
        print(f"  [{rf['severity']}] {rf['flag']}")

    print("\nTop Contributing Features:")
    for feat in summary['top_features'][:3]:
        print(f"  {feat['display_name']}: {feat['shap_value']:.3f} ({feat['direction']} fraud risk)")
