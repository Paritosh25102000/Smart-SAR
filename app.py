"""
Smart-SAR Investigator
AI-Powered Fraud Detection & Suspicious Activity Report Generation

A Streamlit application for financial crime investigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64
from datetime import datetime
import os

# Local imports
from model_utils import (
    generate_synthetic_fraud_data,
    train_fraud_model,
    predict_fraud,
    save_model,
    load_model,
    get_feature_importance,
    get_transaction_metadata,
    MERCHANT_CATEGORIES
)
from explainer_utils import (
    create_shap_explainer,
    explain_transaction,
    get_explanation_summary,
    create_waterfall_plot,
    create_bar_plot,
    create_summary_plot,
    FEATURE_DISPLAY_NAMES
)
from report_gen import (
    generate_full_sar_report,
    check_api_availability
)


# Page configuration
st.set_page_config(
    page_title="Smart-SAR Investigator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .risk-critical {
        background-color: #ffebee;
        border-left: 4px solid #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-high {
        background-color: #fff3e0;
        border-left: 4px solid #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .red-flag-high {
        background-color: #ffcdd2;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .red-flag-medium {
        background-color: #ffe0b2;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_dict' not in st.session_state:
        st.session_state.model_dict = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    if 'selected_transaction' not in st.session_state:
        st.session_state.selected_transaction = None
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    if 'current_red_flags' not in st.session_state:
        st.session_state.current_red_flags = None
    if 'current_metadata' not in st.session_state:
        st.session_state.current_metadata = None


def load_or_train_model():
    """Load existing model or train a new one."""
    model_path = Path('models/fraud_model.joblib')

    if model_path.exists():
        try:
            return load_model(str(model_path))
        except Exception:
            pass

    # Generate data and train model
    with st.spinner("Training fraud detection model with SMOTE..."):
        data = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.02)
        model_dict = train_fraud_model(data, model_type='xgboost', use_smote=True)
        save_model(model_dict, str(model_path))
        return model_dict


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### Data Generation")
        n_samples = st.slider("Number of Transactions", 1000, 50000, 10000, 1000)
        fraud_ratio = st.slider("Fraud Ratio (%)", 1, 10, 2) / 100

        if st.button("🔄 Generate New Data", type="primary"):
            with st.spinner("Generating synthetic transaction data..."):
                st.session_state.data = generate_synthetic_fraud_data(
                    n_samples=n_samples,
                    fraud_ratio=fraud_ratio
                )
                st.session_state.predictions = None
                st.session_state.selected_transaction = None
            st.success(f"Generated {n_samples:,} transactions")

        st.markdown("---")

        st.markdown("### Model Training")
        model_type = st.selectbox("Model Type", ["XGBoost + SMOTE", "Balanced Random Forest"])
        use_smote = "SMOTE" in model_type

        if st.button("🧠 Train Model"):
            if st.session_state.data is None:
                st.error("Please generate data first!")
            else:
                with st.spinner("Training model..."):
                    model_type_param = 'xgboost' if 'XGBoost' in model_type else 'balanced_rf'
                    st.session_state.model_dict = train_fraud_model(
                        st.session_state.data,
                        model_type=model_type_param,
                        use_smote=use_smote
                    )
                    st.session_state.explainer = create_shap_explainer(st.session_state.model_dict)
                    save_model(st.session_state.model_dict)
                st.success("Model trained successfully!")

        st.markdown("---")

        st.markdown("### Risk Thresholds")
        critical_threshold = st.slider("Critical Risk (%)", 80, 99, 90)
        high_threshold = st.slider("High Risk (%)", 60, 79, 70)

        st.markdown("---")

        st.markdown("### SAR Generation")
        api_status = check_api_availability()

        if api_status['anthropic']:
            st.success("✅ Anthropic API Available")
        else:
            st.warning("⚠️ Anthropic API Not Configured")

        if api_status['openai']:
            st.success("✅ OpenAI API Available")
        else:
            st.warning("⚠️ OpenAI API Not Configured")

        if api_status['gemini']:
            st.success("✅ Gemini API Available")
        else:
            st.warning("⚠️ Gemini API Not Configured")

        llm_provider = st.selectbox(
            "LLM Provider",
            ["Template Only", "Anthropic (Claude)", "OpenAI (GPT-4)", "Google (Gemini)"]
        )

        return {
            'critical_threshold': critical_threshold,
            'high_threshold': high_threshold,
            'llm_provider': llm_provider
        }


def render_overview_dashboard(predictions: pd.DataFrame, config: dict):
    """Render the main overview dashboard."""
    st.markdown("## 📊 Transaction Risk Overview")

    # Calculate metrics
    total_transactions = len(predictions)
    critical_count = len(predictions[predictions['risk_score'] >= config['critical_threshold']])
    high_count = len(predictions[(predictions['risk_score'] >= config['high_threshold']) &
                                  (predictions['risk_score'] < config['critical_threshold'])])
    flagged_count = len(predictions[predictions['fraud_prediction'] == 1])

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")

    with col2:
        st.metric("Critical Risk", f"{critical_count:,}",
                  delta=f"{critical_count/total_transactions*100:.1f}%")

    with col3:
        st.metric("High Risk", f"{high_count:,}",
                  delta=f"{high_count/total_transactions*100:.1f}%")

    with col4:
        st.metric("Flagged for Review", f"{flagged_count:,}",
                  delta=f"{flagged_count/total_transactions*100:.1f}%")

    # Risk distribution chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Score Distribution")
        fig = px.histogram(
            predictions,
            x='risk_score',
            nbins=50,
            color_discrete_sequence=['#1f4e79'],
            labels={'risk_score': 'Risk Score (%)'}
        )
        fig.add_vline(x=config['critical_threshold'], line_dash="dash", line_color="red",
                      annotation_text="Critical")
        fig.add_vline(x=config['high_threshold'], line_dash="dash", line_color="orange",
                      annotation_text="High")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Risk Category Breakdown")
        risk_counts = predictions['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'Critical': '#c62828',
                'High': '#ef6c00',
                'Medium': '#f9a825',
                'Low': '#2e7d32'
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if st.session_state.model_dict:
        st.markdown("### Model Feature Importance")
        importance_df = get_feature_importance(st.session_state.model_dict)
        importance_df['feature_display'] = importance_df['feature'].map(
            lambda x: FEATURE_DISPLAY_NAMES.get(x, x)
        )

        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature_display',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=400,
            xaxis_title="Importance Score",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)


def render_high_risk_table(predictions: pd.DataFrame, config: dict):
    """Render the high-risk transactions table."""
    st.markdown("## 🚨 High Risk Transaction Queue")

    # Filter high-risk transactions
    high_risk = predictions[predictions['risk_score'] >= config['high_threshold']].copy()
    high_risk = high_risk.sort_values('risk_score', ascending=False)

    if len(high_risk) == 0:
        st.info("No high-risk transactions detected in current dataset.")
        return

    st.markdown(f"**{len(high_risk)} transactions** require investigation")

    # Create display dataframe
    display_cols = [
        'transaction_id', 'timestamp', 'transaction_amount', 'merchant_category',
        'is_international', 'risk_score', 'risk_category'
    ]

    display_df = high_risk[display_cols].copy()
    display_df['transaction_amount'] = display_df['transaction_amount'].apply(lambda x: f"${x:,.2f}")
    display_df['is_international'] = display_df['is_international'].apply(lambda x: "Yes" if x else "No")
    display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1f}%")

    display_df.columns = [
        'Transaction ID', 'Timestamp', 'Amount', 'Merchant Category',
        'International', 'Risk Score', 'Risk Category'
    ]

    # Color code rows based on risk
    def highlight_risk(row):
        if row['Risk Category'] == 'Critical':
            return ['background-color: #ffebee'] * len(row)
        elif row['Risk Category'] == 'High':
            return ['background-color: #fff3e0'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display_df.head(50).style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=400
    )

    # Transaction selection
    st.markdown("### Select Transaction for Investigation")
    selected_id = st.selectbox(
        "Transaction ID",
        options=high_risk['transaction_id'].tolist()
    )

    if st.button("🔍 Investigate Transaction", type="primary"):
        st.session_state.selected_transaction = high_risk[
            high_risk['transaction_id'] == selected_id
        ].iloc[0]
        st.success(f"Transaction {selected_id} selected. Go to the **Investigation** tab to view details.")
        st.rerun()


def render_transaction_investigation(config: dict):
    """Render the transaction investigation view."""
    if st.session_state.selected_transaction is None:
        st.info("Select a transaction from the High Risk Queue to investigate.")
        return

    transaction = st.session_state.selected_transaction

    st.markdown("## 🔬 Transaction Investigation")

    # Transaction header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"### Transaction: {transaction['transaction_id']}")

    with col2:
        risk_score = transaction['risk_score']
        risk_class = 'critical' if risk_score >= config['critical_threshold'] else \
                     'high' if risk_score >= config['high_threshold'] else \
                     'medium' if risk_score >= 30 else 'low'
        st.markdown(f"""
        <div class="risk-{risk_class}">
            <strong>Risk Score</strong><br>
            <span style="font-size: 2rem;">{risk_score:.1f}%</span><br>
            {transaction['risk_category']}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Amount</strong><br>
            <span style="font-size: 1.5rem;">${transaction['transaction_amount']:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Transaction details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Transaction Metadata")
        metadata = get_transaction_metadata(transaction)

        details_df = pd.DataFrame([
            {"Field": "Timestamp", "Value": metadata['timestamp']},
            {"Field": "Merchant Category", "Value": metadata['merchant_category']},
            {"Field": "Location Type", "Value": metadata['location']},
            {"Field": "Distance from Home", "Value": metadata['distance_from_home']},
            {"Field": "Time of Day", "Value": metadata['time_of_day']},
            {"Field": "Day of Week", "Value": metadata['day_of_week']},
            {"Field": "Transaction Frequency (24h)", "Value": str(metadata['transaction_frequency_24h'])},
            {"Field": "Account Age", "Value": f"{metadata['account_age_days']} days"},
            {"Field": "Amount Deviation", "Value": metadata['amount_deviation_ratio']},
            {"Field": "Merchant Risk Score", "Value": metadata['merchant_risk_score']}
        ])
        st.dataframe(details_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Entity Risk Profile")

        # Risk indicators
        indicators = []
        if transaction['is_international']:
            indicators.append(("🌍 International Transaction", "HIGH"))
        if transaction['is_night_transaction']:
            indicators.append(("🌙 Night Transaction", "MEDIUM"))
        if transaction['distance_from_home_km'] > 500:
            indicators.append(("📍 Geographic Anomaly", "HIGH"))
        if transaction['transaction_frequency_24h'] > 10:
            indicators.append(("⚡ High Velocity", "HIGH"))
        if transaction['amount_deviation_ratio'] > 3:
            indicators.append(("💰 Amount Deviation", "HIGH"))
        if transaction['account_age_days'] < 90:
            indicators.append(("🆕 New Account", "MEDIUM"))

        if indicators:
            for indicator, severity in indicators:
                color = '#ffcdd2' if severity == 'HIGH' else '#ffe0b2'
                st.markdown(f"""
                <div style="background-color: {color}; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.5rem;">
                    {indicator} <span style="float: right; font-weight: bold;">{severity}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No elevated risk indicators")

    # SHAP Explanation
    st.markdown("---")
    st.markdown("### 🧠 AI Explainability Analysis (SHAP)")

    if st.session_state.explainer is None:
        st.warning("Model explainer not initialized. Please train the model first.")
    else:
        with st.spinner("Generating SHAP explanation..."):
            explanation = explain_transaction(
                transaction,
                st.session_state.model_dict,
                st.session_state.explainer
            )
            summary = get_explanation_summary(explanation, transaction)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Feature Contribution Plot")
            try:
                bar_plot_img = create_bar_plot(explanation)
                st.image(f"data:image/png;base64,{bar_plot_img}", use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate plot: {str(e)}")

        with col2:
            st.markdown("#### Top Risk Factors")
            for feat in summary['top_features']:
                direction_icon = "🔴" if feat['direction'] == 'increases' else "🟢"
                impact_badge = f"**{feat['impact'].upper()}**"
                st.markdown(f"""
                {direction_icon} **{feat['display_name']}**
                - SHAP Value: `{feat['shap_value']:+.3f}`
                - Impact: {impact_badge}
                - Direction: {feat['direction']} fraud risk
                """)
                st.markdown("---")

        # Red Flags
        st.markdown("### 🚩 Identified Red Flags")

        red_flags = summary['red_flags']
        if red_flags:
            for rf in red_flags:
                severity_class = 'red-flag-high' if rf['severity'] == 'HIGH' else 'red-flag-medium'
                st.markdown(f"""
                <div class="{severity_class}">
                    <strong>[{rf['severity']}] {rf['flag']}</strong><br>
                    <small>{rf['detail']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant red flags identified")

        # Store for SAR generation
        st.session_state.current_explanation = summary
        st.session_state.current_red_flags = red_flags
        st.session_state.current_metadata = metadata


def render_sar_generation(config: dict):
    """Render the SAR generation section."""
    st.markdown("## 📋 Suspicious Activity Report Generation")

    if st.session_state.selected_transaction is None:
        st.info("Select and investigate a transaction first to generate a SAR.")
        return

    if not hasattr(st.session_state, 'current_explanation') or st.session_state.current_explanation is None:
        st.info("Please view the **Investigation** tab first to analyze the transaction before generating a SAR.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        Generate a professional, regulatory-compliant Suspicious Activity Report (SAR)
        using AI-assisted narrative generation. The report will include:
        - Transaction metadata and risk assessment
        - Identified red flags and entity risk indicators
        - SHAP-based feature analysis
        - AI-generated narrative summary
        - Recommended actions
        """)

    with col2:
        use_llm = config['llm_provider'] != "Template Only"
        if 'Anthropic' in config['llm_provider']:
            provider = 'anthropic'
        elif 'Gemini' in config['llm_provider']:
            provider = 'gemini'
        else:
            provider = 'openai'

        if st.button("📝 Generate SAR", type="primary"):
            with st.spinner("Generating Suspicious Activity Report..."):
                try:
                    report = generate_full_sar_report(
                        transaction_metadata=st.session_state.current_metadata,
                        red_flags=st.session_state.current_red_flags,
                        shap_summary=st.session_state.current_explanation,
                        provider=provider,
                        use_llm=use_llm
                    )
                    st.session_state.generated_report = report
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    # Fallback to template
                    report = generate_full_sar_report(
                        transaction_metadata=st.session_state.current_metadata,
                        red_flags=st.session_state.current_red_flags,
                        shap_summary=st.session_state.current_explanation,
                        provider=provider,
                        use_llm=False
                    )
                    st.session_state.generated_report = report

    # Display generated report
    if st.session_state.generated_report:
        st.markdown("### Generated SAR Report")

        # Download button
        report_bytes = st.session_state.generated_report.encode()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SAR_Report_{st.session_state.selected_transaction['transaction_id']}_{timestamp}.txt"

        st.download_button(
            label="📥 Download SAR Report",
            data=report_bytes,
            file_name=filename,
            mime="text/plain"
        )

        # Display report in expandable section
        with st.expander("View Full Report", expanded=True):
            st.code(st.session_state.generated_report, language=None)


def render_model_performance():
    """Render model performance metrics."""
    st.markdown("## 📈 Model Performance Metrics")

    if st.session_state.model_dict is None:
        st.info("Please train a model to view performance metrics.")
        return

    metrics = st.session_state.model_dict['metrics']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

    with col2:
        st.metric("Average Precision", f"{metrics['average_precision']:.4f}")

    with col3:
        report = metrics['classification_report']
        st.metric("Fraud Recall", f"{report['1']['recall']:.4f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = np.array(metrics['confusion_matrix'])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Legitimate', 'Predicted: Fraud'],
        y=['Actual: Legitimate', 'Actual: Fraud'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text:,}',
        textfont={"size": 16}
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.markdown("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    report_df.index = ['Legitimate', 'Fraud']
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)


def main():
    """Main application entry point."""
    initialize_session_state()

    # Header
    st.markdown('<p class="main-header">🔍 Smart-SAR Investigator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fraud Detection & Suspicious Activity Report Generation</p>',
                unsafe_allow_html=True)

    # Sidebar
    config = render_sidebar()

    # Initialize data and model if needed
    if st.session_state.data is None:
        with st.spinner("Initializing system..."):
            st.session_state.data = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.02)
            st.session_state.model_dict = load_or_train_model()
            st.session_state.explainer = create_shap_explainer(st.session_state.model_dict)

    # Run predictions if needed
    if st.session_state.predictions is None and st.session_state.model_dict is not None:
        with st.spinner("Analyzing transactions..."):
            st.session_state.predictions = predict_fraud(
                st.session_state.data,
                st.session_state.model_dict
            )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview Dashboard",
        "🚨 High Risk Queue",
        "🔬 Investigation",
        "📋 SAR Generation"
    ])

    with tab1:
        if st.session_state.predictions is not None:
            render_overview_dashboard(st.session_state.predictions, config)
            st.markdown("---")
            render_model_performance()

    with tab2:
        if st.session_state.predictions is not None:
            render_high_risk_table(st.session_state.predictions, config)

    with tab3:
        render_transaction_investigation(config)

    with tab4:
        render_sar_generation(config)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Smart-SAR Investigator | AI-Assisted Financial Crime Investigation<br>
        For demonstration purposes only. All data is synthetic.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
