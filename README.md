# Smart-SAR Investigator

AI-Powered Fraud Detection & Suspicious Activity Report (SAR) Generation

## Features

- **Synthetic Fraud Data Generation**: Realistic credit card transaction data with fraud patterns
- **ML Model Training**: XGBoost classifier with SMOTE for handling class imbalance
- **Risk Scoring Dashboard**: Interactive visualization of transaction risk scores
- **Explainability (XAI)**: SHAP-based feature contribution analysis
- **Automated SAR Generation**: LLM-powered regulatory-compliant report generation

## Tech Stack

- **Frontend**: Streamlit
- **ML**: XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **LLM Integration**: Anthropic Claude, OpenAI GPT-4, Google Gemini
- **Visualization**: Plotly, Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Environment Variables

For LLM-powered SAR narrative generation, set one of:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# or
export OPENAI_API_KEY="your-openai-api-key"
```

## Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set secrets in Streamlit Cloud dashboard:
   - Go to App Settings > Secrets
   - Add your API key:
     ```toml
     GEMINI_API_KEY = "your-api-key"
     ```

## Project Structure

```
Smart-SAR/
├── app.py                  # Main Streamlit dashboard
├── model_utils.py          # Data generation & model training
├── explainer_utils.py      # SHAP explainability
├── report_gen.py           # LLM SAR generation
├── requirements.txt        # Dependencies
└── models/                 # Saved models (auto-generated)
```

## Usage

1. **Overview Dashboard**: View risk distribution and model performance
2. **High Risk Queue**: Browse flagged transactions
3. **Investigation**: Analyze selected transaction with SHAP explanations
4. **SAR Generation**: Generate regulatory-compliant reports

## Disclaimer

This application is for demonstration purposes only. All transaction data is synthetic.
Generated SAR reports should be reviewed by qualified compliance officers before regulatory submission.

## License

MIT
