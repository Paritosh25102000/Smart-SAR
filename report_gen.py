"""
Report Generation Utilities for Smart-SAR Investigator
LLM-powered Suspicious Activity Report (SAR) generation.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from string import Template

# LLM clients - imported conditionally
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def get_secret(key: str) -> Optional[str]:
    """Get secret from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variables
    return os.getenv(key)


# SAR Report Template
SAR_TEMPLATE = """
================================================================================
                     SUSPICIOUS ACTIVITY REPORT (SAR)
                          CONFIDENTIAL - INTERNAL USE ONLY
================================================================================

REPORT ID: ${report_id}
GENERATED: ${generation_date}
ANALYST SYSTEM: Smart-SAR Investigator (AI-Assisted)

================================================================================
                              SECTION 1: SUBJECT INFORMATION
================================================================================

Transaction ID:        ${transaction_id}
Transaction Date:      ${transaction_date}
Transaction Amount:    ${transaction_amount}
Merchant Category:     ${merchant_category}
Transaction Location:  ${transaction_location}
Account Age:           ${account_age} days

================================================================================
                              SECTION 2: RISK ASSESSMENT
================================================================================

FRAUD RISK SCORE:      ${risk_score}% (${risk_category})
MODEL CONFIDENCE:      ${model_confidence}

ENTITY RISK INDICATORS:
${entity_risk_indicators}

================================================================================
                              SECTION 3: RED FLAGS IDENTIFIED
================================================================================

${red_flags_section}

================================================================================
                              SECTION 4: TRANSACTION ANALYSIS
================================================================================

${transaction_analysis}

================================================================================
                              SECTION 5: SHAP FEATURE ANALYSIS
================================================================================

The following features contributed most significantly to the fraud risk assessment:

${shap_analysis}

================================================================================
                              SECTION 6: NARRATIVE SUMMARY
================================================================================

${narrative_summary}

================================================================================
                              SECTION 7: RECOMMENDED ACTIONS
================================================================================

${recommended_actions}

================================================================================
                              SECTION 8: COMPLIANCE NOTES
================================================================================

This Suspicious Activity Report was generated using AI-assisted analysis.
All findings should be reviewed by a qualified compliance officer before
submission to regulatory authorities.

Filing Deadline: Within 30 days of detection (per BSA/AML requirements)
Retention Period: 5 years from date of filing

DISCLAIMER: This report is generated for investigative purposes only.
Final determination of suspicious activity requires human review and
institutional judgment.

================================================================================
                                    END OF REPORT
================================================================================
"""


def get_llm_client(provider: str = 'anthropic') -> Any:
    """
    Initialize and return an LLM client.

    Args:
        provider: 'anthropic', 'openai', or 'gemini'

    Returns:
        LLM client object
    """
    if provider == 'anthropic':
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        api_key = get_secret('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        return anthropic.Anthropic(api_key=api_key)

    elif provider == 'openai':
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        api_key = get_secret('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        return OpenAI(api_key=api_key)

    elif provider == 'gemini':
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        api_key = get_secret('GEMINI_API_KEY') or get_secret('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not configured")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')

    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_sar_narrative(
    transaction_metadata: Dict[str, Any],
    red_flags: List[Dict[str, str]],
    shap_analysis: Dict[str, Any],
    provider: str = 'anthropic'
) -> str:
    """
    Generate a professional SAR narrative using an LLM.

    Args:
        transaction_metadata: Transaction details
        red_flags: List of identified red flags
        shap_analysis: SHAP explanation summary
        provider: LLM provider ('anthropic' or 'openai')

    Returns:
        Generated narrative text
    """
    # Build the prompt
    prompt = _build_sar_prompt(transaction_metadata, red_flags, shap_analysis)

    system_prompt = ("You are a senior financial crimes investigator specializing in AML/BSA compliance. "
                     "Generate professional, regulatory-compliant Suspicious Activity Report narratives. "
                     "Use precise financial crime terminology and maintain an objective, factual tone. "
                     "Focus on observable facts and risk indicators without making accusations.")

    try:
        client = get_llm_client(provider)

        if provider == 'anthropic':
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system=system_prompt
            )
            return response.content[0].text

        elif provider == 'openai':
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                max_tokens=2000,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content

        elif provider == 'gemini':
            # Gemini uses a combined prompt approach
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = client.generate_content(full_prompt)
            return response.text

    except Exception as e:
        # Fallback to template-based generation
        return _generate_fallback_narrative(transaction_metadata, red_flags, shap_analysis)


def _build_sar_prompt(
    transaction_metadata: Dict[str, Any],
    red_flags: List[Dict[str, str]],
    shap_analysis: Dict[str, Any]
) -> str:
    """Build the LLM prompt for SAR narrative generation."""

    red_flags_text = "\n".join([
        f"- [{rf['severity']}] {rf['flag']}: {rf['detail']}"
        for rf in red_flags
    ]) if red_flags else "No significant red flags identified."

    top_features_text = "\n".join([
        f"- {feat['display_name']}: SHAP value {feat['shap_value']:.3f} ({feat['direction']} fraud risk)"
        for feat in shap_analysis.get('top_features', [])
    ])

    prompt = f"""
Generate a professional Suspicious Activity Report (SAR) narrative for the following transaction.
Use regulatory-compliant language and financial crime investigation terminology.

TRANSACTION DETAILS:
- Transaction ID: {transaction_metadata.get('transaction_id', 'N/A')}
- Date/Time: {transaction_metadata.get('timestamp', 'N/A')}
- Amount: {transaction_metadata.get('amount', 'N/A')}
- Merchant Category: {transaction_metadata.get('merchant_category', 'N/A')}
- Location Type: {transaction_metadata.get('location', 'N/A')}
- Distance from Home: {transaction_metadata.get('distance_from_home', 'N/A')}
- Time of Transaction: {transaction_metadata.get('time_of_day', 'N/A')}
- Account Age: {transaction_metadata.get('account_age_days', 'N/A')} days

RISK ASSESSMENT:
- Fraud Risk Score: {transaction_metadata.get('risk_score', 0)}%
- Risk Category: {transaction_metadata.get('risk_category', 'Unknown')}

RED FLAGS IDENTIFIED:
{red_flags_text}

KEY RISK FACTORS (from ML Model Analysis):
{top_features_text}

Please generate:
1. A detailed NARRATIVE SUMMARY (2-3 paragraphs) describing the suspicious activity, using terms like "Point of Compromise", "Entity Risk", "Behavioral Anomaly", "Transaction Velocity", etc.

2. RECOMMENDED ACTIONS section with specific next steps for the investigation team.

3. Any additional COMPLIANCE CONSIDERATIONS relevant to this case.

Format the response with clear section headers. Maintain professional, objective tone throughout.
"""
    return prompt


def _generate_fallback_narrative(
    transaction_metadata: Dict[str, Any],
    red_flags: List[Dict[str, str]],
    shap_analysis: Dict[str, Any]
) -> str:
    """Generate a template-based narrative when LLM is unavailable."""

    high_risk_flags = [rf for rf in red_flags if rf['severity'] == 'HIGH']
    medium_risk_flags = [rf for rf in red_flags if rf['severity'] == 'MEDIUM']

    narrative = f"""
NARRATIVE SUMMARY:

This Suspicious Activity Report documents anomalous transaction activity detected through
automated fraud monitoring systems. The transaction in question, ID {transaction_metadata.get('transaction_id', 'N/A')},
was flagged with a risk score of {transaction_metadata.get('risk_score', 0)}% ({transaction_metadata.get('risk_category', 'Unknown')} risk category).

The subject transaction of {transaction_metadata.get('amount', 'N/A')} was initiated at {transaction_metadata.get('time_of_day', 'N/A')}
in the {transaction_metadata.get('merchant_category', 'N/A')} merchant category. Geographic analysis indicates
the transaction originated approximately {transaction_metadata.get('distance_from_home', 'N/A')} from the
cardholder's registered address, representing a potential Point of Compromise indicator.

Entity risk assessment identified {len(high_risk_flags)} high-severity and {len(medium_risk_flags)} medium-severity
red flags. Key behavioral anomalies include transaction velocity concerns, amount deviation from
established spending patterns, and potential geographic displacement inconsistent with known
customer travel patterns.

RECOMMENDED ACTIONS:

1. IMMEDIATE: Place temporary hold on account pending investigation completion
2. VERIFICATION: Attempt customer contact through verified channels to confirm transaction legitimacy
3. ANALYSIS: Review related transactions from past 30 days for pattern analysis
4. ESCALATION: If customer contact unsuccessful within 24 hours, escalate to Senior Fraud Analyst
5. DOCUMENTATION: Preserve all transaction metadata and communication logs for regulatory filing

COMPLIANCE CONSIDERATIONS:

- This transaction meets threshold criteria for SAR filing consideration
- Ensure all BSA/AML documentation requirements are satisfied
- Coordinate with Legal/Compliance for potential law enforcement referral
- Maintain confidentiality per SAR non-disclosure requirements (31 U.S.C. § 5318(g)(2))
"""
    return narrative


def generate_recommended_actions(
    risk_category: str,
    red_flags: List[Dict[str, str]]
) -> str:
    """Generate recommended actions based on risk level."""

    base_actions = [
        "1. Document all findings in case management system",
        "2. Preserve transaction metadata and audit trail"
    ]

    if risk_category in ['Critical', 'High']:
        actions = [
            "1. IMMEDIATE: Implement temporary account restrictions pending investigation",
            "2. PRIORITY: Initiate customer outreach through verified contact channels",
            "3. ANALYSIS: Conduct comprehensive 90-day transaction history review",
            "4. ESCALATION: Route to Senior Financial Crimes Investigator within 4 hours",
            "5. REGULATORY: Prepare SAR filing documentation within 24 hours",
            "6. COORDINATION: Notify relevant internal stakeholders (Risk, Legal, Compliance)"
        ]
    elif risk_category == 'Medium':
        actions = [
            "1. MONITORING: Place account on enhanced monitoring for 30 days",
            "2. VERIFICATION: Schedule customer verification call within 48 hours",
            "3. ANALYSIS: Review transaction patterns for additional anomalies",
            "4. DOCUMENTATION: Update customer risk profile in core systems",
            "5. FOLLOW-UP: Re-assess risk score after verification completion"
        ]
    else:
        actions = base_actions + [
            "3. Continue standard monitoring procedures",
            "4. No immediate escalation required",
            "5. Flag for periodic review during next risk assessment cycle"
        ]

    return "\n".join(actions)


def format_red_flags_section(red_flags: List[Dict[str, str]]) -> str:
    """Format red flags for the SAR report."""

    if not red_flags:
        return "No significant red flags identified through automated analysis."

    sections = []
    for i, rf in enumerate(red_flags, 1):
        section = f"""
RED FLAG #{i}: {rf['flag']}
Severity: {rf['severity']}
Details: {rf['detail']}
"""
        sections.append(section)

    return "\n".join(sections)


def format_shap_analysis(shap_summary: Dict[str, Any]) -> str:
    """Format SHAP analysis for the SAR report."""

    lines = []
    for feat in shap_summary.get('top_features', []):
        direction_symbol = "↑" if feat['direction'] == 'increases' else "↓"
        lines.append(
            f"• {feat['display_name']}: {direction_symbol} fraud risk "
            f"(SHAP: {feat['shap_value']:+.3f}, Impact: {feat['impact'].upper()})"
        )

    return "\n".join(lines) if lines else "No significant feature contributions identified."


def format_entity_risk_indicators(transaction_metadata: Dict[str, Any]) -> str:
    """Format entity risk indicators."""

    indicators = []

    # Geographic risk
    if transaction_metadata.get('location') == 'International':
        indicators.append("• GEOGRAPHIC: International transaction (elevated cross-border risk)")

    # Temporal risk
    time_str = transaction_metadata.get('time_of_day', '12:00')
    hour = int(time_str.split(':')[0]) if ':' in str(time_str) else 12
    if hour < 6 or hour >= 22:
        indicators.append("• TEMPORAL: Off-hours transaction (outside normal business hours)")

    # Velocity risk
    freq = transaction_metadata.get('transaction_frequency_24h', 0)
    if freq > 10:
        indicators.append(f"• VELOCITY: High transaction frequency ({freq} in 24h)")

    # Account age risk
    age = transaction_metadata.get('account_age_days', 365)
    if age < 90:
        indicators.append(f"• ACCOUNT: New account ({age} days - limited behavioral baseline)")

    # Amount deviation
    deviation = transaction_metadata.get('amount_deviation_ratio', '1.0x')
    if isinstance(deviation, str):
        dev_value = float(deviation.replace('x', ''))
    else:
        dev_value = float(deviation)
    if dev_value > 3:
        indicators.append(f"• BEHAVIORAL: Significant amount deviation ({deviation} of typical)")

    return "\n".join(indicators) if indicators else "• No elevated entity risk indicators identified"


def generate_full_sar_report(
    transaction_metadata: Dict[str, Any],
    red_flags: List[Dict[str, str]],
    shap_summary: Dict[str, Any],
    provider: str = 'anthropic',
    use_llm: bool = True
) -> str:
    """
    Generate a complete SAR report.

    Args:
        transaction_metadata: Transaction details
        red_flags: List of identified red flags
        shap_summary: SHAP explanation summary
        provider: LLM provider for narrative generation
        use_llm: Whether to use LLM for narrative (False = template only)

    Returns:
        Complete formatted SAR report
    """
    # Generate report ID
    report_id = f"SAR-{datetime.now().strftime('%Y%m%d')}-{transaction_metadata.get('transaction_id', 'UNKNOWN')[-8:]}"

    # Generate narrative
    if use_llm:
        try:
            narrative_and_actions = generate_sar_narrative(
                transaction_metadata, red_flags, shap_summary, provider
            )
            # Parse the LLM response into sections
            narrative_summary = narrative_and_actions
            recommended_actions = generate_recommended_actions(
                transaction_metadata.get('risk_category', 'Medium'),
                red_flags
            )
        except Exception as e:
            narrative_summary = _generate_fallback_narrative(
                transaction_metadata, red_flags, shap_summary
            )
            recommended_actions = generate_recommended_actions(
                transaction_metadata.get('risk_category', 'Medium'),
                red_flags
            )
    else:
        narrative_summary = _generate_fallback_narrative(
            transaction_metadata, red_flags, shap_summary
        )
        recommended_actions = generate_recommended_actions(
            transaction_metadata.get('risk_category', 'Medium'),
            red_flags
        )

    # Format transaction analysis
    transaction_analysis = f"""
Transaction Pattern Analysis:
• Transaction occurred on {transaction_metadata.get('day_of_week', 'N/A')} at {transaction_metadata.get('time_of_day', 'N/A')}
• Merchant category ({transaction_metadata.get('merchant_category', 'N/A')}) risk score: {transaction_metadata.get('merchant_risk_score', 'N/A')}
• Transaction frequency in preceding 24 hours: {transaction_metadata.get('transaction_frequency_24h', 'N/A')} transactions
• Amount relative to 30-day average: {transaction_metadata.get('amount_deviation_ratio', 'N/A')}

Geographic Analysis:
• Distance from cardholder's registered address: {transaction_metadata.get('distance_from_home', 'N/A')}
• Transaction type: {transaction_metadata.get('location', 'N/A')}
"""

    # Build the report using template
    template = Template(SAR_TEMPLATE)
    report = template.substitute(
        report_id=report_id,
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        transaction_id=transaction_metadata.get('transaction_id', 'N/A'),
        transaction_date=transaction_metadata.get('timestamp', 'N/A'),
        transaction_amount=transaction_metadata.get('amount', 'N/A'),
        merchant_category=transaction_metadata.get('merchant_category', 'N/A'),
        transaction_location=transaction_metadata.get('location', 'N/A'),
        account_age=transaction_metadata.get('account_age_days', 'N/A'),
        risk_score=transaction_metadata.get('risk_score', 0),
        risk_category=transaction_metadata.get('risk_category', 'Unknown'),
        model_confidence="High" if transaction_metadata.get('risk_score', 0) > 80 or transaction_metadata.get('risk_score', 0) < 20 else "Medium",
        entity_risk_indicators=format_entity_risk_indicators(transaction_metadata),
        red_flags_section=format_red_flags_section(red_flags),
        transaction_analysis=transaction_analysis,
        shap_analysis=format_shap_analysis(shap_summary),
        narrative_summary=narrative_summary,
        recommended_actions=recommended_actions
    )

    return report


def check_api_availability() -> Dict[str, bool]:
    """Check which LLM APIs are available."""
    return {
        'anthropic': ANTHROPIC_AVAILABLE and bool(get_secret('ANTHROPIC_API_KEY')),
        'openai': OPENAI_AVAILABLE and bool(get_secret('OPENAI_API_KEY')),
        'gemini': GEMINI_AVAILABLE and bool(get_secret('GEMINI_API_KEY') or get_secret('GOOGLE_API_KEY'))
    }


if __name__ == '__main__':
    # Test report generation with mock data
    test_metadata = {
        'transaction_id': 'TXN-00012345',
        'timestamp': '2024-06-15 02:34:00',
        'amount': '$4,523.00',
        'merchant_category': 'Wire Transfer',
        'location': 'International',
        'distance_from_home': '2,500.0 km',
        'time_of_day': '02:34',
        'day_of_week': 'Sat',
        'transaction_frequency_24h': 15,
        'account_age_days': 45,
        'amount_deviation_ratio': '8.5x',
        'merchant_risk_score': '0.60',
        'risk_score': 94.5,
        'risk_category': 'Critical'
    }

    test_red_flags = [
        {
            'flag': 'Unusual Transaction Amount: $4,523.00',
            'detail': 'This amount deviates significantly from the account\'s typical transaction pattern.',
            'severity': 'HIGH',
            'severity_score': 0.8
        },
        {
            'flag': 'International Transaction Flag',
            'detail': 'Cross-border transaction detected with elevated risk.',
            'severity': 'HIGH',
            'severity_score': 0.7
        },
        {
            'flag': 'Off-Hours Transaction: 02:34',
            'detail': 'Transaction occurred during high-risk hours.',
            'severity': 'MEDIUM',
            'severity_score': 0.4
        }
    ]

    test_shap = {
        'top_features': [
            {'display_name': 'Transaction Amount', 'shap_value': 0.85, 'direction': 'increases', 'impact': 'high'},
            {'display_name': 'Distance from Home', 'shap_value': 0.72, 'direction': 'increases', 'impact': 'high'},
            {'display_name': 'International Transaction', 'shap_value': 0.65, 'direction': 'increases', 'impact': 'high'}
        ]
    }

    print("Generating SAR Report (template-based)...")
    report = generate_full_sar_report(
        test_metadata, test_red_flags, test_shap, use_llm=False
    )
    print(report)

    # Check API availability
    print("\n\nAPI Availability:")
    print(check_api_availability())
