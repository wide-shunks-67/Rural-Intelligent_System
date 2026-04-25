# AI-powered Rural Intelligence System

Transforming rural development datasets into NLP-ready intelligence for analysis, classification, and decision support.

## Problem Statement

Rural development data is often spread across large government datasets that are difficult to interpret quickly. This project converts structured rural indicators into natural language, applies AI-based classification, and produces insight-ready outputs for monitoring, prioritization, and decision-making.

## Key Features

- NLP-based text generation from structured datasets such as MGNREGA, PMGSY, Sanitation, and NFHS
- Issue detection and severity scoring for rural development signals
- AI classification using HuggingFace zero-shot learning with `facebook/bart-large-mnli`
- Insight engine for state-level and district-level analysis
- Interactive Streamlit dashboard for exploration and text analysis

## Tech Stack

- Python
- pandas
- transformers
- torch
- streamlit
- plotly

## System Architecture

```text
Data -> NLP -> ML -> Insight Engine -> Dashboard
```

## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/wide-shunks-67/Rural-Intelligent_System.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Generate the base NLP dataset

```bash
python rural_text_generation.py
```

4. Run AI classification

```bash
python add_hf_predictions.py
```

5. Launch the dashboard

```bash
streamlit run app.py
```

##  What Makes This Project Unique

- Combines rule-based reasoning + AI classification  
- Works on real government datasets  
- Converts data into decision-support insights
 
## Sample Output

### Generated Text

```text
In Sukma, Chhattisgarh, employment conditions appear mixed under MGNREGA. Households received moderate employment support, but participation gaps remain uneven.
```

### Insight Example

```text
Chhattisgarh shows a high share of severity-tagged records, with employment stress and participation gaps emerging as recurring concerns.
```

## Future Improvements

- Multilingual support for regional language outputs
- Real-time API integration for live dataset updates
- Advanced ML models for domain-specific rural classification and forecasting
