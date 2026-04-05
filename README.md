# Credit Risk Scoring — Altman Z''-Score + Machine Learning

A dual-layer credit risk scoring system for NSE-listed Indian companies, combining the classical Altman Z''-Score model with machine learning (Gradient Boosting, Random Forest, Logistic Regression) and SHAP explainability. Built with Python and deployed as an interactive Streamlit dashboard.

---

## What This Project Does

Most credit scoring tools use either a rule-based model (like Altman Z-Score) or a black-box ML model. This project uses both — and compares them.

- **Layer 1 — Altman Z''-Score (1995):** Classical discriminant analysis model adapted for emerging markets. Classifies companies into Safe, Grey Zone, or Distress based on four financial ratios.
- **Layer 2 — ML Classifier:** Gradient Boosting / Random Forest trained on the same four ratios, producing a probability of default (0–100%).
- **Disagreement Analysis:** The most interesting output — companies where Z''-Score says "Safe" but the ML model flags as high risk, explained using SHAP waterfall charts.

---

## Live Dashboard Preview

The Streamlit dashboard includes:
- Z''-Score vs ML Probability of Default scatter plot (4-quadrant view)
- Company deep-dive with radar chart vs sector average
- Disagreement analysis table
- Ratio averages by zone (Safe / Grey / Distress)
- Sector and zone filters
- Export to CSV

---

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-risk-scoring.git
cd credit-risk-scoring
```

### 2. Install dependencies
```bash
pip install yfinance pandas numpy scikit-learn shap matplotlib seaborn streamlit plotly openpyxl xgboost
```

### 3. Run the pipeline
```bash
python project.py
```

This fetches financial data for 50 NSE-listed companies from Yahoo Finance, computes Z''-Scores, trains three ML models, generates SHAP plots, and exports `scored_companies.csv`.

Expected runtime: ~5 minutes on first run (data fetch). Subsequent runs load from cache.

### 4. Launch the dashboard
```bash
streamlit run streamlit_app.py
```

Opens automatically at `http://localhost:8501`.

---

## The Model

### Altman Z''-Score (1995 Emerging Market Variant)
Z'' = 6.56×X1 + 3.26×X2 + 6.72×X3 + 1.05×X4

| Variable | Formula | What It Measures |
|----------|---------|-----------------|
| X1 | Working Capital / Total Assets | Short-term liquidity |
| X2 | Retained Earnings / Total Assets | Cumulative profitability |
| X3 | EBIT / Total Assets | Operating efficiency |
| X4 | Book Equity / Total Liabilities | Leverage (solvency) |

| Z''-Score | Zone |
|-----------|------|
| > 2.6 | Safe |
| 1.1 – 2.6 | Grey Zone |
| < 1.1 | Distress |

The Z''-Score variant is used instead of the original Z-Score because it uses **book value of equity** (not market cap) for X4, making it more appropriate for Indian listed companies where market prices can be volatile and thin.

### ML Layer

Three classifiers trained via 5-fold stratified cross-validation:

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, fully interpretable |
| Gradient Boosting | Usually best AUC, used for SHAP |
| Random Forest | Robust to outliers, good on small datasets |

Evaluation metric: **AUC-ROC** (cross-validated). Class imbalance handled via `class_weight="balanced"`.

### SHAP Explainability

SHAP (SHapley Additive exPlanations) explains every individual prediction:
- **Summary plot:** Which ratios drive distress risk most globally
- **Waterfall chart:** Why the model scored the highest-risk company the way it did

---

## Data Sources

| Source | What It Provides | Access |
|--------|-----------------|--------|
| Yahoo Finance (yfinance) | Balance sheet, income statement, cash flow for NSE stocks | Free, no account |
| NSE (.NS suffix) | All Nifty 100 and stressed company tickers | Public |
| IBBI (ibbi.gov.in) | CIRP case list for distress labels (optional enhancement) | Free, public |
| RBI DBIE | Sector-level NPA rates (optional enhancement) | Free, public |

No paid data sources required. All data is fetched programmatically via yfinance.

---

## Key Finding

The most significant output of this project is the **disagreement quadrant** — companies where the Altman Z''-Score classifies as "Safe" (Z'' > 2.6) but the ML model assigns a high probability of default (PD > 50%).

In testing on Nifty 100 + stressed company universe, SHAP analysis revealed that these disagreements were primarily driven by **declining X3 (EBIT/Total Assets) trends** — companies with strong book equity (keeping Z''-Score elevated) but quietly deteriorating operating profitability that the single-year Z''-Score snapshot misses.

This is the core argument for layering ML on top of classical models in credit analysis.

---

## Limitations

- yfinance data quality varies by ticker — some companies have incomplete or missing line items
- Distress labels are derived from Z''-Score thresholds, not actual NPA/default events — for a production model, labels should come from IBBI CIRP data or RBI NPA classifications
- Sample size (~50 companies) is small for robust ML training — expanding to 200+ companies using Screener.in bulk export would improve model reliability
- Model trained on Nifty 100 (large caps) — may not generalise to mid-cap or small-cap credit risk without retraining

---

## Potential Extensions

- [ ] Add IBBI CIRP list as ground-truth distress labels
- [ ] Expand universe to 200+ companies using Screener.in data
- [ ] Rolling Sharpe / Z''-Score trend analysis (time-series panel)
- [ ] Sector-specific model training (banks need different ratios)
- [ ] FastAPI endpoint for real-time company scoring
- [ ] Deploy dashboard to Streamlit Cloud

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, yfinance |
| ML | scikit-learn, xgboost |
| Explainability | shap |
| Visualization | plotly, matplotlib, seaborn |
| Dashboard | streamlit |
| Export | openpyxl |

---

## References

- Altman, E.I. (1968). *Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.* Journal of Finance, 23(4), 589–609.
- Altman, E.I., Hartzell, J., & Peck, M. (1995). *Emerging Markets Corporate Bonds: A Scoring System.* Salomon Brothers.
- Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. [SHAP paper]
- IBBI — Insolvency and Bankruptcy Board of India: https://ibbi.gov.in
- RBI Financial Stability Report: https://www.rbi.org.in

---

## Author

**Deeraj**  
PGDM, IMT Ghaziabad (Batch 2025–27)  
B.Tech Mechanical Engineering, NIT Calicut  
[LinkedIn](www.linkedin.com/in/deeraj-yerramsetti-2379b31aa) · [GitHub](https://github.com/yssvd011001)
