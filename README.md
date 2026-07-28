# Customer Demand Intelligence — Data Science Demo for AWS SageMaker Unified Studio

> [Customize before commit] Replace this line with your one-sentence demo objective.

This repository is a hands-on, presentation-ready data science demo that shows how to:

- Ingest customer request and demand signals
- Explore and clean demand data
- Engineer useful demand features
- Train a baseline ML model for customer demand prediction
- Evaluate model performance and discuss business impact

## 1) Demo Story

> [Customize before commit] Add your demo narrative in 3-5 bullets. Example:
> - Business challenge
> - Data sources used
> - Model objective
> - Key metric improved
> - Deployment or next-step plan

## 2) Repository Structure

```text
customer-demand-intelligence-ds-demo/
├── data/
│   ├── raw/customer_demand_sample.csv
│   └── processed/.gitkeep
├── notebooks/
│   └── 01_customer_demand_eda_and_baseline.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate.py
├── docs/
│   └── demo_talk_track.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 3) Quick Start

```bash
# [Customize before commit] Update Python version if needed.
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

## 4) Suggested SageMaker Unified Studio Demo Flow

1. Open notebook in Unified Studio.
2. Run EDA and feature engineering cells.
3. Train baseline model.
4. Review evaluation metrics.
5. Explain how this extends to production (pipelines, retraining, monitoring).

> [Customize before commit] Add screenshots or links from your Unified Studio workspace.

## 5) Key Outputs

- Trained baseline demand model
- Evaluation report with MAE/RMSE/R²
- Customer-demand business insights

> [Customize before commit] Add your final achieved metrics after running the notebook/scripts.

## 6) Next Enhancements

- Add time-series model variants
- Integrate external signals (seasonality, promotions, regional events)
- Add model registry + CI/CD workflow for retraining

> [Customize before commit] Replace with your roadmap.
