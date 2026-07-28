# SageMaker Unified Studio Demo Talk Track

## Opening (30-45 sec)
- This demo predicts customer demand using operational and customer-signal features.
- Goal: help teams anticipate requests and allocate resources proactively.

> [Customize before commit] Replace opening with your team/domain context.

## Problem Framing
- Teams often react to customer demand instead of forecasting it.
- We use support, feature requests, pricing, and promo signals to estimate demand units.

## Data + Features
- Inputs include region, product category, customer segment, support tickets, feature requests, promo flag, and market indexes.
- We add simple time features (month/week) and encode categories.

## Model + Evaluation
- Baseline model: RandomForestRegressor.
- Metrics: MAE, RMSE, R².

> [Customize before commit] Add your own metric targets and why they matter to the business.

## Unified Studio Positioning
- Notebook-based exploration in AWS SageMaker Unified Studio.
- Path to production: scheduled pipelines, model registry, monitoring, and periodic retraining.

## Close
- Main result: turn customer demand signals into forecasting insights.
- Next steps: add external drivers and compare advanced models.

> [Customize before commit] Add a final CTA for your audience (e.g., pilot proposal).
