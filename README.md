# Churn Prediction MLOps App

**An end-to-end customer churn project covering prediction, retention actions, monitoring, and model governance.**

<div align="center">

[![Live demo](https://img.shields.io/badge/Live%20demo-Open%20app-ff4b4b?logo=streamlit&logoColor=white)](https://customer-churn-prediction-mlops-crubmoactfmxfhqkzkyw92.streamlit.app)

</div>

This project explores how a telecom team can identify customers at risk of leaving, understand the factors behind each score, and prioritize practical retention follow-up. It combines a Streamlit decision-support app with a trained churn model, optional PostgreSQL history, monitoring, and model-governance tools.

## Problem and approach

A churn score is useful only when it helps someone decide what to do. Teams need to find higher-risk customers, understand the reasons behind a prediction, record follow-up actions, and review whether the model remains useful as new predictions arrive.

I built this project to connect those steps in one workflow. The app scores customer information with the production model, explains the main factors contributing to the score, supports retention-action tracking, and provides views for monitoring model health and reviewing candidate models.

## App sections

The Streamlit app is organized around six work areas:

| Section | What it does |
|---|---|
| **Manager Insights** | Summarizes saved predictions, churn-risk distribution, revenue exposure, and outstanding operational work. |
| **Predictions** | Scores an individual customer, shows the risk result and explanation, and saves the result when database history is configured. |
| **Action Center** | Helps prioritize at-risk customers and record retention actions and follow-up status. |
| **Monitoring & Retraining** | Reviews prediction and alert history, checks production evidence, and provides retraining workflows. |
| **Deep-Dive Analytics** | Explores churn patterns across customer segments and available prediction data. |
| **Governance & Lab** | Evaluates candidate models and records governance decisions about model changes. |

PostgreSQL is an optional integration for persistent prediction, monitoring, and governance history. Without it, prediction and analysis features remain available, but saved history is not available.

## Technology

- **Streamlit** for the interactive app
- **scikit-learn** for churn prediction
- **PostgreSQL** and **Alembic** for persistent history and schema migrations
- **MLflow** for experiment tracking
- **GitHub Actions** for continuous integration and optional scheduled workflows

## Run locally

Install Python dependencies from the repository root and start the app:

~~~bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
~~~

To run the full local stack, including PostgreSQL and MLflow, use Docker Compose:

~~~bash
docker compose up --build
~~~

The app is available at http://localhost:8501. The MLflow interface is available at http://localhost:5000 when running the full stack.

To create or update the PostgreSQL schema, run:

~~~bash
alembic upgrade head
~~~

Database setup is optional if you only want to explore prediction and analysis features.

## Model training and monitoring

Run the training workflow from the repository root:

~~~bash
python Scripts/run_training.py --reason manual_validation
~~~

Run monitoring against saved production predictions:

~~~bash
python Scripts/run_monitoring.py
~~~

The monitoring script also accepts options such as --days 14 and --skip-mlflow-logging. See its command-line help for the available settings.

## Configuration

Optional integrations are configured through environment variables or Streamlit secrets. See .streamlit/secrets.toml.example for the example settings.

- **DATABASE_URL** enables PostgreSQL-backed prediction history, monitoring alerts, and governance records.
- **GitHub Actions settings** enable the app to dispatch configured training or monitoring workflows. These are only needed when using that integration.

Keep real credentials in Streamlit Cloud secrets or a local secrets file that is excluded from Git. Do not commit credentials to the repository.

## Tests

Run the unit tests with:

~~~bash
python -m unittest discover -s tests -v
~~~

## Screenshots

<p align="center">
  <img src="assets/app_1.png" alt="Manager Insights dashboard with risk breakdown and operational queue" width="850">
</p>

<p align="center">
  <img src="assets/app_33.png" alt="MLflow training run with model evaluation metrics" width="800">
</p>

<p align="center">
  <img src="assets/app_37.png" alt="MLflow monitoring run with drift alerts" width="800">
</p>

## License

This project is available under the MIT License. See LICENSE for details.
