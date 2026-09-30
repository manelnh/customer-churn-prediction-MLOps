# Customer Churn Prediction - MLOps Experiment

<div align="center">

[![Open the live app](https://img.shields.io/badge/Live%20demo-Open%20app-ff4b4b?logo=streamlit&logoColor=white)](https://customer-churn-prediction-mlops-crubmoactfmxfhqkzkyw92.streamlit.app)

A customer churn prediction project exploring how model training, inference, tracking, monitoring, and CI/CD can fit together. Training and monitoring are run locally; GitHub Actions handles CI and container delivery. PostgreSQL runs as a Docker Compose service.

**Run locally:** `python -m pip install -r requirements.txt`, then `streamlit run streamlit_app.py` from the repository root. For the full local stack, use `docker compose up --build`.

</div>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-database-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

## Table of contents

- [Preview](#preview)
- [Problem statement](#problem-statement)
- [Technical stack](#technical-stack)
- [About the project](#about-the-project)
- [Architecture](#architecture)
- [Local setup](#local-setup)
- [App sections](#app-sections)
- [Local training and monitoring](#local-training-and-monitoring)
- [GitHub Actions: CI/CD](#github-actions-cicd)
- [Project scope and next steps](#project-scope-and-next-steps)
## Preview

<p align="center">
  <img src="assets/app_1.png" alt="Manager Insights dashboard with churn risk and operational queue" width="850">
</p>

## Problem statement

Customer churn can reduce recurring revenue, and a churn score alone does not explain which customers may need attention or how teams can organize follow-up. This project explores a practical workflow for identifying at-risk customers, reviewing prediction results, and recording retention actions. It also provides a place to experiment with model tracking and monitoring. The project is an MLOps learning experiment; training and monitoring currently run locally.

## Technical stack

| Component | Role in this project |
|---|---|
| [![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/) | User interface for customer predictions, insights, retention actions, and model review. |
| [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-database-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/) | Stores prediction and governance history; runs as a service in Docker Compose. |
| [![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/) | Tracks local training and monitoring runs. |
| [![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/) | Runs the app, PostgreSQL, and MLflow together for local development. |
| [![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions) | Runs CI checks and builds/publishes the container image through CD. |
## About the project

This project applies customer churn prediction to a small operational workflow. The Streamlit app supports individual predictions, customer risk review, retention actions, and model evaluation. PostgreSQL stores prediction and governance records in the local Docker setup, and MLflow records experiment runs.

This is an MLOps learning experiment rather than a fully automated production platform. Model training and monitoring are run locally with the project scripts. GitHub Actions provides automated code checks and container image publishing. Training and monitoring workflows are also defined, but using them remotely requires externally reachable PostgreSQL and MLflow services and appropriate GitHub secrets.

## Architecture

```mermaid
flowchart LR
    A[Streamlit app] --> B[(PostgreSQL in Docker)]
    A --> C[MLflow in Docker]
    D[Local training script] --> C
    E[Local monitoring script] --> B
    E --> C
    F[GitHub Actions CI] --> G[Code checks and Docker build]
    H[GitHub Actions CD] --> I[GitHub Container Registry]
```

## Local setup

### Docker Compose

Start the app, PostgreSQL, and MLflow locally:

```bash
docker compose up --build
```

Open the app at `http://localhost:8501` and MLflow at `http://localhost:5000`.

### Python

Install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional integration settings are documented in `.streamlit/secrets.toml.example`.

## App sections

The app includes views for manager insights, individual predictions, retention actions, monitoring and retraining, churn analytics, and model governance/evaluation.

## Local training and monitoring

Training and monitoring can be started from the repository root:

```bash
python Scripts/run_training.py --reason manual_validation
python Scripts/run_monitoring.py
```

Monitoring accepts options such as `--days 14`, `--threshold 2`, and `--skip-mlflow-logging`. Training saves a local model bundle and can log experiments to MLflow. These scripts are intended to be run locally in the current setup.

<p align="center">
  <img src="assets/app_33.png" alt="MLflow training run with model evaluation metrics" width="800">
  <br>
  <sub>Example model training run recorded in MLflow.</sub>
</p>

<p align="center">
  <img src="assets/app_37.png" alt="MLflow monitoring run with drift alerts" width="800">
  <br>
  <sub>Example monitoring results recorded in MLflow.</sub>
</p>

## GitHub Actions: CI/CD

GitHub Actions is used for repository checks and container delivery:

| Workflow | Trigger | What it does |
|---|---|---|
| CI | Pushes and pull requests | Runs lint checks, compiles Python files, runs the unit suite with coverage, and builds the Docker image. |
| CD | Push to `main`/`master`, version tags, or manual dispatch | Builds and publishes the image to GitHub Container Registry. If `DEPLOY_WEBHOOK_URL` is configured, it also sends a deployment request. |

The repository also contains optional scheduled/manual Training and Monitoring workflows. Those are not the local demo path: they need remote PostgreSQL and MLflow endpoints configured as GitHub secrets.

[![CI workflow](https://github.com/manelnh/customer-churn-prediction-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/manelnh/customer-churn-prediction-MLOps/actions/workflows/ci.yml) [![CD workflow](https://github.com/manelnh/customer-churn-prediction-MLOps/actions/workflows/cd.yml/badge.svg)](https://github.com/manelnh/customer-churn-prediction-MLOps/actions/workflows/cd.yml)

### Workflow screenshots

<p align="center">
  <img src="assets/CI.png" alt="GitHub Actions CI workflow" width="850">
  <br>
  <sub>Continuous integration workflow in GitHub Actions.</sub>
</p>

<p align="center">
  <img src="assets/CD.png" alt="GitHub Actions CD workflow" width="850">
  <br>
  <sub>Continuous delivery workflow in GitHub Actions.</sub>
</p>

## Project scope and next steps

This repository demonstrates selected MLOps practices in a local development setup: a prediction interface, experiment tracking, monitoring scripts, a Dockerized database, and GitHub CI/CD workflows. It is not a fully automated production deployment. Training and monitoring are run locally; the optional remote workflows need hosted PostgreSQL and MLflow services.

Possible next steps include connecting the remote workflows to hosted services and validating a deployed container through the CD deployment hook.