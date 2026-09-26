# Churn Prediction MLOps App

This project packages a churn prediction workflow with:

- a Streamlit application for production-style inference and review
- PostgreSQL for prediction logs, governance records, and monitoring alerts
- MLflow for experiment tracking and monitoring run history
- GitHub Actions pipelines for CI, container delivery, scheduled monitoring, and scheduled retraining
- Alembic migrations for reproducible database schema management

## Local Startup

The easiest first-time setup is Docker Compose:

```powershell
docker compose up --build
```

If you want to initialize the database schema explicitly before running the app:

```powershell
alembic upgrade head
```

Then open:

- Streamlit app: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

## Core User Flow

1. Open the `Single Prediction` tab and run a few customer predictions.
2. Review risk summaries in `Manager Insights`.
3. Save operational decisions in `Action Center`.
4. Use `Technical Lab` to compare a candidate model and log it to MLflow.
5. Open `Monitoring Dashboard` to review live evidence and stored alerts.

## Running Monitoring From The Terminal

The monitoring script can be run manually or on a scheduler without opening Streamlit:

```powershell
python Scripts/run_monitoring.py
```

Useful options:

```powershell
python Scripts/run_monitoring.py --days 14
python Scripts/run_monitoring.py --skip-mlflow-logging
python Scripts/run_monitoring.py --threshold 2 --fail-on-high-alerts
```

What it does:

- reads recent production predictions from PostgreSQL
- computes live coverage and drift-style alerts
- logs monitoring runs to MLflow unless skipped
- prints whether retraining should be considered

## Running Training From The Terminal

You can also trigger the end-to-end training pipeline directly:

```powershell
python Scripts/run_training.py --reason manual_validation
```

What it does:

- loads the churn dataset
- trains multiple logistic regression variants
- logs experiments to MLflow
- saves the best production bundle locally
- attempts MLflow model registration/promotion

## Tests

Run the lightweight unit suite with:

```powershell
python -m unittest discover -s tests -v
```

## Migrations

Alembic is included so the PostgreSQL schema can evolve in a reproducible way:

```powershell
alembic upgrade head
```

The initial migration creates:

- `prediction_logs`
- `monitoring_alerts`
- `governance_decisions`

Schema responsibility is intentionally separated from the app:

- migrations define the database structure
- CI/CD and deployment startup run those migrations explicitly
- the application then uses the schema instead of silently changing it on every request

## CI

GitHub Actions CI is defined in `.github/workflows/ci.yml` and now performs:

- dependency installation with pip cache
- Python lint checks for critical import/name errors
- bytecode compilation checks
- unit tests with coverage
- Docker image build validation

## CI/CD Story For The Jury

This repository uses GitHub Actions as the control plane for quality, delivery, and operational MLOps tasks.

How the integration works:

- `CI` runs on every push and pull request to catch Python errors, broken imports, failing tests, and Docker build regressions before code is merged.
- `Training` is implemented as a scheduled and manually triggerable GitHub workflow with `workflow_dispatch`, which demonstrates remote retraining automation on clean GitHub runners.
- `Monitoring` is implemented as a scheduled and manually triggerable GitHub workflow that evaluates live production evidence and can raise retraining signals in a traceable way.
- `CD` builds and publishes the container image so the deployed app matches a reviewed Git commit and an auditable automation trail.

For the live jury demonstration, the Streamlit app uses local training and local monitoring so the end-to-end flow remains reliable even if external GitHub-connected services are unavailable.

Why this matters in front of a jury:

- it shows that the project is not only a model notebook or dashboard, but a managed ML system with controls around quality, reproducibility, and release
- it separates development, validation, deployment, and operations into explicit stages that can be demonstrated independently
- it reduces manual risk because tests, packaging, and operational jobs run the same way every time
- it improves auditability because retraining and monitoring actions are logged through the same automation layer as code changes
- it makes the project easier to scale to a team setting because the process does not depend on hidden local steps

In short, the CI/CD layer is the bridge between the model and real production practice: it turns a good churn model into an end-to-end MLOps workflow.

## CD And Automation

Additional GitHub Actions workflows are included:

- `.github/workflows/cd.yml`
  Builds and pushes a Docker image to GitHub Container Registry on `main`/`master` and tags. If `DEPLOY_WEBHOOK_URL` is configured as a GitHub secret, it also triggers deployment automatically.
- `.github/workflows/monitoring.yml`
  Runs the monitoring cycle daily on GitHub Actions and also supports manual dispatch against externally reachable PostgreSQL and MLflow services. When high-severity alerts are detected, it can launch remote retraining on GitHub runners automatically.
- `.github/workflows/training.yml`
  Runs retraining weekly on GitHub Actions and also supports manual dispatch against an externally reachable MLflow service.

## Required GitHub Secrets

To activate the full MLOps automation on GitHub, configure these repository or environment secrets:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `DATABASE_URL` or the separate PostgreSQL secrets below
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `DEPLOY_WEBHOOK_URL` for optional deployment triggering

Using `DATABASE_URL` is the simplest option for remote monitoring workflows because it avoids splitting the PostgreSQL connection across multiple secrets.

## Optional Streamlit-To-GitHub Dispatch

The repository still includes optional Streamlit-to-GitHub dispatch support for training and monitoring workflows.

This is kept as implementation evidence, but it is not the recommended jury demo path. For the demonstration, the app uses local execution for monitoring and retraining.

To enable that path, provide these runtime environment variables to the Streamlit app:

- `GITHUB_ACTIONS_TOKEN`
- `GITHUB_REPOSITORY`
- `GITHUB_TRAINING_WORKFLOW` such as `training.yml`
- `GITHUB_MONITORING_WORKFLOW` such as `monitoring.yml`
- `GITHUB_WORKFLOW_REF` such as `main`

The workflow ref must point to a branch or tag where those workflow files are already pushed with the expected `workflow_dispatch` inputs. If GitHub responds with `HTTP 422` and `Unexpected inputs provided`, the target ref is usually still serving an older workflow definition.

If you want the Streamlit app to avoid local execution and prefer GitHub-hosted automation, also set:

- `AUTOMATION_EXECUTION_MODE=github`

When those values are configured, the app can trigger `workflow_dispatch` on the GitHub training pipeline and pass both the retraining reason and the selected training profile.

This is useful when you want retraining to happen through the same CI/CD control plane that handles auditability, centralized logs, runner isolation, and release automation, but it is optional for the current presentation setup.

## Important Note About GitHub-Run Training And Monitoring

The `Training` and `Monitoring` workflows are designed to run on GitHub-hosted runners.

- They require your MLflow and PostgreSQL services to be reachable from GitHub Actions.
- Local Docker addresses such as `http://localhost:5000` or `db` will not work from GitHub-hosted runners.
- If your services are local-only, GitHub automation cannot reach them until you move them to an externally reachable environment.

That is why the project currently uses:

- GitHub Actions as proof of CI/CD and remote MLOps workflow implementation
- local execution inside the app as the safest live demo path

## Full Remote Automation Checklist

To make retraining and monitoring fully remote instead of local:

1. Host PostgreSQL on a service reachable from GitHub Actions.
2. Host MLflow on a service reachable from GitHub Actions.
3. Set GitHub repository secrets for:
   - `MLFLOW_TRACKING_URI`
   - `MLFLOW_EXPERIMENT_NAME`
   - `DATABASE_URL` or `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD`
4. Keep the Streamlit app configured with:
   - `GITHUB_ACTIONS_TOKEN`
   - `GITHUB_REPOSITORY`
   - `GITHUB_TRAINING_WORKFLOW`
   - `GITHUB_MONITORING_WORKFLOW`
   - `GITHUB_WORKFLOW_REF`
   - `AUTOMATION_EXECUTION_MODE=github`
5. Trigger one manual GitHub training run and one manual GitHub monitoring run to validate connectivity before relying on the schedules.

The updated workflows now fail early with explicit reachability checks for MLflow and PostgreSQL, which makes remote setup debugging much faster.

## Project MLOps Coverage

This repository now covers the main blocks expected in a complete PFE-style MLOps project:

- data-driven model training and experiment tracking
- production inference and prediction logging
- governance decision logging
- monitoring and retraining signals
- schema migrations
- CI quality gates
- CD-ready container publishing
- scheduled operational automation
