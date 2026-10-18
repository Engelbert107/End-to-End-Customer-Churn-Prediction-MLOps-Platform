# End-to-End Customer Churn Prediction & MLOps Platform

An end-to-end machine learning project that predicts customer churn using the Telco Customer Churn dataset. The platform combines model experimentation, experiment tracking, a REST API, an interactive dashboard, automated testing, and Docker-based deployment.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who may be at risk of leaving can help businesses investigate customer needs and evaluate retention strategies.

This project demonstrates how to develop a churn classification model and make its predictions available through a web application.

### Key Features

- **Data preprocessing:** Handles missing values, numerical and categorical features, and feature encoding through an ML pipeline.
- **Model comparison:** Evaluates a baseline classifier, Logistic Regression, Random Forest, and XGBoost.
- **Model evaluation:** Uses held-out test data and ROC-AUC to compare model performance.
- **Experiment tracking:** Records model experiments and metrics using MLflow with a SQLite backend.
- **REST API:** Provides customer churn predictions and probabilities using FastAPI.
- **Interactive dashboard:** Allows users to enter customer information and view prediction results with Streamlit.
- **Automated testing:** Tests API endpoints, input validation, and prediction responses using pytest.
- **Containerization:** Packages the API and dashboard using Docker and Docker Compose.

## Architecture

The application follows this workflow:

1. Load and preprocess the Telco Customer Churn dataset.
2. Train and compare classification models.
3. Evaluate candidate models on a held-out test set.
4. Track experiments using MLflow and save the selected model pipeline.
5. Serve predictions through FastAPI.
6. Collect customer feature inputs through Streamlit.
7. Run the API and dashboard as separate Docker Compose services.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Experiment tracking | MLflow, SQLite |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Testing | pytest, HTTPX |
| Model persistence | joblib |
| Deployment | Docker, Docker Compose |

## Dataset

The project uses the Telco Customer Churn dataset, commonly available through Kaggle.

The dataset contains customer demographics, subscribed services, contract details, billing information, and a churn label.

- **Target:** `Churn`
- **Positive class:** `Yes`
- **Negative class:** `No`
- **Customer identifier:** `customerID`, excluded from model features

The `TotalCharges` field is converted to a numeric representation, and missing numerical values are handled within the preprocessing pipeline.

Dataset source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV file in `data/raw/` using the filename expected by the training script.

## Model Training and Evaluation

The project compares the following models:

- Dummy Classifier — baseline
- Logistic Regression
- Random Forest
- XGBoost

The preprocessing pipeline imputes missing values, scales numerical features where appropriate, and one-hot encodes categorical features. The data is split into training and test sets using a fixed random seed and stratification.

### Current Results

The recorded ROC-AUC scores are:

| Model | ROC-AUC |
|---|---:|
| XGBoost | 0.8438 |
| Logistic Regression | 0.8419 |
| Random Forest | 0.8349 |

XGBoost was selected as the best-performing non-baseline model based on ROC-AUC.

ROC-AUC measures how well a classifier ranks positive examples above negative examples across decision thresholds. These results are from the project's held-out test evaluation; they are not a guarantee of performance on future customers.

Run model training from the project root:

```bash
python scripts/train_models.py
```

Training generates the selected model artifact and evaluation reports, including model comparison and summary files.

## MLflow Experiment Tracking

MLflow is used to record experiments and compare candidate models.

The project uses a local SQLite tracking database named `mlflow.db`.

Start the tracking interface from the project root:

```bash
python -m mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db"
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view the experiment runs.

## FastAPI Prediction Service

The API exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Model health check |
| POST | `/predict` | Predict churn for a customer |

Run the API locally:

```bash
python -m uvicorn src.api:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

A successful prediction returns a response similar to:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.7783,
  "no_churn_probability": 0.2217,
  "model": "best_model.joblib"
}
```

The probabilities above are an illustrative example. Actual results depend on the supplied customer features and trained model.

## Streamlit Dashboard

The dashboard provides a form for entering customer attributes, including tenure, contract type, services, and billing details.

It sends the submitted features to the FastAPI service and displays the resulting churn prediction and probability.

Run the dashboard locally in a separate terminal:

```bash
python -m streamlit run src/dashboard.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501).

When running through Docker Compose, the dashboard uses the API service name to reach FastAPI over the internal Docker network.

## Automated Testing

The API tests cover:

- Root endpoint response
- Health endpoint response
- Successful prediction
- Prediction probability consistency
- Invalid input categories
- Missing required fields
- Negative monthly charges
- Unexpected input fields

Run the test suite:

```bash
python -m pytest tests/test_api.py -v
```

**Current test status:** 8 tests passed in the verified test run.

## Docker Deployment

Docker Compose runs the API and Streamlit dashboard as separate services.

Build and start both services from the project root:

```bash
docker compose up --build
```

Access the application:

- **Dashboard:** [http://127.0.0.1:8501](http://127.0.0.1:8501)
- **API documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Run in the background:

```bash
docker compose up --build -d
```

View service logs:

```bash
docker compose logs -f
```

Stop the services:

```bash
docker compose down
```

## Project Structure

```text
Customer-Churn/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── best_model.joblib
├── reports/
│   ├── model_metrics.json
│   ├── model_comparison.csv
│   └── best_model_summary.json
├── scripts/
│   └── train_models.py
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── api.py
│   └── dashboard.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

The generated MLflow database and local virtual environment are development artifacts and should not be committed unless there is a specific reason to include them.

## Future Improvements

- Add automated CI checks using GitHub Actions.
- Monitor prediction volume and input feature distributions.
- Track data drift and evaluate model performance when new labels become available.
- Add model versioning and reproducible training workflows.
- Improve API observability, logging, and error handling.
- Evaluate probability calibration and decision thresholds based on business requirements.

## Limitations and Responsible Use

This project is an educational demonstration using a public dataset. Predictions represent model estimates, not certain outcomes. Model performance may change across populations, time periods, and business contexts.

Before using such a system for real business decisions, validate it on appropriate data, assess potential bias, establish monitoring and privacy controls, and involve human review.

