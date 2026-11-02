# FraudShield – Group 25 Credit Card Fraud Detection

## 1. Project Overview

FraudShield is an educational AI prototype developed for the FUSE AI-201 Module 6 project. It demonstrates how a machine learning model can help identify potentially fraudulent credit card transactions.

The project uses an XGBoost classification model and exposes its prediction functionality through a REST API built with FastAPI.

**Project status:** Educational prototype deployed locally using Docker. It has not been validated for production use.

## 2. Technologies Used

- Python 3.12
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Scikit-learn
- Pandas
- NumPy
- Joblib

## 3. Project Files

The deployment project contains the following main files:

| File | Purpose |
|---|---|
| `app.py` | FastAPI application and prediction logic |
| `fraud_detection_bundle.joblib` | Trained model and preprocessing artifacts |
| `model_metadata.json` | Model information and configuration |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Instructions for building the Docker image |
| `README.md` | Project overview and deployment instructions |
| `FraudShield_Deployment_Documentation.docx` | Detailed deployment documentation |

## 4. Running the Application

### Prerequisites

- Docker Desktop installed and running.
- The project files available in the `fraud_detection_deployment` folder.

### Start the API

Open a terminal in the project folder and run:

```bash
docker build -t fraudshield-api .
```

Then start the container:

```bash
docker run -d --name group25-fraud-api -p 8000:8000 fraudshield-api
```

If the container is already running, do not run the second command again.

### Access the API

Open the following address in a web browser:

`http://127.0.0.1:8000/docs`

This opens the interactive Swagger UI, where the API endpoints can be explored and tested.

## 5. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model availability |
| POST | `/predict` | Predicts the fraud probability of a transaction |
| GET | `/metrics` | Displays basic API monitoring metrics |

### Prediction Input

The prediction endpoint expects 30 transaction features:

- `Time`
- `V1` through `V28`
- `Amount`

The `Class` label is not included in the prediction request.

### Prediction Output

The API returns:

- Fraud probability
- Classification threshold
- Prediction result
- Decision label
- Educational-use notice

The decision labels are:

- `FLAG_FOR_REVIEW`
- `NOT_FLAGGED`

## 6. Testing Results

The API was tested using a synthetic smoke-test input and two transactions from the labelled dataset.

| Test | Expected label | API decision | Result |
|---|---:|---|---|
| Synthetic smoke test | Not applicable | NOT_FLAGGED | Successful API response |
| Fraud-labelled transaction | 1 | FLAG_FOR_REVIEW | Matched |
| Legitimate-labelled transaction | 0 | NOT_FLAGGED | Matched |

The fraud-labelled transaction returned a fraud probability of approximately `0.999923`.

The legitimate-labelled transaction returned a fraud probability of approximately `0.001782`.

These examples demonstrate that the deployed API can process inputs and return predictions. They do not establish overall model accuracy or production readiness.

## 7. Monitoring

The `/metrics` endpoint provides basic in-memory monitoring information, including:

- Total requests
- Successful predictions
- API errors
- Validation failures
- Fraud flags
- Average request latency
- Fraud flag rate

The monitoring counters are held in memory and reset when the container restarts. Persistent monitoring and automated drift detection have not been implemented.

## 8. Deployment Architecture

The prototype follows this deployment flow:

1. A user submits transaction features to the FastAPI application.
2. The API validates the input.
3. The application applies the saved preprocessing transformations.
4. The XGBoost model generates a fraud probability.
5. The API compares the probability with the configured threshold.
6. The API returns a prediction and decision.
7. Basic request statistics are made available through the monitoring endpoint.

The application is packaged and run inside a Docker container.

## 9. Limitations and Future Improvements

This prototype is intended for educational demonstration and is not suitable for live financial transaction decisions.

Potential future improvements include:

- Testing with a larger and more diverse set of transactions.
- Evaluating model performance using appropriate classification metrics.
- Adding persistent monitoring and logging.
- Implementing data and model drift monitoring.
- Improving API security and authentication.
- Deploying to a suitable cloud environment.
- Conducting further validation before any real-world use.

## 10. Evidence

Screenshots documenting the prototype and its tests are saved separately in the `FraudShield_Deployment_Evidence` folder.

The evidence includes:

1. Swagger API endpoints.
2. API health check.
3. Synthetic prediction test.
4. API monitoring metrics.
5. Fraud-labelled transaction test.
6. Legitimate-labelled transaction test.
7. Final API monitoring metrics.

## 11. Conclusion

FraudShield demonstrates the deployment of an XGBoost fraud detection model through a FastAPI application packaged with Docker. The prototype successfully exposes prediction, health-check, and monitoring endpoints and has been tested with fraud-labelled and legitimate-labelled transaction examples.

The project provides a foundation for further testing, evaluation, monitoring, and improvement as part of the AI prototype demonstration and deployment planning assignment.