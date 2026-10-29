# Group 25: Credit Card Fraud Detection
## Monitoring and Maintenance Plan

### 1. Purpose

The purpose of this plan is to monitor the reliability, performance, and continued effectiveness of the Credit Card Fraud Detection system. The system uses an XGBoost model to identify potentially fraudulent credit card transactions.

### 2. Current Implementation Status

The project includes a FastAPI application with a health endpoint and a prediction endpoint. The deployed health endpoint returns `status: "ok"` and reports the model name, feature counts, and decision threshold.

The `/predict` endpoint currently returns a `502 Bad Gateway` response during testing. Therefore, successful end-to-end prediction through the deployed API has not yet been verified.

### 3. Monitoring Activities

| Area | Monitoring activity | Status |
|---|---|---|
| API availability | Check the `/health` endpoint and service availability. | Health endpoint verified |
| Prediction reliability | Test valid requests and record errors from `/predict`. | Requires further debugging |
| Model performance | Track precision, recall, F1-score, ROC-AUC, and average precision using appropriately labeled evaluation data. | Evaluation code exists; ongoing monitoring not verified |
| Data quality | Check required features, missing values, invalid types, and non-finite values. | Validation tests included in project code; deployment verification pending |
| Latency and failures | Record prediction response times, error rates, and server failures. | Ongoing monitoring needs verification |
| Model and threshold | Record the model version and decision threshold used in each release. | Current threshold reported as 0.994541585445404 |

### 4. Maintenance Procedures

**Routine checks:** Check service availability, review application logs, and investigate repeated failures.

**Data validation:** Ensure prediction requests contain the required 30 features and valid numerical values.

**Performance review:** When reliable labeled transaction outcomes become available, compare model predictions with actual outcomes and review precision, recall, false positives, and false negatives.

**Model updates:** Retrain or recalibrate the model only when justified by evidence, using a documented training and validation process. Select thresholds on validation data and reserve the test set for final evaluation.

**Deployment maintenance:** Test model loading, dependencies, API endpoints, and prediction consistency before releasing an updated version.

**Security:** Avoid storing unnecessary sensitive transaction information in logs. Protect credentials and restrict access to deployment settings and model artifacts.

### 5. Alerts and Response

- **Service unavailable:** Inspect deployment events, application logs, and server resource usage.
- **Prediction errors:** Capture the error details and reproduce the issue with a controlled test request.
- **Data validation failure:** Reject invalid inputs with a clear client error rather than silently modifying the data.
- **Model performance deterioration:** Investigate data drift, changes in fraud patterns, and threshold suitability before considering retraining.

### 6. Limitations and Future Improvements

The current health check confirms that the API can report its model configuration, but it does not confirm successful transaction scoring. The deployed prediction endpoint requires further debugging.

Future improvements include durable monitoring storage, dashboards for operational metrics, automated alerts, model-version tracking, and periodic evaluation with newly labeled data.

### 7. Conclusion

Monitoring and maintenance are essential for keeping the fraud detection system reliable and useful. The current project provides a foundation for API health checks and model evaluation. Further work is required to verify deployed predictions and establish continuous monitoring in a production environment.
