# Instamart Customer Intelligence and Support Co-Pilot

## Overview

This project develops an end-to-end customer intelligence and support co-pilot prototype inspired by the Instamart online grocery delivery use case.

The system combines data cleaning, SQL analytics, customer segmentation, lapse prediction, support ticket classification, policy-based RAG, and an automated refund agent.

### Objectives

- Clean customer, order, product, and ticket datasets.
- Build a SQLite database and perform SQL analytics.
- Identify customer behavior patterns and potential lapse risk.
- Classify support tickets into delivery, billing, fraud, and technical categories.
- Retrieve relevant policy information using TF-IDF and cosine similarity.
- Automate refund decisions while supporting escalation and fallback.
- Maintain human oversight for uncertain or high-value decisions.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- pandas
- numpy
- scikit-learn
- joblib

### Setup

Run these commands from the project root directory in PowerShell.

1. Create a virtual environment:

   `py -m venv .venv`

2. Activate the virtual environment:

   `.\.venv\Scripts\Activate.ps1`

3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Generate the project data:

   `python generate_data.py`

5. Clean the datasets:

   `python src/clean_data.py`

6. Build the SQLite database:

   `python src/build_database.py`

7. Validate the cleaned data and database:

   `python src/validate_data.py`

8. Generate customer features:

   `python src/customer_segmentation.py`

9. Run customer lapse prediction:

   `python src/lapse_prediction.py`

10. Identify high-risk customers:

    `python src/high_risk_customers.py`

11. Train and evaluate the ticket classifier:

    `python src/ticket_classifier.py`

12. Test ticket predictions:

    `python src/predict_ticket.py`

13. Run the refund decision engine:

    `python src/refund_engine.py`

14. Evaluate the policy RAG assistant:

    `python src/policy_rag.py`

15. Run the refund agent:

    `python src/refund_agent.py`

Generated results are saved in the `outputs/` directory.

**Important:** The LMS-provided `generate_data.py` and `mock_llm.py` files are required. Use the original provided versions without modifying them.

## Part A: Data Foundation

### Data Cleaning

The project processes customer, order, product, and ticket datasets.

Data preparation includes:

- Handling missing and invalid values.
- Normalizing inconsistent city and gender values.
- Removing duplicate order records.
- Correcting invalid order amounts using product price, quantity, and discount.
- Validating order and customer records.

### SQLite Database

The cleaned datasets are stored in a SQLite database.

The database supports structured queries and customer/order analytics.

### SQL Analytics

SQL queries analyze customer activity, order behavior, revenue, and purchasing patterns.

A cumulative daily revenue query uses SQL window functions to calculate revenue over time.

### Outputs

- `data/clean/`
- `data/instamart.db`
- `outputs/customer_activity_summary.csv`
- `outputs/customer_order_summary.csv`

## Part B: Customer Segments and Lapse Prediction

### Customer Feature Engineering

Customer features include:

- Total orders
- Completed orders
- Cancelled orders
- Total spending
- Average order value
- Cancellation rate
- Recency of customer activity

### Customer Segmentation

K-Means clustering is applied to customer activity features using feature scaling.

The project identifies four customer behavior groups:

- High-spending customers
- Low-spending customers
- Frequent customers
- Inactive customers

Cluster IDs are algorithm-generated and should be interpreted using their customer metrics.

### Lapse Prediction

A Logistic Regression model predicts potential customer lapse using historical activity features.

The model uses earlier customer activity to predict whether a customer completed an order in a later period.

The September completed-order label is used as a lapse proxy.

**Limitation:** This is a proxy label, not a verified long-term churn label.

### Outputs

- `outputs/customer_features.csv`
- `outputs/customer_segments.csv`
- `outputs/lapse_predictions.csv`
- `outputs/high_risk_customers.csv`

## Part C: Support Ticket Triage

### Ticket Classification

The project uses TF-IDF text features and Logistic Regression to classify support tickets into:

- Delivery
- Billing
- Fraud
- Technical

The classifier uses 160 usable labeled tickets.

The initial test split achieved 100% accuracy. This result applies only to the small evaluation set and should not be interpreted as guaranteed real-world performance.

### Ticket Prediction

The prediction script demonstrates classification of new support ticket messages.

### Refund Decision Engine

A rule-based refund engine evaluates sample customer support scenarios using order details and refund policy rules.

Possible decisions include:

- REFUND
- DENIED
- ESCALATE
- NOT_FOUND
- FALLBACK

Seven sample scenarios were tested, with all seven expected outcomes matching the engine's output.

### Outputs

- `outputs/ticket_predictions.csv`
- `outputs/refund_decisions.csv`

## Part D: Policy RAG Assistant and Refund Agent

### Policy RAG Assistant

The RAG system retrieves relevant information from policy documents stored in `data/policies/`.

The retrieval pipeline includes:

- Loading policy documents.
- Splitting policy text into chunks with overlap.
- TF-IDF vectorization.
- Cosine similarity ranking.
- Retrieving the top relevant policy chunks.
- Generating an answer using the provided offline mock LLM.

The policy evaluation tested 14 questions against expected answers.

The evaluation reported 14/14 expected-answer matches. This is a simple evaluation metric and does not establish production-level RAG accuracy.

### Refund Agent

The refund agent combines SQLite order lookup, policy retrieval, refund calculation, and simulated refund actions.

The agent supports:

- Order lookup
- Customer/order ownership validation
- Policy search
- Refund calculation
- Refund issuance simulation
- Human escalation
- Fallback handling for unexpected loops

The agent was tested against seven scenarios and matched all seven expected outcomes.

The agent execution ledger is saved as:

`outputs/refund_agent_ledger.jsonl`

**Important:** Refund issuance and escalation are simulated. The agent does not make real payments or contact a human support team.

## Limitations and Next Steps

### Limitations

- The dataset is synthetic and limited in size.
- The lapse prediction target is a proxy, not verified long-term churn.
- Ticket classification needs evaluation on larger and more diverse datasets.
- The RAG evaluation uses a simple expected-answer matching metric.
- Refund request dates are not available for every scenario, so the seven-day eligibility window cannot always be verified.
- The refund agent uses an offline mock LLM and simulated actions.
- High-value refunds and uncertain decisions require human review.
- The system is a learning prototype, not a production customer-support platform.

### Next Steps

- Improve ticket classification using more labeled data.
- Add stronger PII redaction and ticket guardrails.
- Validate customer lapse predictions using verified churn labels.
- Improve RAG retrieval and evaluation.
- Add more refund edge cases and policy checks.
- Integrate real human review workflows in a controlled environment.
- Improve monitoring, testing, and error handling.