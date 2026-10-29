

# Instamart Customer Intelligence and Support Co-Pilot

## 1. Project Overview

This project builds a prototype customer intelligence and support system for an online grocery delivery platform inspired by the Instamart use case.

It combines customer data analysis, machine learning-based ticket classification, and a rule-based refund decision engine.

## 2. Project Objectives

- Clean and prepare customer, order, product, and support-ticket datasets.
- Generate customer activity and purchasing features.
- Analyze customer behavior using order history.
- Classify support tickets into delivery, billing, fraud, and technical categories.
- Apply refund policy rules to sample support scenarios.
- Identify cases that require human support review.

## 3. Project Structure

```text
Instamart-Capstone/
├── data/
│   ├── raw/
│   ├── clean/
│   └── policies/
├── outputs/
├── src/
└── README.md
|```

## 4. Technologies Used

- Python
- Pandas
- Scikit-learn

## 5. Customer Data Analysis

Customer and order data are cleaned and transformed into useful features:

- Total orders
- Completed orders
- Cancelled orders
- Total spending
- Average order value
- Cancellation rate
- Recency of customer activity

**Note:** The dataset does not contain a verified churn label. These features support customer behavior analysis, but they are not a validated churn prediction model.

## 6. Support Ticket Classification

The project uses a machine learning pipeline to classify support tickets into four categories:

- Delivery
- Billing
- Fraud
- Technical

TF-IDF converts ticket text into numerical features, and Logistic Regression predicts the ticket category.

The dataset contains 160 usable labeled tickets. The initial test split achieved 100% accuracy, but this result applies only to that small test set and does not guarantee real-world performance.

Run the prediction script with:

    py .\src\predict_ticket.py



## 7. Refund Decision Engine

The refund engine applies rule-based logic to sample customer support scenarios using the available order information and policy rules.

Possible decisions include:

- REFUND
- DENIED
- ESCALATE
- NOT_FOUND
- FALLBACK

The engine evaluates seven sample scenarios and records its decisions in:

    outputs/refund_decisions.csv

**Limitation:** Some scenario dates are unavailable, so the refund eligibility time window cannot always be verified. Human review is important for uncertain or high-value cases.

## 8. How to Run

Open PowerShell in the project folder.

Install the required machine learning library:

    py -m pip install scikit-learn

Run the ticket classifier:

    py .\src\ticket_classifier.py

Run the ticket prediction script:

    py .\src\predict_ticket.py

Run the refund decision engine:

    py .\src\refund_engine.py

Generated results are saved in the outputs/ folder.

## 9. Limitations and Future Improvements

### Current Limitations

- The dataset is limited in size and may not represent real-world customer behavior.
- No verified churn label is available for training a churn prediction model.
- Ticket classification performance needs validation on more unseen data.
- Refund decisions depend on the available order details and policy rules.
- Uncertain cases should be reviewed by a human support agent.

### Future Improvements

- Add verified customer churn labels and build a validated churn prediction model.
- Test the ticket classifier on a larger, more diverse dataset.
- Integrate a retrieval-augmented generation (RAG) system for policy-based support.
- Add stronger fraud detection and human approval workflows.
- Build a dashboard for customer insights and support decisions.

## 10. Responsible Use

This project is a learning prototype. Its predictions and refund decisions should not be used as the sole basis for real customer or financial decisions. Protect personal data and require human review for uncertain or high-impact cases.

## 11. Project Status

Core data preparation, customer feature generation, ticket classification, and rule-based refund decision components have been developed. Further testing and integration are needed before real-world use.