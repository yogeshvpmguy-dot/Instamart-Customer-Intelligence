

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

â”œâ”€â”€ data/

â”‚   â”œâ”€â”€ raw/

â”‚   â”œâ”€â”€ clean/

â”‚   â””â”€â”€ policies/

â”œâ”€â”€ outputs/

â”œâ”€â”€ src/

â””â”€â”€ README.md

```



## 4. Technologies Used



- Python

- CSV data processing

- scikit-learn

- TF-IDF text vectorization

- Logistic Regression

- Rule-based decision logic

- PowerShell



## 5. Data Preparation and Customer Intelligence



The project contains raw and cleaned datasets for customers, orders, products, and support tickets.



Customer-level outputs include:



- Total and completed orders

- Cancelled orders

- Total spending and average order value

- Cancellation rate

- Order recency



These features support exploratory customer analysis. The supplied customer data does not contain a verified churn label, so these features should not be presented as a validated churn prediction model.



## 6. Support Ticket Classification



A TF-IDF vectorizer and Logistic Regression classifier categorize ticket messages into:



- Delivery

- Billing

- Fraud

- Technical



The training script is `src/ticket\_classifier.py`.



The interactive sample-message test script is `src/predict\_ticket.py`.



The test dataset contains 160 usable labeled tickets. A previous 40-ticket test split achieved 100% accuracy. This result is limited to the supplied dataset and does not guarantee performance on real-world customer messages.



## 7. Refund Decision Engine



The script `src/refund\_engine.py` applies refund rules to the supplied agent scenarios.



The prototype considers:



- Whether an order exists

- Customer and order matching

- Order status

- Delivery duration

- Refund amount

- Human approval for high-value refunds

- Fallback cases



Seven supplied scenarios were tested, and all seven outcomes matched their expected outcomes.



The dataset does not provide sufficient delivery-date information to verify the seven-day refund request window. This limitation must be addressed before production use.



## 8. Outputs



Generated output files include:



- `outputs/customer\_order\_summary.csv`

- `outputs/customer\_activity\_summary.csv`

- `outputs/customer\_features.csv`

- `outputs/ticket\_predictions.csv`

- `outputs/refund\_decisions.csv`



## 9. How to Run



Open PowerShell in the project root directory.



Install the required machine learning library:



```powershell

py -m pip install scikit-learn

```



Run the ticket classifier evaluation:



```powershell

py .\\src\\ticket\_classifier.py

```



Test new support messages:



```powershell

py .\\src\\predict\_ticket.py

```



Run the refund decision engine:



```powershell

py .\\src\\refund\_engine.py

```



Note: The scripts require the project datasets to be present at the paths referenced in the code. The local Windows environment may have application-control restrictions affecting some Python packages.



## 10. Limitations and Future Improvements



- Evaluate the classifier on larger and more diverse customer messages.

- Add robust confidence thresholds and human review for uncertain predictions.

- Integrate fraud escalation and payment-policy handling.

- Validate the refund request date window when suitable data is available.

- Build a retrieval-based assistant grounded in the supplied policy documents.

- Add end-to-end tests and reproducible environment setup instructions.

- Evaluate customer churn only when a suitable target label or defensible proxy definition is available.



## 11. Responsible Use



This is a learning prototype, not a production customer-support system. Refund decisions and fraud-related cases should be reviewed against the complete policy and escalated to a human when required. Personal information should be masked before automated processing.



## 12. Project Status



Core data preparation, customer feature generation, ticket classification, and sample refund decision logic have been implemented. Additional integration, testing, documentation, and repository preparation remain in progress.




