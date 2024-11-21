# Quantitative Research Project on  Differential Privacy in Loan Approval Modeling
## Research Question
The primary research question of this project was: How does the introduction of differential privacy techniques impact the predictive performance of logistic regression models in a loan approval dataset?

Differential Privacy is a framework designed to ensure that the privacy of individuals is maintained when analyzing data and training machine learning models. It achieves this by introducing a controlled amount of random noise to datasets or outputs, making it difficult to infer any single individual's information even if an attacker has additional background knowledge.

The DP mechanisms studied in this research are:
* For features ($X_i$): Laplace mechanism, Truncated Laplace Mechanism, and Gaussian mechanism
* For response variable ($Y$): Randomized Response

## Data
The dataset used for this research was a loan dataset containing 20,000 entries and 35 features. These features included various socioeconomic and financial attributes such as age, annual income, credit score, employment status, education level, loan amount, loan duration, and more. The target variable, LoanApproved, was a binary variable indicating whether a loan was approved (1) or not (0). The dataset is retrieved from Kaggle. [https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?resource=download]
