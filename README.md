# Quantitative Research Project on  Differential Privacy in Loan Approval Modeling
## Research Question
The primary research question of this project was: How does the introduction of differential privacy techniques impact the predictive performance of logistic regression models in a loan approval dataset?

Differential Privacy is a framework designed to ensure that the privacy of individuals is maintained when analyzing data and training machine learning models. It achieves this by introducing a controlled amount of random noise to datasets or outputs, making it difficult to infer any single individual's information even if an attacker has additional background knowledge.

The DP mechanisms studied in this research are:
* For features ($X_i$): **Laplace mechanism, Truncated Laplace Mechanism, and Gaussian mechanism**
* For response variable ($Y$): **Randomized Response**

## Data
The dataset used for this research was a loan dataset containing 20,000 entries and 35 features. These features included various socioeconomic and financial attributes such as age, annual income, credit score, employment status, education level, loan amount, loan duration, and more. The target variable, LoanApproved, was a binary variable indicating whether a loan was approved (1) or not (0). The dataset is retrieved from Kaggle: https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?resource=download

## Approach
1. Feature selection: : Initial feature selection was performed using Pearson's correlation and Lasso Regression to remove correlated features and identify the most influential predictors for loan approval. This process trims the dataset down from 35 features to 9.
2. Perform sampling with replacement to generate different subsets of data. Draw subsets with sizes ranging from 100 to 5000, increasing in intervals of 100. Repeat this process 100 times for robustness.
3. Standardize each subset using sklearn's *StandardScaler*, then apply three noise mechanisms (Laplace, Gaussian, and Truncated Laplace) with a fixed $\epsilon$ and $\delta$ value.
4. Measure and compare accuracy across sample sizes.

## Findings
* The baseline logistic regression model, trained without noise, consistently showed high accuracy across sample sizes, hovering around 0.98 to 1.0.
* When sample size is smaller than 1000, it is apparent that the Gaussian mechanism yields a better accuracy than Laplace and Truncated Laplace.
* Accuracy between Laplace and Truncated Laplace is not obvious because of the large sample size. The effect of noise diminishes due to the law of large numbers.
* Smaller sample sizes showed more variability in accuracy when differential privacy techniques were applied, while larger samples resulted in more stable performance but with reduced accuracy compared to the baseline.

## Computational Environment
* Programming Language: Python on Google Colab
* Libraries Used:
  * $${\color{greenyellow}pandas, numpy}$$ and $${\color{greenyellow}math}$$ for data manipulation and analysis
  * $${\color{greenyellow}sklearn}$$ for machine learning models, data preprocessing, and evaluation
  * $${\color{greenyellow}matplotlib}$$ for data visualization
  * $${\color{greenyellow}multiprocessing}$$ for parallel processing to improve computational efficiency
