# Portuguese Bank Marketing Classification

Classification analysis of Portuguese bank marketing data to identify customers more likely to subscribe to a term deposit.

This project compares K-Nearest Neighbors, Decision Trees, Bagged Trees, and Random Forests using a common set of quantitative customer and campaign predictors. The analysis focuses not only on overall classification accuracy, but also on performance for the minority subscriber class.

## Business Problem

The dataset represents a Portuguese bank marketing campaign in which customers were contacted and offered a term deposit.

The target variable is:

- `y = yes` : customer subscribed
- `y = no` : customer did not subscribe

Because only about 11.7% of customers subscribed, overall accuracy can be misleading. A model can classify most customers as non-subscribers and still achieve high accuracy while identifying very few actual subscribers.

## Dataset

- 45,211 observations
- 17 original variables
- 5 quantitative predictors used for modeling:
  - `age`
  - `balance`
  - `campaign`
  - `pdays`
  - `previous`
- Target: `y`

The predictor set was restricted to quantitative variables based on the original project requirements.

`duration` was excluded from predictive modeling because it is only known after a marketing call has occurred and would therefore introduce information unavailable at the intended prediction point.

## Modeling Approach

The data was split into:

- 80% training
- 20% holdout testing
- Stratification used to preserve the subscriber/non-subscriber class distribution

Model selection was performed using 5-fold stratified cross-validation on the training data.

For KNN, min-max normalization was implemented inside a preprocessing pipeline so scaling parameters were learned independently within each cross-validation training fold.

Models evaluated:

- K-Nearest Neighbors
- Normalized K-Nearest Neighbors
- Decision Tree
- Bagged Trees
- Random Forest

## Results

| Model | Final Specification | Train Accuracy | Test Accuracy | Subscriber Recall |
| --- | --- | ---: | ---: | ---: |
| KNN | k = 30 | 0.884 | 0.884 | 2.0% |
| Normalized KNN | k = 24 | 0.887 | 0.883 | 7.4% |
| Decision Tree | max_depth = 6 | 0.887 | 0.883 | 7.7% |
| Bagged Trees | 100 trees | 0.992 | 0.868 | 15.0% |
| Random Forest | 10 trees | 0.975 | 0.857 | 13.3% |

The results highlight a tradeoff between overall accuracy and identifying the minority subscriber class.

The Bagged Trees captured the largest share of actual subscribers, while the Decision Tree maintained competitive overall accuracy with substantially greater interpretability.

For the baseline analysis, the Decision Tree was selected as the primary interpretable model.

## Key Findings

- Class imbalance strongly affects model evaluation; accuracy alone does not adequately describe subscriber identification.
- Normalization substantially improved KNN subscriber recall while overall accuracy remained nearly unchanged.
- `pdays` was the most influential predictor in the Decision Tree, followed by age.
- `pdays` must be interpreted carefully because `-1` represents customers who were never previously contacted.
- Ensemble models identified more subscribers but produced more false positives and lower overall accuracy.
- The Decision Tree provides an interpretable way to segment customers based on combinations of prior-contact history, age, balance, campaign activity, and previous contacts.

## Project Files

- [`bank_marketing_classification.ipynb`](bank_marketing_classification.ipynb) : complete Python analysis and modeling workflow
- [`term_deposit_classification_analysis.pdf`](term_deposit_classification_analysis.pdf) : full written analysis and interpretation

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn

## Data Source

Bank Marketing dataset from the UCI Machine Learning Repository.

The raw dataset is not currently included in this repository. Data-source and reproduction instructions will be added separately.
