# Bank Marketing Campaign - Classifier Comparison

This project compares the performance of four classification models: **K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Decision Trees**, and **Support Vector Machines (SVM)**, using a dataset from a Portuguese banking institution. The dataset contains results from various telemarketing campaigns, and the goal is to predict whether a client will subscribe to a term deposit.

## Dataset

The dataset used in this project is publicly available from the UCI Machine Learning Repository:

- **Source**: [Bank Marketing Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

The dataset contains the following key features:
- Personal details (e.g., age, job, marital status)
- Contact information (e.g., last contact day, contact method)
- Campaign details (e.g., number of contacts during the campaign)
- Social/economic context (e.g., employment variation rate, consumer price index)
  
The target variable (`y`) is binary and represents whether the client subscribed to a term deposit (`yes` or `no`).

## Project Structure

The project is organized into the following files:

- **bank-full.csv**: The dataset used for training and testing the models.
- **bank-additional.csv**: An additional dataset with social and economic context features.
- **notebook.ipynb**: Jupyter notebook containing the implementation of the classification models and the evaluation metrics.
- **README.md**: This file, providing an overview of the project and instructions for reproducing the results.

## Classifiers Used

Four classifiers were used to evaluate the dataset:

1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Decision Trees**
4. **Support Vector Machines (SVM)**

The classifiers were implemented using the **scikit-learn** Python library.

## Results

Each classifier was evaluated using accuracy and other metrics such as precision, recall, and F1-score. Below are the detailed results:

### K-Nearest Neighbors (KNN)
- **Accuracy**: 89.12%
- **Classification Report**:

            precision    recall  f1-score   support

       0       0.91      0.97      0.94      7952
       1       0.59      0.33      0.43      1091
accuracy                           0.89      9043
macro avg      0.75      0.65      0.68      9043
weighted avg   0.87      0.89      0.88      9043


### Logistic Regression
- **Accuracy**: 88.80%
- **Classification Report**:

            precision    recall  f1-score   support

       0       0.90      0.98      0.94      7952
       1       0.60      0.22      0.32      1091
      *A                           0.89      9043


### Decision Tree
- **Accuracy**: 87.25%
- **Classification Report**:

            precision    recall  f1-score   support

       0       0.93      0.93      0.93      7952
       1       0.47      0.47      0.47      1091
      *A                           0.87      9043


### Support Vector Machine (SVM)
- **Accuracy**: 89.64%
- **Classification Report**:

            precision    recall  f1-score   support

       0       0.91      0.98      0.94      7952
       1       0.67      0.28      0.40      1091
      *A                           0.90      9043

*A = accuracy

## Model Performance Comparison

| Model                | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Weighted Avg F1-Score |
|----------------------|----------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|-----------------------|
| K-Nearest Neighbors   | 89.12%   | 59%                 | 33%              | 43%                | 91%                 | 97%              | 94%                | 88%                   |
| Logistic Regression   | 88.80%   | 60%                 | 22%              | 32%                | 90%                 | 98%              | 94%                | 86%                   |
| Decision Tree         | 87.25%   | 47%                 | 47%              | 47%                | 93%                 | 93%              | 93%                | 87%                   |
| Support Vector Machine| 89.64%   | 67%                 | 28%              | 40%                | 91%                 | 98%              | 94%                | 88%                   |

### Detailed Classification Reports

#### K-Nearest Neighbors (KNN)


## Conclusion

- **SVM** achieved the highest accuracy (89.64%) but had lower recall for the minority class.
- **KNN** also performed well with an accuracy of 89.12%, though its recall for class 1 was lower.
- **Logistic Regression** performed similarly in terms of accuracy (88.80%) but struggled more with recall for the minority class.
- **Decision Tree** had the lowest accuracy (87.25%) but provided a more balanced recall and precision for both classes.

### Recommendations:
- Consider tuning hyperparameters for better performance.
- Address class imbalance using techniques such as oversampling, undersampling, or using class-weighted algorithms.

## Requirements

To run this project, you need the following Python packages:

- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib` (for visualizations if needed)

You can install the dependencies using:

```bash
pip install -r requirements.txt

