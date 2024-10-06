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

Each classifier was evaluated using accuracy and other metrics such as precision, recall, and F1-score. Below are the key results:

| Model                | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|----------------------|----------|---------------------|------------------|--------------------|
| K-Nearest Neighbors   | 89.12%   | 59%                 | 33%              | 43%                |
| Logistic Regression   | 88.80%   | 60%                 | 22%              | 32%                |
| Decision Tree         | 87.30%   | 47%                 | 48%              | 48%                |
| Support Vector Machine| 89.64%   | 67%                 | 28%              | 40%                |

### Conclusion

- **SVM** achieved the highest accuracy (89.64%) but struggled with recall for the minority class.
- **KNN** had a good balance of accuracy (89.12%) and recall for the minority class.
- **Logistic Regression** performed well in terms of accuracy but had lower recall for class 1.
- **Decision Tree** had the lowest accuracy but better recall than Logistic Regression.

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
