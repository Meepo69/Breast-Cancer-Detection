# Breast Cancer Detection Using Isolation Forest and K-Means Clustering

## Project Overview

This project focuses on detecting anomalies in breast cancer data and improving clustering performance by using a combination of **Isolation Forest** for anomaly detection and **K-Means clustering** for unsupervised learning. The system aims to identify anomalous samples in the dataset, remove them, and then apply K-Means clustering to evaluate the quality of the clustering.

---

## Features

- **Anomaly Detection**: Uses the Isolation Forest algorithm to detect and remove anomalies from the dataset.
- **K-Means Clustering**: Applies K-Means clustering to the cleaned dataset.
- **Performance Evaluation**: Evaluates the clustering performance using the Silhouette Coefficient, which measures the quality of the clusters.
- **Visualization**: Plots the performance of K-Means clustering after removing anomalies with varying anomaly percentages.

---

## Objectives

1. Detect and remove anomalies in the breast cancer dataset using Isolation Forest.
2. Perform K-Means clustering on the cleaned dataset.
3. Evaluate the performance of K-Means clustering using the Silhouette Coefficient.
4. Visualize the effect of anomaly removal on clustering performance.

---

## Technologies Used

### Frameworks and Libraries
- **Scikit-learn**: For K-Means clustering, Isolation Forest, and Silhouette Score evaluation.
- **Matplotlib**: For plotting the results.
- **NumPy**: For data manipulation and handling arrays.

### Model Architecture
- **Anomaly Detection Model**: Isolation Forest for detecting anomalies.
  - **Isolation Forest**: A model that isolates observations by randomly selecting features and splitting data. It works well for high-dimensional data, identifying anomalies efficiently.
- **Clustering Model**: K-Means clustering for grouping the data into clusters.
  - **K-Means**: A clustering algorithm that partitions the data into a pre-defined number of clusters. It minimizes the variance within each cluster.

### Deployment
- **Environment**: Python (Jupyter notebook or script-based execution).
- **Platform**: Local setup or cloud deployment (AWS, Google Cloud).
- **Libraries**: Python libraries for machine learning (Scikit-learn) and data visualization (Matplotlib).

---

## Dataset and Preprocessing

- **Dataset**: Breast cancer dataset (replace with relevant dataset for the project).
- **Preprocessing**:
  - Anomaly detection using Isolation Forest.
  - K-Means clustering applied to the cleaned dataset.
  - Evaluation using Silhouette Scores for performance assessment.

---

## Training Details

- **Anomaly Detection**:
  - **Algorithm**: Isolation Forest.
  - **Hyperparameters**:
    - Sample size: Variable (based on dataset size).
    - Number of trees: 100.
- **Clustering**:
  - **Algorithm**: K-Means clustering.
  - **Number of clusters**: 2 (can be adjusted based on dataset).
  - **Number of runs**: 10 (to improve the clustering results).

---

## Challenges and Solutions

### Challenges
- **Data Variability**: Variations in data due to possible noise, missing values, or outliers.
- **Model Optimization**: Balancing the trade-off between the accuracy of anomaly detection and clustering with computational efficiency.
- **Interpretability**: Understanding how anomaly detection influences clustering results, particularly when removing anomalies.

### Solutions
- **Handling Data Variability**: Applied imputation techniques to handle missing data and used scaling methods (such as Min-Max) to normalize the data.
- **Model Optimization**: Chose optimal hyperparameters for Isolation Forest and K-Means to balance performance and computational efficiency.
- **Interpretability**: Visualized the performance of K-Means clustering using the Silhouette Coefficient, ensuring that the results were interpretable and actionable.

---

## Future Work

1. **Data Expansion**: Include additional features such as demographic data, tumor size, and more clinical variables.
2. **Cloud Deployment**: Consider deploying the model on cloud platforms like AWS or Google Cloud for scalability and better performance.
3. **Extended Applications**: Adapt the anomaly detection and clustering techniques for other healthcare datasets (e.g., heart disease, diabetes).
4. **Real-Time Forecasting**: Implement real-time data ingestion and anomaly detection in production environments.

---

## How to Run

1. Clone the repository.
2. Set up the environment with the required dependencies.
3. Run the Google Colab notebook for model execution .
4. Input historical NO2 data for real-time predictions.
