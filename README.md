# Bankruptcy Prediction
 *By integrating Instance Hardness(IHT) based Under sampling and Supervised Learning Methods in highly Imbalanced dataset and SHAP Interpretations*

Bankruptcy prediction is an essential task in the financial industry, as it can help identify companies at risk of insolvency and take necessary actions to prevent or mitigate potential losses. However, the imbalanced nature of the dataset, where the number of bankrupt companies is significantly lower than the non-bankrupt ones, poses a significant challenge for traditional supervised learning methods. The classification task gets more challenging when the dataset contains both class imbalance and class overlap. The problem is further aggravated when the dataset is high dimensional having many independent variables.

## Introduction
This study focuses on overcoming the above difficulties faced due to imbalanced data, class overlapping and high dimensional data. Feature selection using various statistical techniques such as information gain, f-score, chi-square and machine learning techniques such as embedded tree based method are explored. These selected features are further used to explore various supervised classification techniques such as SVM, Xgboost, Isolation Forest and tree based algorithms specially adapted for imbalanced data. We also further enhanced the performance of these models by using Instance Hardness Threshold (IHT) based under sampling method. IHT is used to select the most difficult-to-classify instances from the majority class, which are then removed and remaining instances combined with the minority class instances to form a balanced dataset. We do comparative evaluation of performance of the models using several metrics, including accuracy, precision, recall, F1-score, and balanced accuracy with and without IHT under-sampling for above mentioned supervised techniques.

### **Objectives**

The study proposes to achieve following objectives :
1.	To explore the instance hardness based under-sampling method in highly imbalanced dataset for classification using supervised learning method
2.	To compare the performance of supervised learning methods designed to handle highly imbalanced bankruptcy dataset.
3.	To compare and analyze the performance of supervised learning methods with and without instance hardness based under-sampling method.
4.	To combine various feature selection techniques to reduce the dimensionality of dataset.
5. To interpret model predictions using SHAP values- Force plots, summary plots , Density plots.
