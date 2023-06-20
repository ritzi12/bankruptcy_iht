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

## Methodology

The below are the broad steps in this research study:
1. Choose candidate supervised models to conduct testing and comparison of performance. Models were chose based on prediction performance on imbalanced dataset. Candidate supervised algorithms applied are SVM, Xgboost, Isolation Forest, Balanced Random Forest, AdaBoost Classifier with integrated under-sampling, Easy Ensemble. 
2. Clean dataset and perform exploratory data analysis (EDA)
3. Apply feature selection techniques such as F-score, Chi-square, Mutual Information Gain, Tree-based embedded methods – Random Forest Classifier and Extra Trees Classifier.
4. Split the dataset into Train and Test set.
5.	Apply the selected candidate supervised learning models.
6.	Compare their performance on test set using classification metrics such as accuracy, precision, recall, F1-score and balanced accuracy.
7.	Apply the Instance Hardness Threshold based under-sampling method on train dataset to reduce the imbalance in the dataset.
8.	Again apply the selected candidate supervised learning models on this under-sampled dataset
9.	Compare the performance of the models without and without IHT under-sampling. 
10. SHAP Interpretation on any supported tree model out of 6 models used here

![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/9b7cfe1f-d9f9-4bbc-8c4a-45cddfbe7586)

## Dataset 
UCI Machine Learning Repository: Taiwanese Bankruptcy Prediction Data Set. (n.d.). UCI Machine Learning Repository: Taiwanese Bankruptcy Prediction Data Set. https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction

![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/6b3452f1-0d8c-45cb-b910-a76072580e44)

## Feature Selections
![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/12de0739-96fc-4a83-8f57-2d0719ef2f7d)
![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/50b5b006-8da2-4d0b-ad7d-027686bc03ab)
![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/ef6e73be-1dc1-48ed-b3ea-720948d41e3b)

## Findings/ Observations

1. The best performing model without IHT under-sampling in terms of accuracy is Xgboost (97%) followed by AdaBoost RUS (95%). In terms of F1-score for positive class(Bankrupt) AdaBoost RUS (46%) is best followed by Xgboost (45%).
2. Highest precision for Bankrupt class without IHT is Xgboost(49%) and recall for Bankrupt class is highest for Balanced Random Forest (98%)
3. After applying IHT precision for Bankrupt class increases for Isolation Forest (from 22% to 29%) for training dataset
4. Balanced accuracy after applying IHT increases for Xgboost( from 70% to 85%) , Isolation forest (from 62% to 70%), Adaboost RUS ( from 83% to 84%).
5. Average balanced accuracy of all 6 models increased from 75% to 77% after applying IHT under-sampling.

## SHAP Plots
### Force Plot

1. We first find shap values for the selected bankrupt instance .
2. Expected values are simply class probability based on freq of occurence of that class as we know Bankrupt class 1 has 3% prob and Non-bankrupt 97%
3. Plot Force plot of Bnakrupt instance wrt to class 1 shap values then again wrt to class 0 shap values.
4. Shap values for an instance is complementary for 2 classes.
5. Here in first plot wrt to class1 we see base value=0.03 which is pushed higher to 0.05 by factors in red they are -
  * Working Capital/ Total Assets
  * Working Capital/ Equity

  Driven down by blue factors-
  * NetIncome to Total Assests 
  * Cash/ Total Assets

![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/19a0c178-f3c5-4cf3-a9c9-719990155e13)

### Summary Plot for Bankrupt Class

Summary for Single Class - A summary plot for single class gives us density plot.

Below is Summary pplot for Bankrupt class.
* We observe High values of feature Net Income to Total Assets drives the probability down of belonging to Bankrupt class.
* Low Values of Total Debt/ Total net Worth drives down the probability of belonging to Bankrupt class.
  
These are in accordance to the norm that if debt is lower as compared to net worth of company then it has lower tendency to be bankrupt and vice -versa for Net Income to total Assets i.e. High values of net income /assets ratio also signifies lower tendency of company to go bankrupt.
![image](https://github.com/ritzi12/bankruptcy_iht/assets/80144294/70776920-e4d4-4153-b240-1af6361cfd8e)


## Conclusion

* We successfully saw how to step by step break down a high dimensional dataset to a subset of important features and draw insights as to which features impact the bankruptcy position of company.
*  The results demonstrate that Xgboost and AdaBoost RUS are the best performing models in terms of accuracy, while AdaBoost RUS and Xgboost have the highest F1-score for the positive class (bankrupt). 
* After applying IHT under-sampling, the precision for the bankrupt class increases for Isolation Forest.
* Balanced accuracy for all six models increases after applying IHT under-sampling, with the most significant increase observed in Xgboost, Isolation Forest, and Adaboost RUS.
* Using instance hardness threshold (IHT) under-sampling in an imbalanced Taiwanese dataset has been shown to be a promising approach for addressing this challenge. By focusing on the hardest instances in the dataset, this technique has demonstrated the potential to improve the performance of machine learning models for bankruptcy prediction
* The results suggest that IHT under-sampling can effectively reduce the negative impact of class imbalance on model performance, leading to more accurate and reliable predictions. 

* Shap Interpretation allows us to understand how the features contribute to the modle predictions and gain better insight on feature understandings.

## Future Scope 

* There are several potential avenues for future work in bankruptcy prediction using instance hardness threshold (IHT) under-sampling. It would be beneficial to explore the performance of other machine learning algorithms, such as neural networks or decision trees, with IHT under-sampling for bankruptcy prediction.

* Additionally, it would be interesting to investigate the use of ensemble techniques, such as stacking or boosting, which is based on concept of **"Wisdom of the Crowd"** to further improve the performance of the models.

## References

1. A Cluster-Based Boosting Algorithm for Bankruptcy Prediction in a Highly Imbalanced Dataset. Symmetry, 10(7), 250. https://doi.org/10.3390/sym10070250

2. Cost Sensitive Evaluation of Instance Hardness in Machine Learning. Machine Learning and Knowledge Discovery in Databases, 86–102. https://doi.org/10.1007/978-3-030-46147-8_6

