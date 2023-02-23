# COVID-19-Virus-Detection-Using-Decision-Trees

This repository contains the code, dataset and, results of using ML and Decision Trees in determining Covid-19 based on Health, Contact and Travel History Data.

This repository is a complement to the study purposed to determine the likelihood of person has contracting the virus given the presence of symptoms, contact, and travel history. The method used for predicting COVID-19 virus infection was decision tree algorithm. Correspondingly, results of the decision tree algorithm was compared to other machine learning algorithms. During the time this study and code was created, vaccines and covid testing kits are yet to be accessible, in hopes to combat the pandemic and lessen its effects solutions and interventions was developed through Artificial Intelligence and Machine Learning. 

## Data Set
Throughout this repository the following data set from https://www.kaggle.com/datasets/hemanthhari/symptoms-and-covid-presence?resource=download was used. This can also be found within the repository

## Results
Using Max Depth 5 - 95.71%
![decision_tree_MaxDepth_5](https://user-images.githubusercontent.com/97860488/220988818-647ad7c2-0df2-4db7-8191-f06d6e231ea7.png)

Using Max Depth 7 - 96.57%
![decision_tree_MaxDepth_7](https://user-images.githubusercontent.com/97860488/220988979-3fa0ee3a-031f-483f-88ee-0ab80818a574.png)

Using Max Depth None - 98.22% 
![decision_tree_MaxDepth_None](https://user-images.githubusercontent.com/97860488/220989026-d15fac07-00e2-458d-94cd-7bc60c2a1454.png)

## Comparing with other ML Algorithms 

  Random Forest Regressor   - 92.39%
  KNN                       - 96.75%
  Naive Bayes               - 76.58% 
  Support Vector Model      - 96.32%
  Logistic Regression       - 96.75%
