# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: A.LAHARI
RegisterNumber:  212223230111
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:

![image](https://github.com/user-attachments/assets/678afa42-87eb-4145-9aaa-c7071ea1d923)

![image](https://github.com/user-attachments/assets/09c1870b-0ec0-423b-b57d-31097dca9cc2)

![image](https://github.com/user-attachments/assets/8eff3339-af4d-4043-909c-17d8c1147858)

![image](https://github.com/user-attachments/assets/ac7d83d1-bd0e-4272-98e9-988cfa8e7ba1)

![image](https://github.com/user-attachments/assets/c985209a-3c87-437f-8a6d-e7816d5b39d9)

![image](https://github.com/user-attachments/assets/9e327798-79dc-4da2-8fba-23123e4b1afd)

![image](https://github.com/user-attachments/assets/4f8291a5-faad-4324-acf7-41e59e624e27)

![image](https://github.com/user-attachments/assets/9b39c90c-f9d4-47c3-b2c2-534c06c9cca7)

![image](https://github.com/user-attachments/assets/5bc874b8-50cb-4521-bc73-65fb09237e12)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
