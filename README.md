# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.

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

##  Data:

![image](https://github.com/user-attachments/assets/678afa42-87eb-4145-9aaa-c7071ea1d923)

## Data Head:

![image](https://github.com/user-attachments/assets/09c1870b-0ec0-423b-b57d-31097dca9cc2)

## Null Dataset:

![image](https://github.com/user-attachments/assets/8eff3339-af4d-4043-909c-17d8c1147858)

## Values count in left column:

![image](https://github.com/user-attachments/assets/ac7d83d1-bd0e-4272-98e9-988cfa8e7ba1)

![image](https://github.com/user-attachments/assets/53a0351d-35e2-45ba-8655-4ba347df42c6)

## Dataset transformed head:


![image](https://github.com/user-attachments/assets/c985209a-3c87-437f-8a6d-e7816d5b39d9)

## x.head():

![image](https://github.com/user-attachments/assets/9e327798-79dc-4da2-8fba-23123e4b1afd)

## y.head():

![image](https://github.com/user-attachments/assets/4f8291a5-faad-4324-acf7-41e59e624e27)

## Accuracy:

![image](https://github.com/user-attachments/assets/9b39c90c-f9d4-47c3-b2c2-534c06c9cca7)

## Data prediction:

![image](https://github.com/user-attachments/assets/5bc874b8-50cb-4521-bc73-65fb09237e12)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
