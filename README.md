# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer.

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

8.Find the accuracy of the model.

9.End the Program.

## Program:
```python
import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()
data.info()
data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc
```
Program to implement the SVM For Spam Mail Detection.

Developed by: Chandru.M

RegisterNumber:  24900224


## Output:

<img width="509" alt="image" src="https://github.com/user-attachments/assets/e41d6973-494a-4858-9ce1-2100742ffddf" />

<img width="676" alt="image" src="https://github.com/user-attachments/assets/4bc1cef8-4777-441c-abd4-d424ff751ee5" />


<img width="542" alt="image" src="https://github.com/user-attachments/assets/dce2fca5-4b8a-44d2-bf74-8aaab5ac87ed" />

<img width="632" alt="image" src="https://github.com/user-attachments/assets/0dcbe94d-ba24-4d20-90ed-156fd5748f17" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
