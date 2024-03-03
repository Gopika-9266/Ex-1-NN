<H3>ENTER YOUR NAME: Gopika R</H3>
<H3>ENTER YOUR REGISTER NO: 212222240031</H3>
<H3>EX.NO:1</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
df.head()

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated()

df.describe()

print(df['CreditScore'].describe())

df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```


## OUTPUT:
### Dataset:
![exp1-1](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/4f84043b-d498-41a8-9520-1b20eda51f80)

### X-values:
![exp1-2](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/fc69a63e-560f-48bc-a1f5-28e417831a0e)

### Y-values:
![exp1-3](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/b092b374-f4d6-4849-80fb-267156423303)

### Null values:
![exp1-4](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/c3ec9c1c-e46e-4c2f-87f9-f373421bd29e)

### Duplicated Values:
![exp1-5](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/4b684957-738a-44f7-92b5-f11be4b14b2b)

### Description:
![exp1-6](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/d22a24e6-ecda-45ac-ad33-b80381acedab)

### Normalised dataset:
![exp1-9](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/332f858a-9866-4de2-9c4a-eff700768a4c)

### Training data:
![exp1-10](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/f6211622-256f-45fa-ae08-dee207c57d5e)

### Test data:
![exp1-11](https://github.com/Gopika-9266/Ex-1-NN/assets/122762773/5a3785bc-ad54-437c-9e12-e6ec217cf4f2)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


