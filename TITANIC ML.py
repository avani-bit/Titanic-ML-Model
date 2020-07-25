#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import numpy as np #scientific computing
import pandas as pd #data analysis


# In[3]:


#importing dataset
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


# In[4]:


#data analysis
train.info()


# In[5]:


train


# In[6]:


train.describe()


# In[7]:


# to check missing values of object datatype 


# In[8]:


total = train.isnull().sum()
total


# In[9]:


#conclusion
# 177 misisng values of age can be filled with random age bw mean and stdage
# 2 missing values of embarked can be filled be most frequent value
# deck no can be obtained from cabin name and rest of the column can be dropped


# In[10]:


# age
data = [train, test]

for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)
# check
train.describe()


# In[11]:


# embarked
train['Embarked'].describe()


# In[12]:


#hence most common value is S
data = [train, test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#check
total = train.isnull().sum()
total


# In[13]:


#cabin
train['Cabin'].describe()


# In[14]:


train['Cabin'].unique().tolist()


# In[15]:


import re
data = [train, test]
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("Z")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
#drop cabin column
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)


# In[16]:


#one hot encoding


# In[17]:


#sex
genders = {"male": 0, "female": 1}
data = [train, test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# In[18]:


#embarked
ports = {"S": 0, "C": 1, "Q": 2}
data = [train, test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[19]:


#converting float value of fare into into
data = [train, test]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[20]:


#deck
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "Z": 8}
data = [train, test]
for dataset in data:
    dataset['Deck']= dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int) #converting float to int


# In[21]:


train


# In[22]:


#feature engineering
#drop name and ticket columns from test and train sets
#drop passengerid from train set
#merge sibsp+parch into a new column family size and drop those columns
#creating a new feature 'fare_per_person'
# categorising age into groups


# In[23]:


#droping columns
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[24]:


#creating new features
data = [train, test]
for dataset in data:
    dataset['Family_Size']=dataset['SibSp']+dataset['Parch']
    dataset['Fare_Per_Person']= dataset['Fare']/(dataset['Family_Size']+1)
    dataset['Fare_Per_Person'].astype(int)


# In[25]:


#categorising age groups
data = [train, test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[26]:


#machine learning model


# In[27]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()


# In[28]:


#random forest
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import SGDClassifier
#from sklearn.tree import DecisionTreeClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc = round(random_forest.score(X_train, Y_train) * 100, 2)
acc


# In[29]:


#k-fold cross validation
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[30]:


#importance feature
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[34]:


#predictions
Y_prediction


# In[35]:


#create submission for kaggle
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Y_prediction})
submission


# In[36]:


#convert submission to csv
filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)


# In[ ]:




