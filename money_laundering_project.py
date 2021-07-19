import os #importing necessary directories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix 
os.chdir("D:\pandas") #changing the working directory
sample_submission=pd.read_csv('sample_submission.csv') #reading sample submission file
test=pd.read_csv('test.csv') #reading the given test file
train=pd.read_csv('train.csv')#reading the given train file
train.isnull().sum()#cheking for the missing values
train.info()#getiing the info about the input values
train.describe()
correlation=train.corr()#finding the correlation between values
print(np.unique(train['Interest_Rate']))#checking the unique input data of interest rate
plt.scatter(train['Annual_Income'],train['Interest_Rate'],c='green')#scatter plot
plt.title('annual income vs interest_rate')
plt.xlabel('annual income')
plt.ylabel('interest rate')
plt.show()
plt.scatter(train['Length_Employed'],train['Interest_Rate'],c='green')#scatter plot
plt.title('length employed vs interest_rate')
plt.xlabel('length employed')
plt.ylabel('interest rate')
plt.show()
sns.boxplot(x='Purpose_Of_Loan',y='Interest_Rate',data=train)#boxplot of purpose of loan vs interest rate
sns.countplot(x='Interest_Rate',data=train)#countplot of interest rate and we get know that outcome 2.0 is more
sns.countplot(x='Gender',data=train)#countplot of gender
sns.countplot(x='Income_Verified',data=train)
sns.regplot(x='Loan_Amount_Requested',y='Interest_Rate',scatter=True,fit_reg=False,data=train)
features = ["Loan_Amount_Requested","Length_Employed","Annual_Income","Home_Owner","Purpose_Of_Loan","Debt_To_Income" , "Inquiries_Last_6Mo","Total_Accounts"]
x = train[features].values
y=train['Interest_Rate'].values
#splitting the data into test and train so that we can use the model
x_train_train,x_train_test,y_train_train,y_train_test= train_test_split(x,y,test_size=0.2,random_state=3)
print(x_train_train.shape,x_train_test.shape,y_train_train.shape,y_train_test.shape)
#impoting random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,max_features=2,max_depth=15,min_samples_split=2,min_samples_leaf=2,random_state=1)
model_rf1=rf.fit(x_train_train,y_train_train)
#predictions for x_train_test
train_predictions_rf1=rf.predict(x_train_test)
accuracy_score(y_train_test,train_predictions_rf1)
confusion_matrix= confusion_matrix(y_train_test,train_predictions_rf1)
#doing the same random forest classifier model for test data
x_testing =["Loan_Amount_Requested","Length_Employed","Annual_Income","Home_Owner","Purpose_Of_Loan","Debt_To_Income" , "Inquiries_Last_6Mo","Total_Accounts"]
x_testingdata = test[x_testing].values
prediction=rf.predict(x_testingdata)
#by removing the interest rate column from submission sample and adding our output from model to it
sample_submission.drop('Interest_Rate',axis=1,inplace=True)
sample_submission.insert(1,"Interest_Rate",prediction.astype(float),True)
#creating the test_prediction file
sample_submission.to_csv('test_prediction.csv',index=False)
