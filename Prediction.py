
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("C:/Users/I-Net Computer/Desktop/AI_Loan/train.csv")

print(df.head(1))

#print(df.describe())

# We have to deal with the non numerical values
# We have to use frequency distribution
print("This is Value count :\n")
df1=df['Property_Area'].value_counts()
print(df1)

#1st take the applicant Income and Loan amount and make a histogram
bins=50
plt.hist(df['ApplicantIncome'], bins=50)
plt.xlabel('Total Income')
plt.ylabel('Total Numbers')
plt.show()
plt.legend()

print(df['ApplicantIncome'])

plt.boxplot(df['ApplicantIncome'])
plt.show()

a=df.boxplot(column='ApplicantIncome', by='Education')
#plt.boxplot(df['ApplicantIncome'], by=df['Education'])
plt.show(a)

#Now chk with the loan amount

#plt.hist(df['LoanAmount'])

ab=df['LoanAmount'].hist(bins=50)
plt.xlabel("Loan Amount")
plt.ylabel("Total count ")
plt.show(ab)

#box plot
abc=df.boxplot(column='LoanAmount')
plt.show(abc)



#categarical variable

temp1=df['Credit_History'].value_counts(ascending=True)

temp2=df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())

print('Credit History Value count: \n')
print(temp1)

print('Pivot Table for Loan Status :\n')
print(temp2)

#cheacking missing values in dataset

print(df.apply(lambda x: sum(x.isnull()),axis=0))

#fill the missing values

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

print(df.apply(lambda x: sum(x.isnull()),axis=0))

df['Self_Employed'].fillna('No', inplace=True)

#Create a Pivot Table

table=df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

print(table)


#Define function to return value of this pivot table

def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]

#Replace Missing Values

#print("Replace Missing Values from pivot table: \n ")
#df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)909
print(df.apply(lambda  x: sum(x.isnull())))

print("My dataset: \n")
print(df.head())

print(df['Gender'].value_counts())

df['Gender'].fillna('Male', inplace=True)
print("filling Gender Missing values \n")
print(df.apply(lambda  x: sum(x.isnull())))
df['Married'].fillna('Yes', inplace=True)

print("Dataframe Datatypes: \n")
#df1=df['Dependents'].convert_objects(convert_numeric=True)

#pd.to_numeric(df['Dependents'], errors='coerce')
df1=df.drop(['Dependents'], axis=1)
print('df1 Data:\n')
print(df1.dtypes)

print(df1.head())

#df['Dependents'].astype(int)

#df['Dependents'].fillna(df['Dependents'].mean(), inplace=True)

#df['Education'].fillna('Graduate', inplace=True)
#df['Self_Employed'].fillna('No', inplace=True)
df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term'].mean(),inplace=True)
df1['Credit_History'].fillna(df1['Credit_History'].mean(), inplace=True)

print("After Treating All Missing Values: \n ")
print(df1.apply(lambda  x: sum(x.isnull())))  #show all missing values

print("Preeti Abhijit : ")
print(df1.dtypes)

#How to Treat Extream Values:


df['LoanAmount_log']=np.log(df['LoanAmount'])
b=df['LoanAmount_log'].hist(bins=20)
plt.show(b)

#Combine Applicant And co-applicant Income to treat with extream values

df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['Total_Income_Log']=np.log(df['TotalIncome'])
c=df['Total_Income_Log'].hist(bins=20)
plt.xlabel('Total + Income')
plt.show(c)

#Building a Predictive Model

#use sklearn library
#Step 1: Convert all categorial variable in to numrical
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df1[i]=le.fit_transform(df1[i])

#le.fit_transform(df1['Gender'])
#le.fit_transform(df1['Married'])

print(df1.dtypes )
#print("After Transform in to numrical: \n")
#print(df1['Gender'])
print("This is data after cleaning : \n")
print(df1.head())


#Use scikit learn:
from sklearn.linear_model import LogisticRegression

#for K-Fold Cross Validation
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn import metrics


#Generic Function for making a classification model and accesing peformance:

def classification_model(model,data,predictors,outcome):
    # 1. fit the model
    model.fit(data[predictors],data[outcome])  #fit the model

    # 2. Make predictions on training set
    predictions=model.predict(data[predictors])

    #3. print accuracy
    accuracy=metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy:  %s" % "{0:3%}".format(accuracy))


    #4. perform k-fold cross-validation with 5 folds
    kf=KFold(data.shape[0], n_folds=5)
    error1=0

    for train, test in kf:
        #filter traning data
         train_predictors= (data[predictors].iloc[train,:])

        #The target we're using to train the algorithm
         train_target=data[outcome].iloc[train]

        #Training algorith using the predictors and target
         model.fit(train_predictors,train_target)

       #Record error from earch cross-validation run
         error1.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

         print("Cross Validation Score : %s" % "{0:.3}".format(np.mean(error1)))


    #fit the model again so that it can be refered by outside function
         model.fit(data[predictors],data[outcome])




#Now use the different machine learning algorithm to deal with the dataframe

#1. LOGISTIC REGRESSION

#In this case of applying logistic regression if we use all the variable at time for the prediction then model may have overfitting problem so to avoide this we can take some variables for the predictions

#As we know the / We observed that the :
#Higher chance of getting loan
#1.Applicant with Credit History
#2. Applicant with Higher income & Co-applicant income
#3. Applicant with Education History
#4. Applicant with urban area property

#So, Lets first make our model with 'Credit_History'

outcome_var='Loan_Status'
model=LogisticRegression()
predictor_var=['Credit_History']
classification_model(model, df1, predictor_var, outcome_var)


#We can try with different combination of variables:


redictor_var=['Credit_History', 'Education', 'Married', 'Self_Employed', 'Propery_Area']
classification_model(model,df1,predictor_var,outcome_var)




