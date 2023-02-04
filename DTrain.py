import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import  seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#second pad
train_df = pd.read_csv('/kaggle/input/predict-cltv-of-a-customer/train_BRCpofr.csv')
test_df = pd.read_csv('/kaggle/input/predict-cltv-of-a-customer/test_koRSKBP.csv')

#Third pad
train_df.head(3)

# Fourth pad
rows,cols=train_df.shape[0],train_df.shape[1]
print('Number of rows in training data:',rows)
print('Number of columns in training data:',cols)

#5Th pad
rows,cols=test_df.shape[0],test_df.shape[1]
print('Number of rows in testing data:',rows)
print('Number of columns in testing data:',cols)

#6th pad
train_df.info()

#7th pad
print('Number of null values in training data:',train_df.isnull().sum().sum())
print('Number of null values in testing data:',test_df.isnull().sum().sum())

#8th pad
print('Unique data types:\n',train_df.dtypes.value_counts())

#9th pad
print('Unique values in training data:\n',train_df.nunique())

#10 pad
train_df = train_df.set_index('id')
test_df = test_df.set_index('id')
print('Setting id as the index column:')
train_df.head(3)
#11 pad
print('Seperating CAtegorical and Numerical columns:')
num = [i for i in train_df.select_dtypes(exclude='object').columns]
cat = [i for i in train_df.select_dtypes(include='object').columns]
print('Categorical Columns:\n',cat)
print('Numerical Columns:\n',num)

#12 pad
########### Function for PLOTTING OBJECT features ################
def cat_plot(feature,data=train_df):
    fig,axes = plt.subplots(1,2,figsize=(15,8))
    ax=axes[0]
    ax=plt.subplot(1,2,1)
    ax = sns.countplot(data=data,x=feature)
    for p in ax.patches:
        ax.text(p.get_x(),p.get_height(),p.get_height())
    plt.xlabel(feature)
    plt.ylabel('Counts')
    plt.title('Counts')
    
    ax=axes[1]
    ax=plt.subplot(1,2,2)
    g = train_df.groupby(feature)['cltv'].median()
    ax = plt.bar(g.index,g.values,color=['blue','red','green','yellow','pink'])
    plt.bar_label(ax,label_type='center')
    plt.xlabel('Classes')
    plt.ylabel('Average clvt for the classes')
    plt.title('Relation with clvt')
    plt.show()
    
    return True
# 13th pad
# Visualise Categorical Features
for feature in cat:
    print(''*50+'For feature {}:\n'.format(feature))
    print('-'*100)
    if cat_plot(feature):
        print('-'*100)
print('#'*48 +'Finished'+ '#'*48)

# 14th pad
########### Function for PLOTTING NUMERIC features ################
import random

def num_plot(feature,data=train_df):
    fig,axes = plt.subplots(1,2,figsize=(15,8))
    ax=axes[0]
    ax=plt.subplot(1,2,1)
    ax = sns.distplot(x=data[feature])
    plt.xlabel(feature)
    plt.ylabel('frequency')
    plt.title('Distribution plot')
    
    color = ['blue','red','green','yellow','pink','orange']
    ax=axes[1]
    ax=plt.subplot(1,2,2)
    ax = plt.scatter(train_df[feature],train_df['cltv'],c=color[random.randint(0,5)])
    plt.xlabel(feature)
    plt.ylabel('CLVT')
    plt.title('Relation with clvt')
    plt.show()
    
    return True
# 15 pad
# Visualise Numerical Features
for feature in num:
    if train_df[feature].nunique()<10:
        print(''*50+'For feature {}:\n'.format(feature))
        print('-'*100)
        if cat_plot(feature):
            print('-'*100)
        
    else:
        print(''*50+'For feature {}:\n'.format(feature))
        print('-'*100)
        if num_plot(feature):
            print('-'*100)
print('#'*48 +'Finished'+ '#'*48)

# 16 pad
# Print Standard Deviation of all numerical features
for i in num:
    print('Standard Deviation of {} is  ----> {} '.format(i,train_df[i].std()))

# 17 pad
#Visualising Correlation matrix
mat = train_df.corr(method='spearman')
sns.heatmap(mat,annot=True,linewidths=1,fmt='.1f')

#18 pad
# Correlation with Categorical values
corr_dict = {}
for col in cat:
    corr = train_df['cltv'].corr(train_df[col], method='spearman')
    corr_dict[col] = corr
    
for feature,value in corr_dict.items():
    print('Correlation of {} with cltv is ----> {}'.format(feature,value))

#19th pad
print('Shape of training data before encoding: ',train_df.shape)
data = train_df.copy()
test = test_df.copy()
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

print('Shape of training data after OneHot encoding: ',train_df.shape)

#20th pad
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso,Ridge

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,RandomizedSearchCV

# 21st pad
X = train_df.drop('cltv',1)
y = train_df['cltv']

# 22nd pad
# Normalising data
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 23rd pad
# 70 : 30 Split from training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=28)

# 24th pad
linear = LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
pred = linear.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))

# 25th pad
rfr = RandomForestRegressor(n_jobs=-1)
rfr.fit(x_train,y_train)
pred = rfr.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))

# 26th pad
xgb = XGBRegressor(n_jobs=-1)
xgb.fit(x_train,y_train)
pred = xgb.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))

# 27 pad
# Replace categorical values with mean cltv
for feature in cat:
    g = data.groupby(feature)['cltv'].mean()
    data[feature] = data[feature].replace(g)

# 28th pad
X = data.drop('cltv',1)
y = data['cltv']

# Normalising data
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 70 : 30 Split from training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=28)

# 29th pad
linear = LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
pred = linear.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))

#30th pad
linear = LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
pred = linear.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))
errors = y - linear.predict(X)
data['Step_1'] = errors

# 31st pad
X = data.drop('cltv',1)
y = data['cltv']

# Normalising data
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 70 : 30 Split from training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=28)

linear = LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
pred = linear.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))
errors = y - linear.predict(X)
data['Step_2'] = errors

# 32nd pad
X = data.drop('cltv',1)
y = data['cltv']

# Normalising data
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 70 : 30 Split from training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=28)

linear = LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
pred = linear.predict(x_test)
print('Accuracy: ',r2_score(y_test,pred))
print('Loss: ',mean_squared_error(y_test,pred))

# 33rd pad
result = pd.DataFrame()

# 34th pad
result['Actual'] = y_test
result['Predicted'] = linear.predict(x_test)
result['Loss'] = result.Actual - result.Predicted

# 35th and final pad
result
