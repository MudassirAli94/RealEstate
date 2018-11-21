
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale

from scipy.stats import randint
from scipy.stats import skew


from keras.layers import Dense
from keras.models import Sequential

import xgboost as xgb

import math as math

import seaborn as sns

import h2o
from h2o.automl import H2OAutoML


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train.head()

#The data is mixed with numerical data and character data
# In[4]:


def extract():
    train_object = []
    train_int = []
    for n in range(0,train.shape[1]):
    
        if train[(train.columns[n])].dtype == "O":
            train_object.append(train[(train.columns[n])])
            #print(pd.DataFrame(train_object))
        elif train[train.columns[n]].dtype != "O":
            train_int.append(train[train.columns[n]])
            #print(pd.DataFrame(train_int))
        else:
            pass
        
        
        extract.train_object = (pd.DataFrame(train_object)).T
        extract.train_int = (pd.DataFrame(train_int)).T


# In[5]:


extract()

#Now we can graph the data to see how the missing values compare with the numerical data and the characters data
# In[6]:


#plt.subplot(2,1,1)

extract.train_object.isnull().sum().plot.bar()
plt.title("Missing Values for Object Data Types")
plt.show()
#plt.subplot(2,1,2)
extract.train_int.isnull().sum().plot.bar()
plt.title("Missing Values for Integer Data Types")
plt.show()


# In[7]:


# train.select_dtypes(include = ["object"])

#We can graph the remodeled status of the house to give us a visualization
# In[8]:


remodeled = pd.DataFrame({"No":(train.YearBuilt == train.YearRemodAdd).sum() , "Yes":(train.YearBuilt != train.YearRemodAdd).sum()}, index = [0])


# In[9]:


(train.YearBuilt == train.YearRemodAdd).sum()
#this is a new house so it is NOT remodeled


# In[10]:


(train.YearBuilt != train.YearRemodAdd).sum()

#this is an old house so it is remodeled


# In[11]:


remodeled.unstack().plot.bar()
plt.xticks([0,1],["No" , "Yes"])
plt.title("Remodeled Status")
plt.show()


# In[12]:


train.duplicated().sum()

#duplicated shows us if there are any duplicated rows in the data frame

#We will make two seperate dataframes, one to make into category then make dummy variables and one to make it numeric
# In[13]:


train_cat = extract.train_object.apply(pd.Categorical)
train_num = extract.train_int

#Now we can graph the categories (bar plots) and the numeric values (line graphs)
# In[14]:


def plot(x):
    
    train_cat[x].value_counts().plot.bar()
    plt.title(train_cat[x].astype(str).name)
    plt.show()
    return ""


# In[15]:


for n in train_cat.columns:
    print(plot(n))


# In[16]:


def plot_num(x):
    
    train_num[x].plot.kde()
    plt.title(train_num[x].astype(str).name)
    plt.show()
    return ""


# In[17]:


for n in train_num.columns:
    print(plot_num(n))


# In[18]:


train_cat = train_cat.astype("object")
train_cat = train_cat.fillna("Not Available")
train_cat = train_cat.apply(pd.Categorical)

#There are some N/A values so we can replace them with the mean and for the LotFrontage we can use linear regression to predict those N/A values
# In[19]:


train_num.MasVnrArea = train_num.MasVnrArea.fillna(math.ceil(np.mean(train_num.MasVnrArea)))
train_num.GarageYrBlt = train_num.GarageYrBlt.fillna(math.ceil(np.mean(train_num.GarageYrBlt)))


# In[20]:


train_num.LotFrontage = train_num.LotFrontage.fillna(999999)


# In[21]:


X_test = train_num[train_num.LotFrontage == 999999].drop("LotFrontage" , axis = 1)
X_train = train_num[train_num.LotFrontage != 999999].drop("LotFrontage" , axis = 1)
y_test = train_num.LotFrontage[train_num.LotFrontage == 999999]
y_train = train_num.LotFrontage[train_num.LotFrontage != 999999]


# In[22]:


steps = [('scaler' , StandardScaler()),('LinearRegression' , linear_model.LinearRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train , y_train)
y_pred = (pipeline.predict(X_test))


# In[23]:


abc = (y_pred)
train_num.LotFrontage[train_num.LotFrontage == 999999] = abc

#If we want to see the proper correlation on the numerical data we can use a chart and a heat map to show the correlation
# In[24]:


train_num.LotFrontage = train_num.LotFrontage.astype("int")


# In[25]:


train_num.corr().style.background_gradient().set_precision(2)


# In[26]:


sns.heatmap(train_num.corr() , square = True , cmap = 'RdYlGn')
plt.show()

#Lastly we can do a scatter plot to see the relationship
# In[27]:


def plot_sale(x):
    
    plt.scatter(train_num["SalePrice"], train_num[x],alpha=0.5)
    plt.title(train_num[x].astype(str).name)
    plt.show()
    
    return " "


# In[28]:


for n in train_num.columns:
    print(plot_sale(n))


# In[29]:


train_cat_dummies = pd.get_dummies(train_cat)

train_cat_dummies.head()


# In[30]:


train_1 = pd.concat([train_cat_dummies , train_num] , axis = 1 )
train_1.head()


# In[31]:


X_sale = train_1.drop("SalePrice" , axis = 1)
X_sale = np.sqrt(X_sale)
y_sale = train_1.SalePrice

X_train, X_val, y_train, y_val = train_test_split(X_sale, y_sale, test_size = 0.3, random_state=42)

#We can start training the train data to verify the accuracy of it
# In[34]:


X_sale = train_1.drop("SalePrice" , axis = 1)
y_sale = train_1.SalePrice

X_train, X_val, y_train, y_val= train_test_split(X_sale, y_sale, test_size = 0.3, random_state=42)

Linear Regression
# In[35]:


steps_1 = [('scaler' , StandardScaler()),('LinearRegression' , linear_model.LinearRegression())]

pipeline_1 = Pipeline(steps_1)

pipeline_1.fit(X_train , y_train)
y_pred = pd.DataFrame((pipeline_1.predict(X_val)))

RMSE = np.sqrt(mean_squared_error(y_val,y_pred ))
print("RMSE: " , RMSE)
MSE = mean_squared_error(y_val,y_pred )
print("MSE: " ,MSE)
print("R^2: " , r2_score(y_val, y_pred))

#We can see the r-squared value is really low, so we are going to try to normalize the data and see if there is a difference 
# In[36]:


def unskew(x):
    
    if skew(train_num[x]) > 0.6:
        train_num[x] = np.sqrt(train_num[x])
    else:
        pass
    return train_num[x]


# In[37]:


train_num_1 = train_num.drop("Id" , axis = 1)


# In[38]:


for n in train_num_1.columns:
    unskew(n)


# In[39]:


train_2 = pd.concat([train_cat_dummies , train_num_1] , axis = 1 )

X_sale_2 = train_2.drop("SalePrice" , axis = 1)
y_sale_2 = train_2.SalePrice

X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_sale_2, y_sale_2, test_size = 0.3, random_state=42)


# In[41]:


steps_2 = [('scaler' , StandardScaler()),('LinearRegression' , linear_model.LinearRegression())]

pipeline_2 = Pipeline(steps_2)

pipeline_2.fit(X_train_2 , y_train_2)
y_pred = pd.DataFrame((pipeline_2.predict(X_val_2)))

RMSE = np.sqrt(mean_squared_error(y_val_2,y_pred ))
print("RMSE: " , RMSE)
MSE = mean_squared_error(y_val_2,y_pred )
print("MSE: " ,MSE)
print("R^2: " , r2_score(y_val_2, y_pred))

#Normalizing the data does not give us much difference so we do not need to do itRidge
# In[42]:


ridge = Ridge(alpha = 0.4 , normalize = True)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
print("RMSE: " , np.sqrt(mean_squared_error(y_val,ridge_pred)))
print("MSE: " , mean_squared_error(y_val,ridge_pred))
print("R^2: " , r2_score(y_val, ridge_pred))

Lasso
# In[43]:


lasso = Lasso(alpha = 0.4, normalize = True)

lasso.fit(X_train,y_train)

lasso_pred = ridge.predict(X_val)
print("RMSE: " , np.sqrt(mean_squared_error(y_val,lasso_pred)))
print("MSE: " , mean_squared_error(y_val,lasso_pred))
print("R^2: " , r2_score(y_val, lasso_pred))

#Our ridge and lasso models give us the best accuracyDecision Tree + GridSearchCV
# In[44]:


param_dist = {'max_depth' : [3,10],
             'max_features' : np.arange(1,50),
             'min_samples_leaf':np.arange(1,50),
             "criterion" :["gini" , "entropy"]}

steps_2 = [('scaler' , StandardScaler()), ("GridSearch" , GridSearchCV(DecisionTreeClassifier(), param_dist , cv = 5 , verbose = 1))]

pipeline_2 = Pipeline(steps_2)
pipeline_2.fit(X_train , y_train)
y_pred_2 = pd.DataFrame(pipeline_2.predict(X_val))
#print(tree.best_params_)
#print(tree.best_score_)
print("RMSE: " , np.sqrt(mean_squared_error(y_val,y_pred_2)))
print("MSE: " ,mean_squared_error(y_val,y_pred_2))
print("R^2: " , r2_score(y_val, y_pred_2))

Keras
# In[45]:


predictors_sale = train_1.drop(["SalePrice"] , axis = 1).values
target_sale = train_1.SalePrice.values
n_cols = predictors_sale.shape[1]

model_data = Sequential()
model_data.add(Dense(100 , activation = 'relu' , input_shape = (n_cols,)))
for n in range (1,100):
    model_data.add(Dense(n,activation = 'relu'))
model_data.add(Dense(1, activation = 'softmax'))

model_data.compile(optimizer = 'adam' , loss = 'mean_squared_error' , 
                  metrics = ['accuracy'])
model_data.fit(predictors_sale , target_sale)


# In[46]:


X_train.iloc[:,1:] = X_train.iloc[:,1:].apply(pd.to_numeric)
y_train = pd.to_numeric(y_train)
X_val.iloc[:,1:] = X_val.iloc[:,1:].apply(pd.to_numeric)
y_val = pd.to_numeric(y_val) 


# In[47]:


xg = xgb.XGBClassifier(objective='reg:linear', n_estimators = 10, seed=1234)
xg.fit(X_train, y_train)

y_pred = xg.predict(X_val)

accuracy =  accuracy_score(y_val, y_pred)
print(accuracy)

Now we will do everything we did for the train dataset on the test dataset to find the prediction of the sale price
# In[48]:


test.head()


# In[49]:


def extract():
    test_object = []
    test_int = []
    for n in range(0,test.shape[1]):
    
        if test[(test.columns[n])].dtype == "O":
            test_object.append(test[(test.columns[n])])
            #print(pd.DataFrame(test_object))
        elif test[test.columns[n]].dtype != "O":
            test_int.append(test[test.columns[n]])
            #print(pd.DataFrame(test_int))
        else:
            pass
        
        
        extract.test_object = (pd.DataFrame(test_object)).T
        extract.test_int = (pd.DataFrame(test_int)).T


# In[60]:


extract()


# In[51]:


extract.test_object.isnull().sum().plot.bar()
plt.title("Missing Values for Object Data Types")
plt.show()
#plt.subplot(2,1,2)
extract.test_int.isnull().sum().plot.bar()
plt.title("Missing Values for Integer Data Types")
plt.show()


# In[52]:


remodeled_test = pd.DataFrame({"No":(test.YearBuilt == test.YearRemodAdd).sum() , "Yes":(test.YearBuilt != test.YearRemodAdd).sum()}, index = [0])


# In[53]:


remodeled_test.unstack().plot.bar()
plt.xticks([0,1],["No" , "Yes"])
plt.title("Remodeled Status")
plt.show()


# In[61]:


test_cat = extract.test_object.apply(pd.Categorical)
test_num = extract.test_int


# In[55]:


def plot_test(x):
    
    test_cat[x].value_counts().plot.bar()
    plt.title(test_cat[x].astype(str).name)
    plt.show()
    return ""


# In[56]:


for n in test_cat.columns:
    print(plot_test(n))


# In[57]:


def plot_num_test(x):
    
    test_num[x].plot.kde()
    plt.title(test_num[x].astype(str).name)
    plt.show()
    return ""
    
for n in test_num.columns:
    print(plot_num_test(n))


# In[62]:


test_cat = test_cat.astype("object")
test_cat = test_cat.fillna("Not Available")
test_cat = test_cat.apply(pd.Categorical)


# In[70]:


test_num.LotFrontage = test_num.LotFrontage.fillna(0)
for n in test_num:
    test_num[n].fillna(math.ceil(np.mean(test_num[n])) , inplace = True)


# In[63]:


test_num.MasVnrArea = test_num.MasVnrArea.fillna(math.ceil(np.mean(test_num.MasVnrArea)))
test_num.GarageYrBlt = test_num.GarageYrBlt.fillna(math.ceil(np.mean(test_num.GarageYrBlt)))

test_num.LotFrontage = test_num.LotFrontage.astype("int64") 


# In[80]:


X_test = test_num[test_num.LotFrontage == 0].drop("LotFrontage" , axis = 1)
X_train = test_num[test_num.LotFrontage != 0].drop("LotFrontage" , axis = 1)
y_test = test_num.LotFrontage[test_num.LotFrontage == 0]
y_train = test_num.LotFrontage[test_num.LotFrontage != 0]


# In[81]:


steps = [('scaler' , StandardScaler()),('LinearRegression' , linear_model.LinearRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_test , y_test)
y_pred = (pipeline.predict(X_test))


# In[82]:


abc = (y_pred)
test_num.LotFrontage[test_num.LotFrontage == 0] = abc


# In[83]:


test_num.corr().style.background_gradient().set_precision(2)


# In[84]:


sns.heatmap(test_num.corr() , square = True , cmap = 'RdYlGn')
plt.show()


# In[85]:


test_cat_dummies = pd.get_dummies(test_cat)


# In[86]:


test_1 = pd.concat([test_cat_dummies , test_num] , axis = 1 )


# In[87]:


test_1["SalePrice"] = 0
test_1.head()


# In[88]:


X_test_sale = train_1.drop("SalePrice" , axis = 1)
y_test_sale = train_1.SalePrice

X_train_test, X_test, y_train_test, y_test = train_test_split(X_test_sale, y_test_sale, test_size = 0.3, random_state=42)

#Linear Regression
# In[89]:


steps_1 = [('scaler' , StandardScaler()),
           ('LinearRegression' , linear_model.LinearRegression()),
           ]

pipeline_1 = Pipeline(steps_1)

pipeline_1.fit(X_train_test , y_train_test)
y_pred = pd.DataFrame((pipeline_1.predict(X_test)))

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: " , RMSE)
MSE = mean_squared_error(y_test,y_pred )
print("MSE: " ,MSE)
print("R^2: " , r2_score(y_test, y_pred))

#Ridge
# In[90]:


ridge = Ridge(alpha = 0.4 , normalize = True)

ridge.fit(X_train_test, y_train_test)
ridge_pred = ridge.predict(X_test)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,ridge_pred)))
print("MSE: " , mean_squared_error(y_test,ridge_pred))
print("R^2: " , r2_score(y_test, ridge_pred))

#Lasso
# In[91]:


lasso = Lasso(alpha = 0.4, normalize = True , max_iter = 1000000)

lasso.fit(X_train_test,y_train_test)

lasso_pred = ridge.predict(X_val)
print("RMSE: " , np.sqrt(mean_squared_error(y_val,lasso_pred)))
print("MSE: " , mean_squared_error(y_val,lasso_pred))
print("R^2: " , r2_score(y_val, lasso_pred))

#Pipeline + GridSearchCV + DecisionTreeClassifier
# In[92]:


param_dist = {'max_depth' : [3,10],
             'max_features' : np.arange(1,50),
             'min_samples_leaf':np.arange(1,50),
             "criterion" :["gini" , "entropy"]}

steps_2 = [('scaler' , StandardScaler()), ("GridSearch" , GridSearchCV(DecisionTreeClassifier(), param_dist , cv = 5 , verbose = 1))]

pipeline_2 = Pipeline(steps_2)
pipeline_2.fit(X_train_test , y_train_test)
y_pred_2 = pd.DataFrame(pipeline_2.predict(X_test))
#print(tree.best_params_)
#print(tree.best_score_)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,y_pred_2)))
print("MSE: " ,mean_squared_error(y_test,y_pred_2))
print("R^2: " , r2_score(y_test, y_pred_2))

#Keras
# In[94]:


from sklearn.preprocessing import scale

predictors_sale = test_1.drop(["SalePrice"] , axis = 1).values
predictors_sale = scale(predictors_sale)
target_sale = test_1.SalePrice.values
target_sale = scale(target_sale)
n_cols = predictors_sale.shape[1]

model_data = Sequential()
model_data.add(Dense(100 , activation = 'relu' , input_shape = (n_cols,)))
for n in range (1,100):
    model_data.add(Dense(n,activation = 'relu'))
model_data.add(Dense(1, activation = 'softmax'))

model_data.compile(optimizer = 'adam' , loss = 'mean_squared_error' , 
                  metrics = ['accuracy'])
model_data.fit(predictors_sale , target_sale)

#H2O/automl
# In[104]:


h2o.init()


# In[108]:


train_h2o = h2o.H2OFrame(train_1)
test_h2o = h2o.H2OFrame(test_1)


# In[109]:


splits = train_h2o.split_frame(ratios = [0.8] , seed = 1234)
train_h = splits[0]
validation = splits[1]


# In[115]:


test_h2o = test_h2o.drop("SalePrice"  , axis = 1)


# In[119]:


validation.columns == train_h.columns


# In[120]:


y = "SalePrice"
aml = H2OAutoML( seed = 1234 , project_name = "orim")
aml.train(y = y , training_frame = train_h , leaderboard_frame = validation)

#There is an error for the automl, I will figure this out lateraml.leaderboard.head()pred = aml.predict(validation[0:])
pred.head()perf = aml.leader.model_performance(validation)
perf


#As we can see lasso & ridge are still the best method that gives us the best accuracy for our predictionsClassification for the price range of the houses, it is difficult to get exact predictions when using regression. However getting a prediction on the price range should be better.
# In[121]:


for n in range(1,5):

    print(n*(min(train.SalePrice) + 144020))


# In[122]:


min(train.SalePrice) , max(train.SalePrice)


# In[123]:


train_1["SaleRange"] = np.nan


# In[124]:


train_1.SaleRange[train_1.SalePrice <= 125000] = 0
train_1.SaleRange[np.logical_and(train_1.SalePrice > 125000, train_1.SalePrice <= 175000 ) == True] = 1
train_1.SaleRange[np.logical_and(train_1.SalePrice > 175000, train_1.SalePrice <= 230000 ) == True] = 2
train_1.SaleRange[train_1.SalePrice > 230000] = 3


# In[125]:


train_1.SaleRange.value_counts()


# In[126]:


train_1.drop("SalePrice" , axis = 1 , inplace = True)

X_range = train_1.drop("SaleRange" , axis = 1)

y_range = train_1.SaleRange


X_train, X_val, y_train, y_val = train_test_split(X_range, y_range, test_size = 0.3, random_state=42)


# In[127]:

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print(accuracy_score(y_val , y_pred))

# KNN
# In[128]:


param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit(X_train , y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred = knn_cv.predict(X_val)

print("accuracy:" , accuracy_score(y_val , y_pred))

# DecisionTreeClassifier
# In[129]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred = tree.predict(X_val)
print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print(accuracy_score(y_val , y_pred))

#Decision Tree + Grid SearchCV
# In[130]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 5 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred = tree_cv.predict(X_val)
print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print(accuracy_score(y_val , y_pred))

#SVC
# In[131]:


clf = SVC()

clf.fit(X_train , y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val , y_pred))
print('accuracy:' , accuracy_score(y_val , y_pred))

# XGBoost
# In[132]:


train_2 = train_1[0:-1].astype("int64")


# In[133]:


train_2.head()


# In[134]:


X_salerange = train_2.drop("SaleRange" , axis = 1)
y_salerange = train_2.SaleRange

X_train, X_val, y_train, y_val = train_test_split(X_salerange, y_salerange, test_size = 0.3, random_state=42)


# In[135]:


import xgboost as xgb
xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train, y_train)

y_pred = xg.predict(X_val)

accuracy =  accuracy_score(y_val, y_pred)
print(accuracy)


# In[148]:


salerange_dmatrix = xgb.DMatrix(data = X_salerange, label = y_salerange)

# Create the parameter dictionary: params
# Xgboost parameters
params = {'learning_rate': 0.05, 
              'max_depth': 4,
              'subsample': 0.9,        
              'colsample_bytree': 0.9,
              'objective': 'multi:softprob',
              'num_class': 4,
              'silent': 1, 
              'n_estimators':100, 
              'gamma':1,         
              'min_child_weight':4} 

#params = {"objective":"reg:logistic", "max_depth":3}
#params = {"objective":"multi:softprob", "max_depth":3, 'num_class': 3, 'eta': 0.3, 'silent': True}#, 'num_round' : 20}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = salerange_dmatrix, params = params, nfold=3, num_boost_round=5, metrics="mlogloss", as_pandas=True, seed=1234)

# Print cv_results
print(cv_results)

# Print the accuracy
#print(((1-cv_results["test-mlogloss-mean"]).iloc[-1]))

#As we can see, making this into a classification problem gave us a better accuracy with xgboost , knn.