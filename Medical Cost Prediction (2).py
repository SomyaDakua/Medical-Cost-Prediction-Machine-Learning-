#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ![HT_Healthcare_Big_data_analytics_ThinkstockPhotos-470578316.jpg](attachment:HT_Healthcare_Big_data_analytics_ThinkstockPhotos-470578316.jpg)
# 

# #### The medical insurance dataset contains information about a number of factors that can affect medical expenses, including age, sex, BMI, smoking status, number of children, and region. Our job here is to derive insights from the datasets that contribute to higher insurance costs and help the company make more informed decisions regarding pricing and risk assessment.
# 
# 

# ### Importing libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ### Importing dataset

# In[2]:


Dataset = pd.read_csv("D:\Medical_insurance.csv")


# ### Data preprocessing

# In[3]:


Dataset.head(10)


# In[4]:


Dataset.shape


# In[5]:


Dataset.isnull().sum()


# #### Now that we know,our dataset contains no null values and consists of 2772 records of data across 7 columns we proceed to the next steps

# In[6]:


distinct_values = Dataset['region'].unique()
distinct_values


# #### The dataset consists of two string dtypes which needs to be encoded before splitting into training and testing set.

# In[7]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[8]:


le = LabelEncoder()
Dataset['sex'] = le.fit_transform(Dataset['sex'])
Dataset['smoker'] = le.fit_transform(Dataset['smoker'])
Dataset.head()


# In[9]:


ct = ColumnTransformer(transformers = [('encode',OneHotEncoder(),[5])], remainder = 'passthrough')
d = ct.fit_transform(Dataset)
d=pd.DataFrame(d)
d


# ####  The new one-hot encoded columns are added to the left side, and the original columns are appended on the right due to remainder='passthrough'. So, the order should be: one-hot encoded columns for 'region' (0, 1, 2, 3), followed by the other columns (age, sex, bmi, children, smoker, charges), maintaining the original order.

# In[10]:


X = d.iloc[:, 0:9]
Y = d.iloc[:, -1]
# X being the features and Y being the target.


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
# Splitting into training and test set, where training set contains 80% of data.


# #### Now that we are done with the data preprocessing, we try implementing different kinds of regression models to predict the medical charges and find the accurate one using performance metrics.
# 

# ## Multiple Regression

# In[12]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)


# In[13]:


reg.coef_


# In[14]:


reg.intercept_


# In[15]:


Y_pred = reg.predict(X_test)
np.set_printoptions(precision = 2)
Y_pred_reshaped = Y_pred.reshape(-1, 1)
Y_test_reshaped = Y_test.values.reshape(-1, 1)

print(np.concatenate((Y_pred_reshaped, Y_test_reshaped), axis=1))


# In[16]:


""" We try to predict the medical charges for a male belonging to the southeast region,with 23 years of age,
with a bmi of 32 with 0 kids,whose also a smoker """

reg.predict([[0.0,0.0,1.0,0.0,23,1.0,32,0,1.0]])


# #### The R2 score indicates the percentage of the target variable's variance that can be predicted by the features in our regression model. A higher R2 score suggests that our model is better at explaining the variability in the target variable.

# In[17]:


from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)


# #### However, we use Mean Absolute Error to evaluate the performance of your model. These metrics provide information about the average magnitude of errors in your predictions.

# In[18]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_test, Y_pred)


# #### The MAE value of 4177.04 means that, on average, our model's predictions are off by approximately $4177.04 in terms of the medical charges, In the context of our regression model, a lower MAE is generally better.

# ## Polynomial Regression

# In[19]:


from sklearn.preprocessing import PolynomialFeatures


# In[20]:


# fitting polynomial features of degree 2
pf = PolynomialFeatures(degree = 2)
X2_Train = pf.fit_transform(X_train)
X2_test = pf.fit_transform(X_test)


# In[21]:


Lr2 = LinearRegression()
Lr2.fit(X2_Train,Y_train)
Y_pred2 = Lr2.predict(X2_test)


# In[22]:


"""We try to predict the medical charges for a male belonging to the southeast region,with 23 years of age,with a bmi of 32 
   with 0 kids,
   whose also a smoker,
   which is the same case as before, to compare model accuracy""" 

Lr2.predict(pf.fit_transform([[0.0,0.0,1.0,0.0,23,1.0,32,0,1.0]]))


# In[23]:


Lr2.coef_


# In[24]:


Lr2.intercept_


# #### Checking R2 score.

# In[25]:


r2_score(Y_test, Y_pred2)


# #### Checking Mean Absolute error for polynomial regression.

# In[26]:


mean_absolute_error(Y_test, Y_pred2)


# ## Decision Tree Regression

# In[27]:


from sklearn.tree import DecisionTreeRegressor


# In[28]:


regressor = DecisionTreeRegressor(random_state = 0)


# In[29]:


regressor.fit(X_train,Y_train)


# In[30]:


Y_pred3 = regressor.predict(X_test)


# In[31]:


"""predicting the medical charges for a male belonging to the southeast region,with 23 years of age,with a bmi of 32 
   with 0 kids,
   whose is also a smoker"""
   
regressor.predict([[0.0,0.0,1.0,0.0,23,1.0,32,0,1.0]])


# #### Checking R2 score.

# In[32]:


r2_score(Y_test, Y_pred3)


# #### Checking MAE for Decision tree regression.

# In[33]:


mean_absolute_error(Y_test, Y_pred3)


# ## Random Forest Regression

# In[34]:


from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 10,random_state = 0)
regressor2.fit(X_train,Y_train)


# In[35]:


Y_pred4 = regressor2.predict(X_test)


# In[36]:


"""predicting the medical charges for a male belonging to the southeast region,with 23 years of age,with a bmi of 32 
   with 0 kids,
   whose is also a smoker"""

regressor2.predict([[0.0,0.0,1.0,0.0,23,1.0,32,0,1.0]])


# #### Checking R2 score

# In[37]:


r2_score(Y_test, Y_pred4)


# #### Checking MAE for Random forest regression.

# In[38]:


mean_absolute_error(Y_test, Y_pred4)


# #### Finally, we see that out of all the regression models, Decision tree regression and Random forest regression have the least Mean absolute arror and highest R2 scores, therefore if our aim was to find a model in which  the proportion of the variance in   the medical charges explained by the models is more, then higher R2 suggests a better fit. However, our aim is to attain higher prediction accuracy i.e., closeness of predictions to actual values. Therefore, Decision tree regression is the better suited model.
