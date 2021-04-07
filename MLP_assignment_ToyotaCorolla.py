#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sn


# In[15]:


df= pd.read_csv("C:/Users/vinay/Downloads/ToyotaCorolla (1).csv",error_bad_lines=False,encoding = 'ISO-8859-1')


# In[16]:


df


# In[32]:


toyota1= df.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota1.rename(columns = {'Age_08_04': 'Age'},inplace=True)


# In[33]:


eda=toyota1.describe();eda


# In[38]:


plt.boxplot(toyota1["Price"])
plt.boxplot(toyota1["Age"])
plt.boxplot(toyota1["HP"])


# In[42]:


plt.boxplot(toyota1["Quarterly_Tax"])
plt.boxplot(toyota1["Weight"])


# ## All the data is not normally distributed. Price, Age, KM, HP, Quarterly_Tax and Weight have outliers

# In[44]:


plt.hist(toyota1["Price"]) ## This shows that Price is right skewed


# In[45]:


plt.hist(toyota1["Age"]) ## This shows the data is highly left skewed


# In[47]:


plt.hist(toyota1["HP"])## The data is very unevenly distributed, Left skewed


# In[46]:


plt.hist(toyota1["Quarterly_Tax"]) # The data is unevenly distributed, right skewed data


# In[48]:


plt.hist(toyota1["Weight"]) # The data is right skewed.


# ### Doors and Gears are categorical data

# In[50]:


#pair plot
sn.pairplot(toyota1)
correlation_values= toyota1.corr()


# ## Building model1

# In[59]:



import statsmodels.formula.api as smf
m1= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= toyota1).fit()
print(m1.tvalues, '\n', m1.pvalues)


# ## cc and Doors are insignificant

# In[58]:


## building on individual model
m1_cc = smf.ols("Price~cc",data= toyota1).fit()
print(m1_cc.tvalues, '\n', m1_cc.pvalues)


# ## cc is significant

# In[57]:


m1_doors = smf.ols("Price~Doors", data= toyota1).fit()
print(m1_doors.tvalues, '\n', m1_doors.pvalues)


# ## doors is also significant

# In[56]:


m1_to = smf.ols("Price~cc+Doors",data= toyota1).fit()
print(m1_to.tvalues, '\n', m1_to.pvalues)


# ## Both are significant

# # Plotting the Influence plot

# In[60]:


import statsmodels.api as sm
sm.graphics.influence_plot(m1)


# 
# ## removing 80 and 221, where 221 is the next most influencing index

# In[65]:


toyota2= toyota1.drop(toyota1.index[[80]],axis=0)
m2= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= toyota2).fit()
print(m2.tvalues, '\n', m2.pvalues)


# In[68]:


toyota3= toyota1.drop(toyota1.index[[80,221,960]],axis=0)

m4= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = toyota4).fit()
print(m2.tvalues, '\n', m2.pvalues) 


# ## As all the vaiables are significant, we select it as the final model

# In[70]:


#final model
finalmodel = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = toyota3).fit()
print(finalmodel.tvalues, '\n', finalmodel.pvalues) ### 0.885( r squared)


# In[72]:


finalmodel_pred = finalmodel.predict(toyota4);finalmodel_pred


# In[73]:


plt.scatter(toyota4["Price"],finalmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Predicted values")


# ## the observed values and fitted values are linear

# In[74]:


### Residuals v/s Fitted values
plt.scatter(finalmodel_pred, finalmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")


# In[75]:


## histogram--- for checking if the errors are normally distributed or not.
plt.hist(finalmodel.resid_pearson) 


# ## improving model

# In[77]:


from sklearn.model_selection import train_test_split

train_data,test_Data= train_test_split(toyota1,test_size=0.3)


# In[80]:


finalmodel1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = train_data).fit()
print(finalmodel1.tvalues, '\n', finalmodel1.pvalues)


# In[83]:


## prediction
finalmodel_pred = finalmodel1.predict(train_data);finalmodel_pred


# In[84]:


## test prediction
finalmodel_testpred = finalmodel1.predict(test_Data);finalmodel_testpred


# In[90]:


#train residuals
finalmodel_res = train_data["Price"]-finalmodel_pred


# In[91]:


## test residuals
finalmodel_testres= test_Data["Price"]-finalmodel_testpred


# In[92]:


##train rmse
finalmodel_rmse = np.sqrt(np.mean(finalmodel_res*finalmodel_res));finalmodel_rmse


# In[93]:


## test rmse
finalmodel_testrmse = np.sqrt(np.mean(finalmodel_testres*finalmodel_testres));finalmodel_testrmse


# ### train rmse is 1380 and test rmse is 1240

# In[ ]:




