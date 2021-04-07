#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


data2 = pd.read_csv("C:/Users/vinay/Downloads/50_Startups.csv")


# In[3]:


data2.info()


# In[4]:


data2.isnull().sum()


# In[5]:


data2.corr()


# In[6]:


#pair plot
sns.set_style(style='darkgrid')
sns.pairplot(data2)


# In[7]:


data2.rename(columns = {'Marketing Spend':'Marketing_Spend','R&D Spend':'R'},inplace = True) ;data2


# In[8]:


model = smf.ols("Profit ~ Marketing_Spend+Administration ",data=data2).fit() 
print(model.tvalues, '\n', model.pvalues)  


# In[9]:


model.summary()


# # Simple Linear Regression Models

# In[11]:


ml_v=smf.ols('Profit~R',data = data2).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues)  


# In[12]:


ml_v=smf.ols('Profit~Administration',data = data2).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues)  


# In[13]:


ml_w=smf.ols('Profit~Marketing_Spend',data = data2).fit()  
print(ml_w.tvalues, '\n', ml_w.pvalues)  


# In[15]:


ml_w=smf.ols('Profit~ R+Marketing_Spend +Administration ',data = data2).fit()  
print(ml_w.tvalues, '\n', ml_w.pvalues)  


# In[20]:


# calculate VIF
profit = smf.ols('Profit~ R+Marketing_Spend +Administration',data=data2).fit().rsquared  
vif_profit = 1/(1-profit)

market = smf.ols('Marketing_Spend~R+Profit+Administration',data=data2).fit().rsquared  
vif_market = 1/(1-profit)

adm = smf.ols('Administration~R+Profit+Marketing_Spend',data=data2).fit().rsquared  
vif_adm = 1/(1-profit)

spend =smf.ols('R~Administration+Profit+Marketing_Spend',data=data2).fit().rsquared  
vif_spend = 1/(1-profit)


# In[22]:


d1 = {'Variables':['Profit','Marketing_Spend','Administration','R'],'VIF':[vif_profit,vif_market,vif_adm,vif_spend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Residual Plot for Homoscedasticity

# In[23]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[24]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[25]:


import statsmodels.api as sm
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Marketing_Spend", fig=fig)
plt.show()


# In[26]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()


# In[27]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()


# # Detecting Influencers/Outliers
# 
# # cook's distance

# In[28]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[29]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data2)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# From the above plot, it is evident that data point 19 and 47 are the influencers

# In[30]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# ## Since the value is <1 , we can stop the diagnostic process and finalize the model

# # High Influence points
# 

# In[31]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[32]:


k = data2.shape[1]
n = data2.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[33]:


leverage_cutoff


# In[34]:


data2[data2.index.isin([19,47])]


# # improving model

# In[39]:


data_new = pd.read_csv("50_Startups.csv")


# In[40]:


#Discard the data points which are influencers and reasign the row number (reset_index())
data_1=data_new.drop(data_new.index[[19,47]],axis=0).reset_index()


# In[41]:


#Drop the original index
data_1=data_1.drop(['index'],axis=1)


# In[42]:


data_1


# In[43]:


data_1.rename(columns = {'Marketing Spend':'Marketing_Spend','R&D Spend':'r&d_Spend'},inplace = True) ;data_1


# # Bulid model

# In[44]:


#Exclude variable "WT" and generate R-Squared and AIC values
final_ml_V= smf.ols('Profit~Marketing_Spend+Administration',data=data2).fit()


# In[46]:


(np.argmax(c),np.max(c))


# Since the value is <1 , we can stop the diagnostic process and finalize the model

# In[47]:


#Check the accuracy of the mode
final_ml_V= smf.ols('Profit~Marketing_Spend+Administration+R',data = data2).fit()


# In[48]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[50]:


prediction_profit=final_ml_V.predict(data2)


# In[51]:


prediction_profit


# In[52]:


data2['prediction_profit']=prediction_profit


# In[56]:


data2.head(10)


# In[ ]:




