#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


#data- www.kaggle.com
df = pd.read_csv("./Files/AQI_city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
df = df.dropna(how="all")

ahmed = df[df["City"] == "Ahmedabad"]
delhi = df[df["City"] == "Delhi"]
mum = df[df["City"] == "Mumbai"]
chen = df[df["City"] == "Chennai"]
hyd = df[df["City"] == "Hyderabad"]
kol = df[df["City"] == "Kolkata"]


kol.head()


# In[5]:


ahmed = ahmed.resample("YS").mean()
mum = mum.resample("YS").mean()
delhi = delhi.resample("YS").mean()
chen = chen.resample("YS").mean()
hyd = hyd.resample("YS").mean()
kol = kol.resample("YS").mean()

mum.dropna(axis=0, thresh=11, inplace=True)
kol


# ## AQI graph:
# 
# Lesser the AQI of the city better is the air quality.
# 
# Incomplete lines represent lack of data for that particular period.
# 

# In[6]:


fig, ax = plt.subplots(figsize=(30,15))

ax.plot(delhi.index, delhi["AQI"], linewidth=3)
ax.plot(chen.index, chen["AQI"], linewidth=3)
ax.plot(hyd.index, hyd["AQI"], linewidth=3)
ax.plot(kol.index, kol["AQI"], linewidth=3)
ax.plot(mum.index, mum["AQI"], linewidth=3)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_xlabel("Year", fontsize=20)
ax.set_ylabel("Air Quality Index(AQI)", fontsize=20)
ax.legend(["Delhi","Chennai","Hyderabad","Kolkata","Mumbai"], loc="upper right", fontsize=20)


# ## Temperature Comparison between the years 2019-2020(Mumbai).

# In[7]:


#data- www.wunderground.com

temp = pd.read_excel("./Files/Temp.xlsx", sheet_name='Sheet1')
temp["Date"] = pd.to_datetime(temp["Date"])
temp = temp.set_index(temp["Date"])
twenty_19 = temp[(temp.index >= pd.datetime(2019, 1, 1)) & (temp.index <= pd.datetime(2019, 12, 31))]
twenty_20 = temp[(temp.index >= pd.datetime(2020, 1, 1)) & (temp.index <= pd.datetime(2020, 12, 31))]

twenty_20


# In[8]:


fig, ax = plt.subplots(figsize=(30, 15))

index = np.arange(0,12,1)
bar_width = 0.35

ax.set_ylabel("Temperature in degree", fontsize=20)
ax.set_xlabel("Date")
ax.bar(index, twenty_19["Avg"], bar_width, color="darksalmon")

ax.set_xlabel("Date", fontsize=20)
ax.bar(index + bar_width, twenty_20["Avg"], bar_width, color="navy" )


ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=20)
ax.legend(("2019","2020"), loc="upper right", fontsize=15)
ax.set_ylim([20, 32])
ax.tick_params(axis='y', labelsize=15)
fig.autofmt_xdate()


# In[ ]:




