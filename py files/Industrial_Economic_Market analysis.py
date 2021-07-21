#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Reading and extracting necessary data using pandas
# 
# 
# 

# In[3]:


#datatset from kaggle
df = pd.read_csv('./Files/NIFTY50_all.csv') 
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
sectors = {"MARUTI":"Automobile", "TATAMOTORS":"Automobile", "HEROMOTOCO":"Automobile", "BAJAJ-AUTO":"Automobile", "EICHERMOT":"Automobile", "M&M":"Automobile", 
          "GRASIM":"Cement", "SHREECEM":"Cement", "ULTRACEMCO":"Cement","ITC":"Cigarettes", 
          "HINDUNILVR":"Consumer Goods","BRITANNIA":"Consumer Goods","NESTLEIND":"Consumer Goods", "TITAN":"Consumer Goods", "ASIANPAINT":"Consumer Goods",
          "ONGC":"Energy", "NTPC":"Energy", "POWERGRID":"Energy", "BPCL":"Energy", "IOC":"Energy", "RELIANCE":"Energy", "GAIL":"Energy",
          "LT":"Engineering", "UPL":"Fertilizer", 
          "AXISBANK":"Financial Services", "BAJAJFINSV":"Financial Services", "BAJFINANCE":"Financial Services", "HDFC":"Financial Services", "HDFCBANK":"Financial Services", "ICICIBANK":"Financial Services", "INDUSINDBK":"Financial Services", "KOTAKBANK":"Financial Services", "SBIN":"Financial Services",
          "HCLTECH":"Information Technology", "INFY":"Information Technology", "TCS":"Information Technology", "TECHM":"Information Technology", "WIPRO":"Information Technology",
          "ZEEL":"Media & Entertainment", 
          "HINDALCO":"Metals & Mining", "VEDL":"Metals & Mining", "JSWSTEEL":"Metals & Mining", "TATASTEEL":"Metals & Mining", "COALINDIA":"Metals & Mining",
          "CIPLA":"Pharma", "DRREDDY":"Pharma", "SUNPHARMA":"Pharma",
          "ADANIPORTS":"Shipping","MUNDRAPORT":"Shipping",
          "BHARTIARTL":"Telecom"
   
         }

df["SECTORS"] = df["Symbol"].map(sectors)
df


# In[4]:


df_2017 = df[(df.index >= pd.datetime(2017, 1, 1)) & (df.index <= pd.datetime(2017, 12, 31))]

cmp_17 = df_2017.groupby(["Symbol"])
companies_17 = cmp_17.resample('MS').mean()
companies_17


# In[5]:


df_2018 = df[(df.index >= pd.datetime(2018, 1, 1)) & (df.index <= pd.datetime(2018, 12, 31))]

cmp_18 = df_2018.groupby(["Symbol"])
companies_18 = cmp_18.resample('MS').mean()
companies_18


# In[6]:


df_2019 = df[(df.index >= pd.datetime(2019, 1, 1)) & (df.index <= pd.datetime(2019, 12, 31))]

cmp_19 = df_2019.groupby(["Symbol"])
companies_19 = cmp_19.resample('MS').mean()
#companies_19


# In[7]:



df_2020 = df[(df.index >= pd.datetime(2020, 1, 1)) & (df.index <= pd.datetime(2020, 12, 31))] 
cmp_20 = df_2020.groupby(["Symbol"])
companies_20 = cmp_20.resample('MS').mean()
#companies_20


# In[8]:


sec_17 = df_2017.groupby(["SECTORS"])
ind_17 = sec_17.resample('Y').mean()
ind_17 = ind_17.reset_index("Date")
ind_17.index


# In[9]:


sec_18 = df_2018.groupby(["SECTORS"])
ind_18 = sec_18.resample('Y').mean()
ind_18 = ind_18.reset_index("Date")
ind_18.index


# In[10]:


sec_19 = df_2019.groupby(["SECTORS"])
ind_19 = sec_19.resample('Y').mean()
ind_19 = ind_19.reset_index("Date")
ind_19.index


# In[11]:


sec_20 = df_2020.groupby(["SECTORS"])
ind_20 = sec_20.resample('Y').mean()
ind_20 = ind_20.reset_index("Date")
ind_20


# ## Visualizing a comparison of sector performance of the years 2019 vs 2020 and 2017 vs 2018

# In[12]:


fig1 = plt.figure(figsize=(25, 15))
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

index = np.arange(0,14,1)
bar_width = 0.35


ax1.set_yticklabels(["100K","200K","300K","400K", "500K", "600K", "700K"], fontsize=15)
ax1.set_ylabel("Turnover in Cr. --->", fontsize=15)
ax1.set_xlabel("Sectors")
for19 = ax1.bar(index, ind_19["Turnover"], bar_width)


ax1.set_xlabel("Sectors", fontsize=15)
for20 = ax1.bar(index + bar_width, ind_20["Turnover"], bar_width )



ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(ind_19.index, fontsize=12)
ax1.legend(("2019","2020"), loc="upper right")
fig1.autofmt_xdate()


# for years 2017-18
ax2.set_yticklabels(["100K","200K","300K","400K", "500K", "600K", "700K"], fontsize=15)
ax2.set_ylabel("Turnover in Cr. --->", fontsize=15)
ax2.set_xlabel("Sectors")
for17 = ax2.bar(index, ind_17["Turnover"], bar_width)


ax2.set_xlabel("Sectors", fontsize=15)
for18 = ax2.bar(index + bar_width, ind_18["Turnover"], bar_width )



ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(ind_17.index, fontsize=12)
ax2.legend(("2017","2018"), loc="upper right")
fig1.autofmt_xdate()


# In[ ]:





# From the above figure we can notice a considerable growth of the following sectors:
# 
# 1. Financial Services
# 2. Pharma
# 3. Telecom
# 
# 
# 

# ## Visualizing Company wise performance for the years 2019 and 2020

# In[13]:


companies_19 = cmp_19.resample('Y').mean()
companies_19 = companies_19.reset_index("Date")
companies_19


companies_20 = cmp_20.resample('Y').mean()
companies_20 = companies_20.reset_index("Date")
companies_20


# In[14]:


fig2 = plt.figure(figsize=(30, 20))
ax1 = fig2.add_subplot(211)
ax1.bar(companies_19.index, companies_19["Turnover"], color="lightgreen")
ax1.set_ylabel("Turnover in Trillion Rs.--->", fontsize=25)
ax1.set_xlabel("Companies", fontsize=25)
ax1.set_xticklabels(companies_19.index, rotation=90, fontsize=20)
ax1.set_yticklabels(["0", "20", "40", "60", "80", "100", "120"], fontsize=20)
ax1.set_title("Average Annual Turnover of NIFTY-50 Companies for the year 2019", fontsize=30)
ax1.tick_params()



ax2 = fig2.add_subplot(212)
ax2.set_ylabel("Turnover in Trillion Rs.--->", fontsize=25)
ax2.set_xlabel("Companies", fontsize=25)
ax2.bar(companies_20.index, companies_20["Turnover"], color="skyblue")
ax2.set_xticklabels(companies_20.index, rotation=90, fontsize=20)
ax2.set_yticklabels(["0", "20", "40", "60", "80", "100", "120"], fontsize=20)
ax2.set_title("Average Annual Turnover of NIFTY-50 Companies for the year 2020", fontsize=30)

fig2.tight_layout()


# ##  Historical data scraping and visulaization 

# In[15]:


#data source- https://in.finance.yahoo.com/

import pandas_datareader as pdr
from datetime import datetime

nifty_index = pdr.get_data_yahoo('^NSEI', datetime(2007, 1, 1), datetime(2021, 3, 1), interval='m')

bse_sensex = pdr.get_data_yahoo('^BSESN', datetime(2007, 9, 30), datetime(2021, 3, 1), interval='m')

bse_sensex


# In[16]:


fig3 = plt.figure(figsize=(30, 15))
ax1 = fig3.add_subplot(211)

ax1.plot(nifty_index.index, nifty_index["Adj Close"], linewidth=4, color="teal")
ax1.set_ylabel("Adjusted Close Index--->", fontsize=22)
ax1.set_xlabel("Year", fontsize = 22)
ax1.set_title("NIFTY Index over the years", fontsize=20)
ax1.annotate("* Global Financial Crisis", xy=(nifty_index.index[8],nifty_index["Adj Close"][8]), xycoords='data', xytext=(nifty_index.index[5], 6000),  fontsize=20)
ax1.annotate("* Covid-19 Pandemic", xy=(nifty_index.index[150],nifty_index["Adj Close"][150]), xycoords='data', xytext=(nifty_index.index[148],8000),fontsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=25) 


ax2 = fig3.add_subplot(212)
ax2.plot(bse_sensex.index, bse_sensex["Adj Close"], linewidth=4, color="indigo")
ax2.set_ylabel("Adjusted Close Index--->", fontsize=22)
ax2.set_xlabel("Year", fontsize = 22)
ax2.set_title("BSE Index Over the years", fontsize=20)
ax2.annotate("* Global Financial Crisis", xy=(bse_sensex.index[8],bse_sensex["Adj Close"][8]), xycoords='data', xytext=(bse_sensex.index[5], 22000),  fontsize=20)
ax2.annotate("* Covid-19 Pandemic", xy=(bse_sensex.index[150],bse_sensex["Adj Close"][150]), xycoords='data', xytext=(bse_sensex.index[148],26000),fontsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20) 



fig3.autofmt_xdate()
fig3.tight_layout()
#for i in range(140, 155):
    #print(nifty_index.index[i] , ":" , nifty_index["Adj Close"][i])


# ## Analysis of unemployment rates before and during the lockdown.

# In[17]:


#data source- https://data.worldbank.org/
#DATA CLEANING

year_wise = pd.read_csv("./Files/india_unemployment_rate.csv")
year_wise["date"] = pd.to_datetime(year_wise["date"])
year_wise = year_wise.set_index("date")
year_wise = year_wise.dropna(axis=1, how="all")
year_wise = year_wise.fillna(value=0)
year_wise.columns = ["Unemployment Rate", "Annual Change"]
#year_wise



xls = pd.ExcelFile('./Files/World_Unemployment.xls')
world_data = pd.read_excel(xls, 'Data')
for i in range(1960, 1992):
    world_data = world_data.drop(str(i), axis=1)
world_data = world_data.drop(world_data.index[0])
world_data = world_data.T
world_data = world_data.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=0)
world_data.rename(index={'Country Name':None},inplace=True)

world_data.index = pd.to_datetime(world_data.index)
world_data.index = world_data.index.fillna("Date")
world_data.columns = world_data.iloc[0]
world_data = world_data[1:]
world_data = world_data.dropna(axis=1, how="all")
#world_data["World"]


# In[18]:


fig, ax = plt.subplots(figsize=(30,15))

ax.plot(year_wise.index, year_wise["Unemployment Rate"], linewidth=4)
ax.plot(world_data.index, world_data["World"], linewidth=4)
ax.legend(("India", "World"), loc="upper left", fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_ylabel("Unemployment rate in %", fontsize=25)
ax.set_xlabel("Year", fontsize=25)
ax.set_title("Unemployment rate of India and the World", fontsize=25)
ax.tick_params(axis='y', labelsize=20)


# In[ ]:




