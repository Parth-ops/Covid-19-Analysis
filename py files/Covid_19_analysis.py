#!/usr/bin/env python
# coding: utf-8

# # Importing Packages
# 

# In[3]:


#import packages
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
import seaborn as sns
import ipywidgets as widgets

from ipywidgets import interact, interactive, fixed, interact_manual

from IPython.core.display import display, HTML


# # Reading Datasets

# In[4]:


#Datasets

death_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
confirmed_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
recovered_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
country_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")


# In[5]:


death_df.head()


# In[6]:


confirmed_df.head()


# In[7]:


recovered_df.head()


# In[8]:


country_df.head()


# # Data cleaning

# In[9]:


#data cleaning -renaming

country_df.columns = map(str.lower, country_df.columns)
recovered_df.columns = map(str.lower, recovered_df.columns)
death_df.columns = map(str.lower, death_df.columns)
confirmed_df.columns = map(str.lower, confirmed_df.columns)


# In[10]:


confirmed_df = confirmed_df.rename(columns = {'province/state': 'state','country/region': 'country'})
recovered_df = recovered_df.rename(columns = {'province/state': 'state', 'country/region': 'country'})
death_df = death_df.rename(columns = {'province/state': 'state', 'country/region': 'country'})
country_df = country_df.rename(columns = {'country_region': 'country'})
                                                                      


# In[11]:



confirmed_df.drop(["state"],axis = 1, inplace = True)
recovered_df.drop(["state"],axis = 1, inplace = True)
death_df.drop(["state"],axis = 1, inplace = True)

sorted_country_df = country_df.sort_values('confirmed', ascending=False).head(5)


# In[12]:


sorted_country_df


# # Highlighting the useful columns for plotting

# In[13]:


def highlight_col(x):
    r = 'background-color: red'
    p = 'background-color: purple'
    g = 'background-color: grey'
    temp_df = pd.DataFrame('',index=x.index, columns = x.columns)
    temp_df.iloc[:,4] = p
    temp_df.iloc[:,5] = r
    temp_df.iloc[:,6] = g
    return temp_df
    
sorted_country_df.style.apply(highlight_col, axis=None)


# # Plotting the data in bubble  graph

# In[14]:


fig = px.scatter(sorted_country_df.head(10), x ='country' , y='confirmed' , size='confirmed', color='country', 
                 hover_name="country", size_max=60)

fig.update_layout()
fig.show()


# # Line Graph for confirmed cases Vs confirmed deaths

# In[15]:


def plot_cases_for_country(country):
    labels = ['confirmed', 'deaths']
    colors = ['blue', 'red']
    mode_size = [6,8]
    line_size = [4,5]

    df_list = [confirmed_df , death_df]

    fig = go.Figure()

    for i, df in enumerate(df_list):
        if country == 'World' or country == 'world':
            x_data = np.array(list(df.iloc[:, 5:].columns))
            y_data = np.sum(np.asarray(df.iloc[:, 5:]), axis=0)
            
        else:
            x_data = np.array(list(df.iloc[:, 5:].columns))
            y_data = np.sum(np.asarray(df[df['country']==country].iloc[:,5:]), axis=0)
        
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', 
                             name=labels[i],
                             line = dict(color=colors[i], width=line_size[i]),
                            connectgaps=True,
                            text = "Total " + str(labels[i]) + ": "+ str(y_data[-1])
                            ))
        fig.show()
         

#plot_cases_for_country('India')

interact(plot_cases_for_country, country='World');


# # Bar graph for top 5 worst affected countries (Confirmed Cases)

# In[16]:


px.bar(
    sorted_country_df.head(10),
    x = "country",
    y = "confirmed",
    title= "Top 5 worst affected countries (Confirmed Cases)", # the axis names
    color_discrete_sequence=["green"], 
    height=500,
    width=800
)


# # Bar graph top 5 worst affected countries (Confirmed Cases)

# In[17]:


px.bar(
    sorted_country_df.head(10),
    x = "country",
    y = "deaths",
    title= "Top 5 worst affected countries (Death Cases)", # the axis names
    color_discrete_sequence=["Brown"], 
    height=500,
    width=800
)


# In[18]:


confirmed_df=confirmed_df.dropna(subset=['long'])
confirmed_df=confirmed_df.dropna(subset=['lat'])    

    


# # Plotting the cases on a World map

# In[19]:



world_map = folium.Map(location=[11,0], tiles="cartodbpositron", zoom_start=2, max_zoom= 6, min_zoom=2)

for i in range(len(confirmed_df)):
    folium.Circle(
    location=[confirmed_df.iloc[i]['lat'], confirmed_df.iloc[i]['long']],
    fill = True,
    radius = (int((np.log(confirmed_df.iloc[i,-1]+1.00001)))+ 0.2)*50000,
    fill_color = 'indigo', 
    color = 'red',
    tooltip = "<div style='margin: 0; background-color: black; color: white;'>"+
                    "<h4 style='text-align:center;font-weight: bold'>"+confirmed_df.iloc[i]['country'] + "</h4>"
                    "<hr style='margin:10px;color: white;'>"+
                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
                        "<li>Confirmed: "+str(confirmed_df.iloc[i,-1])+"</li>"+
                        "<li>Deaths:   "+str(death_df.iloc[i,-1])+"</li>"+
                        "<li>Death Rate: "+ str(np.round(death_df.iloc[i,-1]/(confirmed_df.iloc[i,-1]+1.00001)*100,2))+ "</li>"+
                    "</ul></div>",
    ).add_to(world_map)
    
world_map


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




