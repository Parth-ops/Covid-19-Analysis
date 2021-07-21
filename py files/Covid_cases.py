import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

df1 = pd.read_csv("E:/Python_projects/Mini-Project/covid-19-data-master/public/data/jhu/total_cases.csv")
df1["date"] = pd.to_datetime(df1["date"])
df1 = df1.set_index("date")
# print(df1.head())
india1 = df1["India"]
print(india1)
tc = india1.resample('MS').mean()
# print('')
# print("Total cases of Covid-19 in India : \n\n", tc)

df2 = pd.read_csv("E:/Python_projects/Mini-Project/covid-19-data-master/public/data/jhu/total_deaths.csv")
df2["date"] = pd.to_datetime(df2["date"])
df2 = df2.set_index("date")
# print(df2.head())
india2 = df2["India"]
td = india2.groupby(pd.Grouper(freq='M')).mean()
# print(type(td))
# print('')
# print("Total deaths of Covid-19 in India : \n\n", md)


df3 = pd.read_csv("E:/Python_projects/Mini-Project/covid-19-data-master/public/data/jhu/new_cases.csv")
df3["date"] = pd.to_datetime(df3["date"])
df3 = df3.set_index("date")
# print(df3.head())
india3 = df3["India"]
nc = india3.resample('MS').sum()

# print(type(nc))
# print('')
# print("New cases of Covid-19 in India : \n\n", nc)


df4 = pd.read_csv("E:/Python_projects/Mini-Project/covid-19-data-master/public/data/jhu/new_deaths.csv")
df4["date"] = pd.to_datetime(df4["date"])
df4 = df4.set_index("date")
# print(df4.head())
india4 = df4["India"]
nd = india4.resample('MS').sum()

# print(type(nd))
# print('')
# print("Total deaths of Covid-19 in India : \n\n", md)


#  plotting
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(tc)
ax1.plot(td)

# print(x)
ax1.set_title("Total cases and deaths due to Covid-19")
ax1.set_yticks(np.arange(0, 13000000, 1000000))
ax1.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax1.annotate("157,000", xy=(td.index[13], td[13]), xycoords='data', xytext=(td.index[11], 2000000), arrowprops=dict(arrowstyle="->"))
ax1.annotate("11.1 Mn.", xy=(tc.index[13], tc[13]), xycoords='data', xytext=(tc.index[9], 11300000), arrowprops=dict(arrowstyle="->"))
ax1.set_ylabel("Population in Millions --->")
ax1.set_xlabel("Date")
ax1.legend(["Cases", "Deaths"])
# ax1.set_xticks(tc.index)  #to display all the months on x-axis

ax2.plot(nc)
ax2.plot(nd)

ax2.set_yticks(np.arange(0, 3250000, 250000))
ax2.set_title("New cases and deaths due to Covid-19")
ax2.set_ylabel("Population in Millions --->")
ax2.set_xlabel("Date")
ax2.legend(["Cases", "Deaths"])
ax2.annotate("2,459", xy=(nd.index[13], nd[13]), xycoords='data', xytext=(nd.index[9], 200000), arrowprops=dict(arrowstyle="->"))
ax2.annotate("322,369", xy=(nc.index[13], nc[13]), xycoords='data', xytext=(nd.index[10], 2000000), arrowprops=dict(arrowstyle="->"))
ax2.annotate("2,621,418", xy=(nc.index[8], nc[8]), xycoords='data', xytext=(nd.index[5], 2750000), arrowprops=dict(arrowstyle="->"))
fig.autofmt_xdate()

st.pyplot(fig)
