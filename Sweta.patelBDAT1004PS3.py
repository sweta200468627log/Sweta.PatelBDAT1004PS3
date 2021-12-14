#!/usr/bin/env python
# coding: utf-8

# # Question 1

# # step 1 
# Import the necessary libraries 

# In[1]:


import pandas as pd


# # step 2
# Import the dataset from this address.

# In[2]:


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user")
df


# # step 3
# Assign it to a variable called users 

# In[3]:



users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')


# In[4]:


users


# # step 4
# 
# Discover what is the mean age per occupation

# In[5]:


users.groupby('occupation').age.mean()


# # step 5
# Discover the Male ratio per occupation and sort it from the most to the least

# In[6]:


gender_total = users['gender'].value_counts()
gender_occupation = users.groupby('gender')['occupation'].value_counts()
sex_ratio = gender_occupation['M']/gender_occupation['F']
print(sex_ratio.sort_values(ascending=False))


# # step 6
#  For each occupation, calculate the minimum and maximum ages
# 
# 

# In[7]:


print(users.groupby('occupation').age.min())
print(users.groupby('occupation').age.max())


# In[8]:


users.groupby('occupation').age.agg(['min', 'max'])


# # step 7
# For each combination of occupation and sex, calculate the mean age

# In[9]:


users.groupby(['occupation', 'gender']).age.mean()


# # step 8
#  For each occupation present the percentage of women and men?.

# In[10]:


gender_total = users['gender'].value_counts()
gender_occupation = users.groupby('gender')['occupation'].value_counts()
male_sex_ratio = gender_occupation['M']/(gender_occupation['M'] + gender_occupation['F'])*100
female_sex_ratio = gender_occupation['F']/(gender_occupation['M'] + gender_occupation['F'])*100
print(male_sex_ratio,female_sex_ratio)


#  # Question 2
#  Euro Teams Step 1. Import the necessary libraries Step 2. Import the dataset from this address Step 3. Assign it to a variable called euro12 Step 4. Select only the Goal column Step 5. How many team participated in the Euro2012? Step 6. What is the number of columns in the dataset? Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline Step 8. Sort the teams by Red Cards, then to Yellow Cards Step 9. Calculate the mean Yellow Cards given per Team

# # step 1
# Import the necessary libraries 
# 

# In[11]:


import pandas as pd


# # step 2
# Import the dataset from this address

# In[12]:


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv")
df


# # step 3
# Assign it to a variable called euro12

# In[13]:


euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')


# In[14]:


euro12


# # step 4
#  Select only the Goal column

# In[15]:


euro12["Goals"]


# # step 5
#  How many team participated in the Euro2012?

# In[16]:


len(euro12['Team'].unique())


# # step 6
# What is the number of columns in the dataset?

# In[17]:


euro12.shape[1]


# # step 7
# View only the columns Team, Yellow Cards and Red Cards and assign them 
# to a dataframe called discipline

# In[18]:


discipline_Df = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline_Df


# # step 8
#  Sort the teams by Red Cards, then to Yellow Cards

# In[19]:


red_cards = euro12.sort_values(by=['Red Cards', 'Yellow Cards'])
red_cards


# # step 9
# Calculate the mean Yellow Cards given per Team

# In[20]:


euro12["Yellow Cards"].mean()


# # step 10
#  Filter teams that scored more than 6 goals

# In[21]:


euro12['Goals conceded']


# In[22]:


euro12[(euro12['Goals'] >= 6) & (euro12['Goals conceded'] <= 6)]['Team'].values


# # step 11
#  Select the teams that start 
# with G
# 

# In[23]:


euro12.Team.str[0]=='G'


# # step 12
#  Select the first 7 columns

# In[24]:


euro12.iloc[ : , :7]


# # step 13
#  Select all columns except the last 3

# In[25]:


euro12.iloc[ : , :-3]


# # step 14
# Present only the Shooting Accuracy from England, Italy and Russia

# In[26]:


euro12[(euro12['Team'] == 'England') | (euro12['Team'] == 'Italy') | (euro12['Team'] =='Russia')][['Team', 'Shooting Accuracy']]


# In[27]:


teams = ['England', 'Italy', 'Russia']
euro12[euro12['Team'].isin(teams)]


# # Question 3:
# Housing
# Step 1. Import the necessary libraries
# Step 2. Create 3 differents Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000
# Step 3. Create a DataFrame by joinning the Series by column
# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it 
# to 'bigcolumn'
# Step 6. Ops it seems it is going only until index 99. Is it true?
# Step 7. Reindex the DataFrame so it goes from 0 to 299
# 

# # step 1
# Import the necessary libraries

# In[28]:


import pandas as pd


# # step 2 & 3
# Create 3 differents Series, each of length 100, as follows: • The first a random number from 1 to 4 • The second a random number from 1 to 3 • The third a random number from 10,000 to 30,000
# 
# Create a DataFrame by joinning the Series by column 

# In[29]:


import pandas as pd
import numpy as np
import random

# series with numpy linespace()
ser1 = pd.Series(random.randint(1,4) for _ in range(100))
ser2 = pd.Series(random.randint(1,3) for _ in range(100))
ser3 = pd.Series(random.randint(1000,3000) for _ in range(100))
df = pd.concat([ser1, ser2, ser3], axis = 1)
df


# # step 4 
# 
#  Change the name of the columns to bedrs, bathrs, price_sqr_meter

# In[30]:


df.columns = ["bedrs", "bathrs", "price_sqr_meter"]
df


# # Step 5. 
# Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

# In[31]:


df2 = pd.concat([ser1, ser2, ser3])
df2


# # step 6
# Ops it seems it is going only until index 99. Is it true?

# In[32]:


True 


#  # Step 7. 
#  Reindex the DataFrame so it goes from 0 to 299

# In[33]:


df2.reset_index(drop=True)


# # Question 4 :
# 
# Wind Statistics 
# The data have been modified to contain some missing values, identified by NaN.
# Using pandas should make this exercise easier, in particular for the bonus question.
# You should be able to perform all of these operations without using a for loop or 
# other looping construct.
# The data in 'wind.data' has the following format:
# Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL 
# MAL
# 61 1 1 15.04 14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.04
# 61 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 17.54 13.83
# 61 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
# The first three columns are year, month, and day. The remaining 12 columns are 
# average windspeeds in knots at 12 locations in Ireland on that day.

# # step 1& 2
# 
#  Import the necessary libraries
#  
#  Import the dataset from the attached file wind.txt
# 

# In[34]:


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data")
df


# # step 3
# 
# Assign it to a variable called data and replace the first 3 columns by a proper 
# datetime index

# In[35]:


import pandas as pd
data = pd.read_table('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data', sep='\s+', parse_dates=[[0,1,2]])


# In[36]:


data


# # step 4
#  Year 2061? Do we really have data from this year? Create a function to fix it and apply it.

# In[37]:


import datetime
def date_fix(d):
    year = d.year - 100 if d.year > 1989 else d.year
    return datetime.date(year, d.month ,d.day)
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(date_fix)
data


# # step 5
# 
# Set the right dates as the index. Pay attention at the data type, it should be 
# datetime64[ns].

# In[38]:


data["Yr_Mo_Dy"] = pd.to_datetime(data["Yr_Mo_Dy"])
data = data.set_index('Yr_Mo_Dy')
data


# # step 6
# 
# Compute how many values are missing for each location over the entire 
# record.They should be ignored in all calculations below.

# In[39]:


data.isnull().sum()


# # step 7
# Compute how many non-missing values there are in total.

# In[40]:


data.notnull().sum()


# # step 8
# 
# Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
# 
# 

# In[41]:


y = data.mean()
y.mean()


# # step 9
# 
# Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days A different set of numbers for each location.

# In[42]:


data.describe(percentiles=[])


# # step 10
# 
# Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day. A different set of numbers for each day.

# In[43]:


day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis = 1) 
day_stats['max'] = data.max(axis = 1) 
day_stats['mean'] = data.mean(axis = 1) 
day_stats['std'] = data.std(axis = 1)

day_stats.head()


# # step 11
# 
# Find the average windspeed in January for each location.
# 
# Treat January 1961 and January 1962 both as January.

# In[44]:



data.loc[data.index.month == 1].mean()


# # step 12
# Downsample the record to a yearly frequency for each location.

# In[45]:


data.groupby(data.index.to_period('A')).mean()


# # step 13
# Downsample the record to a monthly frequency for each location.

# In[46]:


data.groupby(data.index.to_period('M')).mean()


# # step 14
# 
# Downsample the record to a weekly frequency for each location

# In[47]:


data.groupby(data.index.to_period('W')).mean()


# # step 15
# 
# Calculate the min, max and mean windspeeds and standard deviations of the 
# windspeeds across all locations for each week (assume that the first week starts on 
# January 2 1961) for the first 52 weeks

# In[48]:


first_year = data[data.index.year == 1961]
stats1 = data.resample('W').mean().apply(lambda x: x.describe())
print (stats1)


# # Question 5
# 
# Step 1. Import the necessary libraries Step 2. Import the dataset from this address. Step 3. Assign it to a variable called chipo. Step 4. See the first 10 entries Step 5. What is the number of observations in the dataset? Step 6. What is the number of columns in the dataset? Step 7. Print the name of all the columns. Step 8. How is the dataset indexed? Step 9. Which was the most-ordered item? Step 10. For the most-ordered item, how many items were ordered? Step 11. What was the most ordered item in the choice_description column? Step 12. How many items were orderd in total? Step 13. • Turn the item price into a float • Check the item price type • Create a lambda function and change the type of item price • Check the item price type Step 14. How much was the revenue for the period in the dataset? Step 15. How many orders were made in the period? Step 16. What is the average revenue amount per order? Step 17. How many different items are sold?

# # step 1
# 

# In[49]:


import pandas as pd


# # step 2

# In[50]:


import pandas as pd
df = pd.read_table("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv")
df


# # step 3

# In[51]:


chipo = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv')


# In[52]:


chipo


# # step 4

# In[53]:


chipo.head(10)


# # step 5

# In[54]:


# solution of observation
chipo.shape[0]


# # step 6

# In[55]:


len(chipo.columns)


# # step 7

# In[56]:


chipo.columns


# # step 8

# In[57]:


chipo.index


# # step 9

# In[58]:


item_quants = chipo.groupby(['item_name']).agg({'quantity':'sum'})
item_quants.sort_values('quantity',ascending=False)[:5]


# # step 10

# In[59]:


item_quants = chipo.groupby(['item_name']).agg({'quantity':'sum'})
item_quants.sort_values('quantity',ascending=False)


# # step 11

# In[60]:


item_quants = chipo.groupby(['choice_description']).agg({'quantity':'sum'})
item_quants.sort_values('quantity',ascending=False)[:5]


# # step 12

# In[61]:


chipo.quantity.sum()


# # step 13(a)

# In[62]:


chipo.item_price.str.slice(1).astype(float).head()


# # step 13(b)

# In[63]:


chipo.info()


# # step 13(c)

# In[64]:


lam = lambda x : float(x[1:])
chipo.item_price.apply(lam)[:5]


# # step 13(d)

# In[65]:


chipo['item_price']=chipo.item_price.apply(lam)


# # step 14

# In[66]:


chipo['item_price'].sum()


# # step 15

# In[67]:


chipo.shape


# # step 16

# In[68]:


chipo['item_price'].sum() / 4622


# In[69]:


chipo['item_price'].mean()


# # step 17

# In[70]:


chipo.item_name.nunique()


# # Question 6
# 
# Create a line plot showing the number of marriages and divorces per capita in the 
# U.S. between 1867 and 2014. Label both lines and show the legend.
# Don't forget to label your axes!
# 

# In[71]:


import pandas as pd

df = pd.read_csv("us-marriages-divorces-1867-2014.csv")
df


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("us-marriages-divorces-1867-2014.csv")
marriages_per_capita = data['Marriages_per_1000']
divorces_per_capita = data['Divorces_per_1000']

columns = marriages_per_capita, divorces_per_capita
years = data['Year']
ind = ["marriages per capita", "divorces per capita"]

x_data = range(1867, 1867 + data.shape[0])
fig, ax = plt.subplots()


for i in range(len(columns)):
    ax.plot(x_data, columns[i], label=ind[i])

ax.set_xlabel('Year')
ax.set_ylabel('Marriage/Divorces per 1000 people')
ax.legend()


# # Question 7

# In[73]:



import pandas as pd

df = pd.read_csv("us-marriages-divorces-1867-2014.csv")
df


# In[74]:


marriages_per_capita = data['Marriages_per_1000']
divorces_per_capita = data['Divorces_per_1000']

year = [1900, 1950, 2000]
ind = ["marriages per capita", "divorces per capita"]

specific_data = data.loc[data['Year'] == 1900]
specific_data = specific_data.append(data.loc[data['Year'] == 1950])
specific_data = specific_data.append(data.loc[data['Year'] == 2000])
columns = specific_data['Marriages_per_1000'], specific_data['Divorces_per_1000']


# In[75]:


df = data[['Year','Marriages_per_1000','Divorces_per_1000']]
df


# In[76]:


import matplotlib.pyplot as plt
x = ["1900","1950","2000"]
Marriages = [9.3,11.0,8.2]
Divorces = [0.7,2.5,3.3]

plt.bar(x,Marriages,0.2,label="Marriages")
plt.bar(x,Divorces,0.2,label="Divorces")
       
plt.xlabel("year")
plt.ylabel("Marriages/Divorces people")
plt.legend()
plt.show()


# # Question 8
# 
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort 
# the actors by their kill count and label each bar with the corresponding actor's name.
# Don't forget to label your axes!

# In[77]:


import pandas as pd

Hollywood_actors = pd.read_csv("actor_kill_counts.csv")
Hollywood_actors


# In[78]:


import matplotlib.pyplot as plt

x =Hollywood_actors['Actor'].values
y =Hollywood_actors['Count'].values
plt.xticks(rotation='vertical')
plt.xlabel('Number of kills')
plt.ylabel('Actors')

plt.barh(x,y)
plt.show()


# # Question 9
# 
# Create a pie chart showing the fraction of all Roman Emperors that were assassinated. Make sure that the pie chart is an even circle, labels the categories, and shows the percentage breakdown of the categories.

# In[79]:


import pandas as pd
df = pd.read_csv("roman-emperor-reigns.csv")
df


# In[80]:


val = df.loc[df['Cause_of_Death'] == 'Assassinated']
val = val.append(df.loc[df['Cause_of_Death'] == 'Possibly assassinated'])
v = len(val)/len(df)*100
plt.pie([v, 100-v], labels=['Assissanated', 'Not Assissanated'], autopct='%1.2f%%')


# In[81]:


df = pd.read_csv("roman-emperor-reigns.csv")
val = df.loc[df['Cause_of_Death'] == 'Assassinated']
roman_emperor_reign_lengths = df['Length_of_Reign'].values

val = val.append(df.loc[df['Cause_of_Death'] == 'Possibly assassinated'])
plt.pie(val['Length_of_Reign'], labels=val['Emperor'], autopct='%1.2f%%')


# # Question 10
# 
# Create a scatter plot showing the relationship between the total revenue earned by 
# arcades and the number of Computer Science PhDs awarded in the U.S. between 
# 2000 and 2009.
# 

# In[82]:


import pandas as pd
arcade_revenue_vs = pd.read_csv("arcade-revenue-vs-cs-doctorates.csv")
arcade_revenue_vs


# In[83]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


arcade_revenue_cs_doctorates = pd.read_csv('arcade-revenue-vs-cs-doctorates.csv')
arcade_revenue = arcade_revenue_cs_doctorates['Total Arcade Revenue (billions)'].values
cs_doctorates_awarded = arcade_revenue_cs_doctorates['Computer Science Doctorates Awarded (US)'].values

# arcade_revenue_cs_doctorates

fig, ax = plt.subplots()

colors = cm.rainbow(np.linspace(0, 1, len(arcade_revenue_cs_doctorates['Year'])))

for i in range(len(arcade_revenue_cs_doctorates['Year'])):
    ax.scatter(arcade_revenue[i], cs_doctorates_awarded[i],color=colors[i])

ax.set_xlabel('Total Arcade Revenue (billions)')
ax.set_ylabel('Computer Science Awarded (US)')


# In[ ]:




