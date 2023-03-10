#!/usr/bin/env python
# coding: utf-8

# ## Python Lottery Dataset Analyze
# 
# Jupyter's notebook illustrating several methods for analyzing a dataset with the historical results of various lotteries.
# The example shows how to analyze the linear correlation between individual fields, using the extension of the result set with astronomical data, and shows how to visualize the distribution of numbers at individual positions.
# 
# (c) 2022 Marcin "szczyglis" Szczygliński
# 
# GitHub page: https://github.com/szczyglis-dev/python-lottery-dataset-analyze
# 
# Email: szczyglis@protonmail.com
# 
# Version: 1.0.0
# 
# This package is licensed under the MIT License.
# 
# License text available at https://opensource.org/licenses/MIT

# In[ ]:


#get_ipython().system('pip install pandas')
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install seaborn')
#get_ipython().system('pip install scipy')
#get_ipython().system('pip install skyfield')


# **1. Configuration, initialization and modules import.**
# 
# Historical drawing results for several popular number lotteries in Poland will be used as input data. The results of the draws will be downloaded to CSV files and saved in a local directory on the disk. The block includes the configuration for each of these lotteries, such as the names of the columns that will be used later in the DataFrame object created from the data set, ranges of numbers, and the format in which the individual drawing dates are saved. At the end of the block, specify the name of the lottery for which the data will be analyzed. The block will also load astronomical data for several celestial bodies, which will then be used to extend the data set with the distances between individual celestial bodies during a given draw. This will be used to test the correlation between these events/variables.

# In[ ]:


import os
import math
from datetime import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from skyfield.api import load

# define URLs with lotteries historical results in CSV
#csv_urls = {
 #   'lotto': 'https://www.wynikilotto.net.pl/download/lotto.csv',
  #  'lotto_plus': 'https://www.wynikilotto.net.pl/download/lotto_plus.csv',
   # 'eurojackpot': 'https://www.wynikilotto.net.pl/download/eurojackpot.csv',
   # 'minilotto': 'https://www.wynikilotto.net.pl/download/mini_lotto.csv',
   # 'multi': 'https://www.wynikilotto.net.pl/download/multi_multi.csv'    
#}

# [CSV config]
# header - list with CSV column names
#   idx - number of record
#   date - date field
#   time - time/hour field
#   n(x) - primary number(x) field
#   m(x) - secondary number(x) field
# n_range - list with primary numbers range [from, to]
# m_range - list with secondary numbers range [from, to]
# n_count - number of primary numbers
# m_count - number of secondary numbers
# date_format - date field string format

csv_config = {
    'lotto': {        
        'header':  ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7'],
        'n_range': [1, 49],
        'm_range': [],
        'n_count': 6,
        'm_count': 0,
        'date_format': '%Y-%m-%d'
    },
    'GVlotto': {
        'header': ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6'],
        'n_range': [1, 49],
        'm_range': [],
        'n_count': 6,
        'm_count': 0,
        'date_format': '%Y-%m-%d'

    },
    '649': {
        'header': ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7'],
        'n_range': [1, 49],
        'm_range': [],
        'n_count': 7,
        'm_count': 0,
        'date_format': '%Y-%m-%d'
    },
    'eurojackpot': {
        'header': ['idx', 'date', 'n1', 'n2', 'n3', 'n4', 'n5', 'm1', 'm2'],
        'n_range': [1, 50],
        'm_range': [1, 12],
        'n_count': 5,
        'm_count': 2,
        'date_format': '%d.%m.%Y'
    },
    'minilotto': {
        'header': ['idx', 'date', 'n1', 'n2', 'n3', 'n4', 'n5'],
        'n_range': [1, 42],
        'm_range': [],
        'n_count': 5,
        'm_count': 0,
        'date_format': '%d.%m.%Y'
    },
    'multi': {
        'header': ['idx', 'date', 'time', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 
                    'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'm1'],
        'n_range': [1, 80],
        'm_range': [1, 80],
        'n_count': 20,
        'm_count': 1,
        'date_format': '%d.%m.%Y'
    }
}

# specify download dir for CSV files
csv_dir = os.path.join(os.getcwd(), 'csv')

# choose lottery
lottery = 'GVlotto'

# init astronomical data
planets = load('de421.bsp')
earth, moon, sun, mars = planets['earth'], planets['moon'], planets['sun'], planets['mars']  


# **2. Functions definitions.**
# 
# The following cell defines the functions that will be used in subsequent blocks. Functions include downloading and saving the data set to CSV and extending the downloaded data set with new values that will then be used for further analysis.

# In[ ]:


# create directory for CSV download if not exists
def csv_dir_create(csv_dir):
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        

# download and save CSV dataset
def csv_update(csv_urls, csv_dir):
    for k, url in csv_urls.items():
        r = requests.get(url, allow_redirects=True)
        name = k + '.csv'
        fname = os.path.join(csv_dir, name)
        open(fname, 'wb').write(r.content)
        print('Downloaded: ' + fname)
        
        
# load CSV dataset        
def csv_load(name, header, csv_dir):
    file = os.path.join(csv_dir, name+'.csv')
    return pd.read_csv(file, header=None, names=header)


# save dataframe to CSV file
def csv_save(df, name, csv_dir):
    file = os.path.join(csv_dir, name+'.csv')
    df.to_csv(file, index=False) 
    

# append date part to series
def df_append_date_part(part, row, dt_format):
    dt = datetime.strptime(row.date, dt_format)
    return int(dt.strftime(part))


# get date parts
def df_get_date_parts(row, dt_format):
    dt = datetime.strptime(row.date, dt_format)
    y = int(dt.strftime('%Y'))
    m = int(dt.strftime('%m'))
    d = int(dt.strftime('%d'))
    return y, m, d


# append astro planets distance to series
def df_append_astro_distance(obj1, obj2, row, dt_format):
    ts = load.timescale() 
    y, m, d = df_get_date_parts(row, dt_format)
    t = ts.utc(y, m, d, 9, 0) 
    return obj1.at(t).observe(obj2).apparent().distance().au


# append numbers ranges to series
def df_append_range(row, num_idx):
    j = 10;
    while j <= 100:
        if row[num_idx] >= (j - 10) and row[num_idx] < j:
            return int((j - 10)/10)
        j+= 10


# **3. Download CSV files with data sets.**
# 
# The cell below will download data sets to CSV files. In order not to download new data and to use only those already downloaded, this block should be commented out.

# In[ ]:


#print('Downloading datasets....')
      
#csv_dir_create(csv_dir)
#csv_update(csv_urls, csv_dir)


# **4. Extend the data set with additional fields.**
# 
# The following code expands the data set with new fields. Numerical values for the saved draw dates will be added to it, such as: year, month, day, day of the week and day of the year. In addition, the distances to individual celestial bodies (Earth - Moon, Earth - Sun, Earth - Mars) that occurred during each of the draws will be calculated and attached to the set. The set will also include fields that define the range of numbers at a given position.

# In[ ]:


# get CSV config for selected lottery
cfg = csv_config[lottery]
dt_format = cfg['date_format']
header = cfg['header']

# load CSV dataset and create Data Frame from it
df = csv_load(lottery, header, csv_dir)

# append date parts as integers
df['year'] = df.apply(lambda row: df_append_date_part('%Y', row, dt_format), axis=1)
df['month'] = df.apply(lambda row: df_append_date_part('%m', row, dt_format), axis=1)
df['day'] = df.apply(lambda row: df_append_date_part('%d', row, dt_format), axis=1)
df['day_of_week'] = df.apply(lambda row: df_append_date_part('%w', row, dt_format), axis=1)
df['day_of_year'] = df.apply(lambda row: df_append_date_part('%j', row, dt_format), axis=1)

# append distances from earth to moon, sun & mars
df['dist_moon_au'] = df.apply(lambda row: df_append_astro_distance(earth, moon, row, dt_format), axis=1)
df['dist_sun_au'] = df.apply(lambda row: df_append_astro_distance(earth, sun, row, dt_format), axis=1)
df['dist_mars_au'] = df.apply(lambda row: df_append_astro_distance(earth, mars, row, dt_format), axis=1)

# append decimal ranges of numbers to n(i)r fields that corresponds numbers at positions n1-n(i)
limit = cfg['n_count']
if limit > 0:
    for i in range(1, limit+1):
        range_field = 'n' + str(i) + 'r'
        num_field = 'n' + str(i)           
        df[range_field] = df.apply(lambda row: df_append_range(row, num_field), axis=1)

# append decimal ranges of numbers to m(i)r fields that corresponds numbers at positions m1-m(i)
limit = cfg['m_count']
if limit > 0:
    for i in range(1, limit+1):
        range_field = 'm' + str(i) + 'r'
        num_field = 'm' + str(i)           
        df[range_field] = df.apply(lambda row: df_append_range(row, num_field), axis=1)

# save extended dataset with appended extra data
csv_save(df, lottery + '_extended', csv_dir)

#df = df.iloc[4424:,:] # you can truncate dataset to period in time


# **5. Linear regression relationship calculation.**
# 
# In the code below, the correlation between the events will be calculated, such as the influence of the distance between celestial bodies on the numbers and the correlation between the individual numbers among themselves. The result for the correlation between the distances of the celestial bodies and the given number will oscillate around 0 here, which will show that these events are not correlated with each other. A slightly higher result will appear when you try to correlate the numbers that were drawn in a given draw with the other numbers drawn in the same draw.

# In[ ]:


# define relationship pairs to check
relations = {
    'sun_n1': ['dist_sun_au', 'n1'],
    'moon_n1': ['dist_moon_au', 'n1'],
    'mars_n1': ['dist_mars_au', 'n1'],
    'sun_n1r': ['dist_sun_au', 'n1r'],
    'moon_n1r': ['dist_moon_au', 'n1r'],
    'day_n1': ['day', 'n1'],
    'month_n1': ['month', 'n1'],
    'year_n1': ['year', 'n1'],
    'dweek_n1': ['day_of_week', 'n1'],
    'dweek_n1r': ['day_of_week', 'n1r'],
    'dyear_n1': ['day_of_year', 'n1'],
    'dyear_n1r': ['day_of_year', 'n1r'],
    'n1_n2': ['n1', 'n2'],
    'n2_n3': ['n2', 'n3'],
    'n3_n4': ['n3', 'n4'],
    'n4_n5': ['n4', 'n5'],
    'n5_n6': ['n5', 'n6']
}

# calculate linear regression relationship between fields
print("[R] - linear regression relationship:\n")
rX = []
rY = []
for name, item in relations.items():
    x1 = item[0]
    x2 = item[1]
    if x1 in df and x2 in df:
        slope, intercept, r, p, std_err = linregress(df[x1], df[x2])        
        rX.append(name)
        rY.append(r)
        print(x1+' > '+x2+': ' + str(r))
        
rY, rX = zip(*sorted(zip(rY, rX)))
    
# display relationships on plot
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Relationship')
ax.set_ylabel('Pair')
ax.set_xlabel('R value')
ax.barh(rX, rY)
plt.show()


# **6. Distributions of the frequency of occurrence of numbers in particular positions.**
# 
# The following cell displays the frequency distribution of the given numbers by their position.

# In[ ]:


print('Distributions of the frequency of occurrence of numbers in particular positions:')

num_of_numbers = cfg['n_count']
cols = 3
rows = math.ceil(num_of_numbers/cols)
row = 0
col = 0
f, ax = plt.subplots(rows, cols, figsize=(25, 15))
for i in range(1, num_of_numbers+1):
    idx = 'n' + str(i)
    data = df[idx].to_numpy()
    sns.histplot(data, kde=True, ax=ax[row][col])
    ax[row][col].set_title('Position: '+idx)
    ax[row][col].set_xlabel('Number')
    ax[row][col].set_ylabel('Count')
    ax[row][col].axvline(x=data.mean(), color='red')
    if col >= (cols - 1):
        col = 0
        row+=1
    else:
        col+= 1
        
plt.show()


# **7. Display a dataset.**
# 
# The cell below displays the extended data set prepared in the previous steps.

# In[ ]:


# display dataset
print("[DATASET]\n")
print(df.to_string())

