#!/usr/bin/env python
# coding: utf-8

# import the library I use to open URLs

# In[1]:


import urllib.request


# specify which URL/web page I am going to be scraping
# and open the url using urllib.request and put the HTML into the page variable

# In[2]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

page = urllib.request.urlopen(url)


# import the BeautifulSoup library so I can parse HTML and XML documents

# In[3]:


from bs4 import BeautifulSoup


# Then I use Beautiful Soup to parse the HTML data we stored in our ‘url’ variable and store it in a new variable called ‘soup’ in the Beautiful Soup format. 
# Jupyter Notebook prefers we specify a parser format so we use the “lxml” library option:
# parse the HTML from our URL into

# In[4]:


soup = BeautifulSoup(page, "lxml")
soup


# use the 'find_all' function to bring back all instances of the 'table' tag in the HTML and store in 'all_tables' variable

# In[5]:


all_tables=soup.find_all("table")
all_tables


# In[6]:


right_table=soup.find('table', class_='wikitable sortable')
right_table


# Now I select the table on the website with 3 columns and I assign the column name to all all of them

# In[8]:


A=[]
B=[]
C=[]

for row in right_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))


# In[9]:


import pandas as pd
df=pd.DataFrame(A,columns=['PostalCode'])
df['Borough']=B
df['Neighborhood']=C
df


# Ignore cells with a borough that is Not assigned

# In[10]:


dd=df[df.Borough != 'Not assigned']
dd


# To check that there are no cells with Not assigned in neighborhood

# In[11]:


dd_nei = dd.loc[dd['Neighborhood'] == 'Not assigned\n']
dd_nei


# Now I want to remove the '\n' at the end of some string in the Neighbourhood column

# In[12]:


dd = dd.replace('\n',' ', regex=True) 
dd


# More than one neighborhood can exist in one postal code area. Rows will be combined into one row with the neighborhoods separated with a comma

# In[13]:



dd1=dd.groupby(['PostalCode'])['Neighborhood'].apply(','.join).reset_index()
dd1


# In[14]:


dd1.shape


# In[ ]:




