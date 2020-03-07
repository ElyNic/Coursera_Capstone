#!/usr/bin/env python
# coding: utf-8

# # $3^{rd}$ part of the submission

# Explore and cluster the neighborhoods in Toronto. You can decide to work with only boroughs that contain the word Toronto and then replicate the same analysis we did to the New York City data. It is up to you.
# 
# Just make sure:
# 
# to add enough Markdown cells to explain what you decided to do and to report any observations you make.
# to generate maps to visualize your neighborhoods and how they cluster together.

# Install folium and geopy.geocoders

# In[ ]:


conda install -c conda-forge geopy
conda install -c conda-forge folium


# In[45]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

print('Libraries imported.')


# To check the number of boroughs and neighborhoods:

# In[103]:


print('The dataframe has {} boroughs and {} neighborhood.'.format(
        len(df1['Borough'].unique()),
        df1.shape[0]
    )
)


# In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent <em>toronto_explorer</em>, as shown below.

# In[104]:


address = 'Toronto, CA'

geolocator = Nominatim(user_agent="to_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ### Create a map of Toronto with neighborhoods superimposed on top.

# In[105]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df1['Latitude'], df1['Longitude'], df1['Borough'], df1['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# Let's slice the original dataframe and create a new dataframe with only boroughs that contain the word Toronto

# In[106]:


etoronto_data = df1[df1['Borough'] == 'East Toronto'].reset_index(drop=True)
etoronto_data.head()


# In[107]:


etoronto_data.shape


# Let's get the geographical coordinates of the Borough maned Toronto.

# In[108]:


address = 'East Toronto, CA'

geolocator = Nominatim(user_agent="to_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of East Toronto are {}, {}.'.format(latitude, longitude))


# Let's visualize Toronto and the neighborhoods in it.

# In[109]:


# create map of Manhattan using latitude and longitude values
map_etoronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(etoronto_data['Latitude'], etoronto_data['Longitude'], etoronto_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_etoronto)  
    
map_etoronto


# I am going to start utilizing the Foursquare API to explore the neighborhoods and segment them.
# Define Foursquare Credentials and Version

# In[98]:


CLIENT_ID = 'KSJYMEYCFWVXEPGRHRPZDJUIZGDEQY4T0DIJWGJX0YRY4VSJ' # your Foursquare ID
CLIENT_SECRET = '3JVANRI2MPWO3QQV3GMWN1CZEBGK0OZAZFRAFSUQYWUCGBSY' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Let's explore the first neighborhood in our dataframe.

# In[110]:


etoronto_data.loc[0, 'Neighborhood']


# Get the neighborhood's latitude and longitude values.

# In[111]:


neighborhood_latitude = etoronto_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = etoronto_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = etoronto_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# let's get the top 100 venues that are in East Toronto within a radius of 500 meters.

# In[112]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[113]:


results = requests.get(url).json()
results


# In[114]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# Now we are ready to clean the json and structure it into a pandas dataframe

# In[115]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[116]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# I want to explore the neighborhood in East Toronto

# In[117]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# Now I run the above function on each neighborhood and create a new dataframe called *etoronto_venues*.

# In[118]:


etoronto_venues = getNearbyVenues(names=etoronto_data['Neighborhood'],
                                   latitudes=etoronto_data['Latitude'],
                                   longitudes=etoronto_data['Longitude']
                                  )


# Let's check the size of the resulting dataframe

# In[120]:


print(etoronto_venues.shape)
etoronto_venues.head()


# Let's check how many venues were returned for each neighborhood

# In[122]:


etoronto_venues.groupby('Neighborhood').count()


# Let's find out how many unique categories can be curated from all the returned venues

# In[123]:


print('There are {} uniques categories.'.format(len(etoronto_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[125]:


# one hot encoding
etoronto_onehot = pd.get_dummies(etoronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
etoronto_onehot['Neighborhood'] = etoronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [etoronto_onehot.columns[-1]] + list(etoronto_onehot.columns[:-1])
etoronto_onehot = etoronto_onehot[fixed_columns]

etoronto_onehot.head(50)


# ####  And let's examine the new dataframe size.

# In[126]:


etoronto_onehot.shape


# ####  Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[127]:


etoronto_grouped = etoronto_onehot.groupby('Neighborhood').mean().reset_index()
etoronto_grouped


# #### Let's confirm the new size

# In[129]:


etoronto_grouped.shape


# #### Let's print each neighborhood along with the top 6 most common venues

# In[130]:


num_top_venues = 6

for hood in etoronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = etoronto_grouped[etoronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a *pandas* dataframe

# First, let's write a function to sort the venues in descending order

# In[133]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood

# In[134]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = etoronto_grouped['Neighborhood']

for ind in np.arange(etoronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(etoronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# #### Cluster Neighborhoods
# Run *k*-means to cluster the neighborhood into 5 clusters.

# In[139]:


# set number of clusters
kclusters = 5

etoronto_grouped_clustering = etoronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(etoronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# ####  Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[142]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'etorontoCluster Labels', kmeans.labels_)

etoronto_merged = etoronto_data

# merge etoronto_grouped with etoronto_data to add latitude/longitude for each neighborhood
etoronto_merged = etoronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

etoronto_merged.head() # check the last columns!


# Finally, let's visualize the resulting clusters

# In[143]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(etoronto_merged['Latitude'], etoronto_merged['Longitude'], etoronto_merged['Neighborhood'], etoronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Now, I can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, I can then assign a name to each cluster.

# In[145]:


etoronto_merged.loc[etoronto_merged['Cluster Labels'] == 0, etoronto_merged.columns[[1] + list(range(5, etoronto_merged.shape[1]))]]


# In[146]:


etoronto_merged.loc[etoronto_merged['Cluster Labels'] == 1, etoronto_merged.columns[[1] + list(range(5, etoronto_merged.shape[1]))]]


# In[150]:


etoronto_merged.loc[etoronto_merged['Cluster Labels'] == 2, etoronto_merged.columns[[1] + list(range(5, etoronto_merged.shape[1]))]]


# In[149]:


etoronto_merged.loc[etoronto_merged['Cluster Labels'] == 3, etoronto_merged.columns[[1] + list(range(5, etoronto_merged.shape[1]))]]


# In[148]:


etoronto_merged.loc[etoronto_merged['Cluster Labels'] == 4, etoronto_merged.columns[[1] + list(range(5, etoronto_merged.shape[1]))]]


# In[ ]:




