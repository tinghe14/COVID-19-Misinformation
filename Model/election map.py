# Importing libraries
import numpy as np
import pandas as pd

# Loading base dataset
electionData = pd.read_csv('https://raw.githubusercontent.com/browninstitute/pointsunknowndata/main/presidentialElectionData/countypres_2000-2020.csv')
electionData2020 = electionData[electionData['year'] == 2020].copy(deep=True)

# Creating the states dataset
statesData = pd.pivot_table(electionData2020, index='state', values='candidatevotes', columns='candidate', aggfunc=np.sum).reset_index()
statesData = statesData.rename_axis(None, axis=1)
statesData.fillna(0, inplace=True)
statesData['Total'] = statesData['DONALD J TRUMP'] + statesData['JOSEPH R BIDEN JR'] + statesData['JO JORGENSEN'] + statesData['OTHER']
statesData['Winner'] = statesData[['DONALD J TRUMP','JOSEPH R BIDEN JR','JO JORGENSEN','OTHER']].idxmax(axis=1)
statesData['Winner'] = statesData['Winner'].str.title()
statesData['WnrPerc'] = np.where(statesData['Winner']=='Donald J Trump', statesData['DONALD J TRUMP']/statesData['Total'], statesData['JOSEPH R BIDEN JR']/statesData['Total'])

# Creating the counties dataset
electionData2020.dropna(subset=['county_fips'], inplace=True)
electionData2020['county_fips'] = electionData2020['county_fips'].astype('int').astype('str').str.zfill(5)
electionData2020.rename(columns={'county_fips':'GEOID'}, inplace=True)
countiesData = pd.pivot_table(electionData2020, index='GEOID', values='candidatevotes', columns='candidate', aggfunc=np.sum).reset_index()
countiesData = countiesData.rename_axis(None, axis=1)
countiesData.fillna(0, inplace=True)
countiesData['Total'] = countiesData['DONALD J TRUMP'] + countiesData['JOSEPH R BIDEN JR'] + countiesData['JO JORGENSEN'] + countiesData['OTHER']
countiesData['Winner'] = countiesData[['DONALD J TRUMP','JOSEPH R BIDEN JR','JO JORGENSEN','OTHER']].idxmax(axis=1)
countiesData['Winner'] = countiesData['Winner'].str.title()
countiesData['WnrPerc'] = np.where(countiesData['Winner']=='Donald J Trump', countiesData['DONALD J TRUMP']/countiesData['Total'], countiesData['JOSEPH R BIDEN JR']/countiesData['Total'])

# Loading the state and county shapefiles
#!pip install geopandas # Use this if you are working on Google Colab
import geopandas as gpd

counties = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip')
states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_5m.zip')

# Merging the states datasets
statesData['state'] = statesData['state'].str.title()
states['NAME'] = states['NAME'].str.title()
statesData.rename(columns={'state':'NAME'}, inplace=True)
statesElections = states.merge(statesData, on='NAME')
statesElections = statesElections[['NAME','geometry','DONALD J TRUMP','JOSEPH R BIDEN JR','JO JORGENSEN','OTHER','Total','Winner','WnrPerc']].copy(deep=True)
statesElections.rename(columns={'NAME':'State','DONALD J TRUMP':'Trump','JOSEPH R BIDEN JR':'Biden','JO JORGENSEN':'Jorgensen','OTHER':'Other'}, inplace=True)

# Merging the counties datasets
countiesElections = counties.merge(countiesData, on='GEOID')
countiesElections = countiesElections[['NAME','STATE_NAME','geometry','DONALD J TRUMP','JOSEPH R BIDEN JR','JO JORGENSEN','OTHER','Total','Winner','WnrPerc']].copy(deep=True)
countiesElections.rename(columns={'NAME':'County','STATE_NAME':'State','DONALD J TRUMP':'Trump','JOSEPH R BIDEN JR':'Biden','JO JORGENSEN':'Jorgensen','OTHER':'Other'}, inplace=True)

# Creating the county points dataset
countiesPoints = countiesElections.copy(deep=True)
countiesPoints['point'] = countiesPoints.representative_point()
countiesPoints['geometry'] = countiesPoints['point']
countiesPoints.drop(columns=['point'], inplace=True)

# Exporting datasets as GeoJSONs
countiesElections.to_file('countiesElections.geojson', driver='GeoJSON')
statesElections.to_file('statesElections.geojson', driver='GeoJSON')
countiesPoints.to_file('countiesPoints.geojson', driver='GeoJSON')