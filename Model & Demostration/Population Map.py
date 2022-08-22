import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly as plt
import plotly.express as px
from collections import defaultdict

st.set_page_config(layout="wide")
st.title('COVID Misleading Informaiton Detection 🔥')
#st.header("General Map 🗺")
st.sidebar.markdown("General Map 🗺")

# LOAD DATA ONCE
@st.cache(allow_output_mutation=True)
def read_tweet_dataset(nrows):
    headers = ['text','is_misinfo','state','abbr','latitude','longitude']
    data = pd.read_csv(
        "lat_long_complete.csv",
        header=None,
        nrows=nrows,
        names=headers,
        skiprows=1,
        usecols=[19,36,38,37,39,40]
    )
    return data
data_load_state = st.text('Loading data...')
data = read_tweet_dataset(20000)
data_load_state.text('Done! (using st.chane)')
def read_original_tweet(nrows):
    headers=['text']

# CREATE PYDECK MAP
st.subheader('US Map of COVID Misleading Tweets')
# use pandas to caculate additional data
data['data_radius'] = data['is_misinfo'].groupby(data['abbr']).transform('count')
data['mis_num'] = data['is_misinfo'].groupby(data['abbr']).transform('sum')
data['mis_per'] = round((data['mis_num'] / data['data_radius'])*100,2)
data['non_mis_num'] = data['data_radius'] - data['mis_num'] 
data['non_mis_per'] = 100 - data['mis_per'] 

layers=[
    pdk.Layer(
        'ScatterplotLayer',
        data=data,
        pickable=True,
        opacity = 0.000001,
        stroked=False,
        filled=True,
        radius_scale=200,
        radius_min_pixels=8,
        radius_max_pixels=28,
        line_width_min_pixels=1,
        get_position='[longitude, latitude]',
        get_radius='data_radius',
        get_fill_color=[255,255,255],
        auto_hightlight=True,
    ),
]

# RENDER MAP
viewstate=pdk.ViewState(latitude=37.09,longitude=-95.71,zoom=2)
r = pdk.Deck(layers=layers, initial_view_state=viewstate, map_style='default mapbox template or your own customized template', tooltip={
    'text':'State: {state}\n Num of Tweets: {data_radius}\n Misleading Tweets Detected by Model: {mis_num}({mis_per})%\n Non-missleading Tweets Detected by Model: {non_mis_num}({non_mis_per}%)\n'})
    # TO-DO: misleading num,(percentage). non-misleading num (precentage)
r

abbrs = ['AL','AK','AZ', 'AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
sub_keys = ['mis_num','non_mis_num']
pie_chart_dict = defaultdict(dict)
for abbr in abbrs:
    for sub_key in sub_keys:
        pie_chart_dict[abbr][sub_key] = data.loc[data['state']==abbr, sub_key].iloc[1]

# BAR PLOT
'---'
st.subheader('Number of Misleading Tweets per State by Party')
republican = ['AK', 'ID', 'MT', 'WY', 'ND', 'SD', 'NE', 'IA', 'UT', 'KS', 'MO', 'IN', 'OH', 'KY', 'WV', 'OK', 'TX', 'AR', 'LA', 'TN', 'MS', 'AL', 'FL', 'SC', 'NC']
democratic = [i for i in abbrs if i not in republican]
republican_mis, republican_non = [], []
democratic_mis, democratic_non = [], []
for i in republican:
    #st.bar_chart(pie_chart_dict['AK']['mis_num'])
    republican_mis.append(pie_chart_dict[i]['mis_num'])
    republican_non.append(pie_chart_dict[i]['non_mis_num'])
for i in democratic:
    #st.bar_chart(pie_chart_dict['AK']['mis_num'])
    democratic_mis.append(pie_chart_dict[i]['mis_num'])
    democratic_non.append(pie_chart_dict[i]['non_mis_num'])
# group data together
dem_rep = democratic + republican + democratic + republican
party = ['Democratic']*len(democratic) + ['Republican']*25 + ['Democratic']*len(democratic) + ['Republican']*25
mis_label = ['Mislead']*51 + ['Nonmislead']*51
count = democratic_mis + republican_mis + democratic_non + republican_non
list_of_tuple = list(zip(dem_rep, party, mis_label, count))
df = pd.DataFrame(list_of_tuple, columns=['State','Party','Label', 'Count'])
fig = px.histogram(df, x='State', y='Count', text_auto='2s',color='Party', pattern_shape='Label')
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# ANOTHER BAR PLOT
'---'
st.subheader('Percentage of Misleading Tweets per State by Party')
republican = ['AK', 'ID', 'MT', 'WY', 'ND', 'SD', 'NE', 'IA', 'UT', 'KS', 'MO', 'IN', 'OH', 'KY', 'WV', 'OK', 'TX', 'AR', 'LA', 'TN', 'MS', 'AL', 'FL', 'SC', 'NC']
democratic = [i for i in abbrs if i not in republican]
republican_mis_per = []
democratic_mis_per = []
for i in republican:
    #st.bar_chart(pie_chart_dict['AK']['mis_num'])
    ratio = pie_chart_dict[i]['mis_num']/(pie_chart_dict[i]['mis_num'] + pie_chart_dict[i]['non_mis_num'])
    republican_mis_per.append(ratio)
for i in democratic:
    #st.bar_chart(pie_chart_dict['AK']['mis_num'])
    ratio = pie_chart_dict[i]['mis_num']/(pie_chart_dict[i]['mis_num'] + pie_chart_dict[i]['non_mis_num'])
    democratic_mis_per.append(ratio)
# group data together
dem_rep = democratic + republican
party = ['Democratic']*len(democratic) + ['Republican']*25
count = democratic_mis_per + republican_mis_per 
list_of_tuple = list(zip(dem_rep, party,count ))
df = pd.DataFrame(list_of_tuple, columns=['State','Party','Percentage'])
fig = px.bar(df, x='State', y='Percentage',color='Party')
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)


# PIE CHART
'---'
st.subheader('Details of Tweets per State')
left_column, right_column = st.columns(2)
with left_column:
    pie_state_choose = st.selectbox('Choose a State:',abbrs)
    fig = plt.graph_objects.Figure(
        plt.graph_objects.Pie(
            hole=0.5,
            labels = ['Misleading Tweets', 'Non-misleading Tweets'],
            values = [pie_chart_dict[pie_state_choose]['mis_num'], pie_chart_dict[pie_state_choose]['non_mis_num']],
            hoverinfo = "label+percent",
            textinfo = "value"
    ))
    st.plotly_chart(fig)
with right_column:
    st.write("Random Sample of Misleading Tweets from ",pie_state_choose, 'State:')
    mis_leading_df = data['text'][data['is_misinfo']==1].sample(n=5)
    st.write(mis_leading_df)
    st.write("Random Sample of Nonmisleading Tweets from ",pie_state_choose, 'State:')
    mis_leading_df = data['text'][data['is_misinfo']==0].sample(n=5)
    st.write(mis_leading_df)

#st.write(data)
