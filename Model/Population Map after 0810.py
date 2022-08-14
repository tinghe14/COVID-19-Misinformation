import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title('COVID Misleading Informaiton Detection')
st.header("Population Map 🗺")
st.sidebar.markdown("Population Map 🗺")

# LOAD DATA ONCE
@st.cache(allow_output_mutation=True)
def read_dataset(nrows):
    headers = ['is_misinfo','state','abbr','latitude','longitude']
    #dtypes = [float, float] #no because of the dtype but because of the empty field
    data = pd.read_csv(
        "lat_long_complete.csv",
        header=None,
        nrows=nrows,
        names=headers,
        skiprows=1,
        usecols=[36,38,37,39,40]
    )
    return data
data_load_state = st.text('Loading data...')
data = read_dataset(20000)
data_load_state.text('Done! (using st.chane)')

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
        opacity = 0.001,
        stroked=False,
        filled=True,
        radius_scale=200,
        radius_min_pixels=8,
        radius_max_pixels=28,
        line_width_min_pixels=1,
        get_position='[longitude, latitude]',
        get_radius='data_radius',
        get_fill_color=[180, 0, 200],
        auto_hightlight=True,
    ),
]

viewstate=pdk.ViewState(latitude=37.09,longitude=-95.71,zoom=2)
# RENDER
r = pdk.Deck(layers=layers, initial_view_state=viewstate, map_style='mapbox://styles/mapbox/dark-v10', tooltip={
    'text':'State: {state}\n Num of Tweets: {data_radius}\n Misleading Tweets Detected by Model: {mis_num}({mis_per})%\n Non-missleading Tweets Detected by Model: {non_mis_num}({non_mis_per}%)\n'})
    # TO-DO: misleading num,(percentage). non-misleading num (precentage)
#r

components.html('''
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />
    <script src='https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.js'></script>
    <link href='https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.css' rel='stylesheet' />
    <link rel="stylesheet" href="styles.css">
</head>

<body>
    <div id='map'></div>
    <script type='text/javascript' src="map.js"></script>
    <div class='map-overlay' id='features'><h2>US population density</h2><div id='pd'><p>Hover over a state!</p></div></div>
    <div class='map-overlay' id='legend'></div>
</body>
''')
st.write(data)