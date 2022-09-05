import streamlit as st
from PIL import Image


# Page setting
st.set_page_config(layout="wide")
st.title('Tweet Analysis for Searchterm "TEKNOFEST"')
st.sidebar.markdown("Visualization Selection")

# Add a selectbox to the sidebar:
leftside = st.sidebar.selectbox(
    'Please select a chart for visualization1',
    ('Top 10 Hashtags', 
    'Top 20 Mentions', 
    'Mostliked Tweet Owners',
    )
)

rightside = st.sidebar.selectbox(
    'Please select a chart for visualization2',
    ('Top 10 Hashtags', 
    'Top 20 Mentions', 
    'Mostliked Tweet Owners',
    )
)

a1, a2 = st.columns(2)
if leftside == 'Top 10 Hashtags':
    left = 'charts//hashtag_top_10.png'

if rightside == 'Top 20 Mentions':
    right = 'charts\mention_top_20.png'



a1.image(Image.open(left))
a2.image(Image.open(right))
