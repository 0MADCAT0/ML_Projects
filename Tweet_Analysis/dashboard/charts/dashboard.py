import streamlit as st
from PIL import Image


# Page setting
st.set_page_config(layout="wide")
st.title('Tweet Analysis for Keyword "TEKNOFEST"')

left_chart,_, right_chart = st.columns([5, 2, 5])

with left_chart:
    leftside = st.radio(
    'Please select a chart for visualization1',
    ('Top 10 Hashtags', 
    'Top 20 Mentions', 
    'Mostliked Tweet Owners',
    "Top 20 Emojis",
    "Number Of Emojis in a Tweet",
    "Top 10 Retweet Owners",
    "Top 20 Tweeter",
    "Word Cloud",
    "Ngram of Words",
    "Sentiment (Bullying)",
    "Sentiment (Classification)",
    "Sentiment (Polarity)",
    )
)

with right_chart:
    rightside = st.selectbox(
    'Please select a chart for visualization2',
    ('Top 10 Hashtags', 
    'Top 20 Mentions', 
    'Mostliked Tweet Owners',
    "Top 20 Emojis",
    "Number Of Emojis in a Tweet",
    "Top 10 Retweet Owners",
    "Top 20 Tweeter",
    "Word Cloud",
    "Ngram of Words",
    "Sentiment (Bullying)",
    "Sentiment (Classification)",
    "Sentiment (Polarity)",
    )
)






a1, a2 = st.columns(2)


# left side charts
if leftside == 'Top 10 Hashtags':
    left = 'hashtag_top_10.png'

if leftside == 'Top 20 Mentions':
    left = 'mention_top_20.png'

if leftside == 'Mostliked Tweet Owners':
    left = 'most_liked_twt_owners.png'

if leftside == 'Top 20 Emojis':
    left = 'emojis_top_20.png'

if leftside == 'Number Of Emojis in a Tweet':
    left = 'emoji_count_in_twt.png'

if leftside == 'Top 10 Retweet Owners':
    left = 'retwt_owners_top_10.png'

if leftside == 'Top 20 Tweeter':
    left = 'twter_top_20.png'

if leftside == 'Word Cloud':
    left = 'word_cloud.png'

if leftside == 'Ngram of Words':
    left = 'bigram _chart.png'

if leftside == 'Sentiment (Bullying)':
    left = 'bullying_of_twt.png'

if leftside == 'Sentiment (Classification)':
    left = 'twt_class.png'

if leftside == 'Sentiment (Polarity)':
    left = 'polarity_of_twt.png'





# rigth side charts
if rightside == 'Top 10 Hashtags':
    right = 'hashtag_top_10.png'

if rightside == 'Top 20 Mentions':
    right = 'mention_top_20.png'

if rightside == 'Mostliked Tweet Owners':
    right = 'most_liked_twt_owners.png'

if rightside == 'Top 20 Emojis':
    right = 'emojis_top_20.png'

if rightside == 'Number Of Emojis in a Tweet':
    right = 'emoji_count_in_twt.png'

if rightside == 'Top 10 Retweet Owners':
    right = 'retwt_owners_top_10.png'

if rightside == 'Top 20 Tweeter':
    right = 'twter_top_20.png'

if rightside == 'Word Cloud':
    right = 'word_cloud.png'

if rightside == 'Ngram of Words':
    right = 'bigram _chart.png'

if rightside == 'Sentiment (Bullying)':
    right = 'bullying_of_twt.png'

if rightside == 'Sentiment (Classification)':
    right = 'twt_class.png'

if rightside == 'Sentiment (Polarity)':
    right = 'polarity_of_twt.png'



a1.image(Image.open(left))
a2.image(Image.open(right))


