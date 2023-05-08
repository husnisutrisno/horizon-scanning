from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import pandas as pd
import emojis

import string
import json

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

import dash        
from dash import html, dcc
from dash.dependencies import Output, Input, State

import dash_bootstrap_components as dbc 
import plotly.express as px
import plotly.io as pio
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date, datetime
from ibm_watson import DiscoveryV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException

nltk.data.path.append('./nltk_data/')
sid = SentimentIntensityAnalyzer()

## Download data from IBM COS
def __iter__(self): return 0
endpoint_access = 'https://s3.eu.cloud-object-storage.appdomain.cloud'

client_access = ibm_boto3.client(
    service_name='s3',
    ibm_api_key_id='VMs5gCK0I3klNde_s8RO-A_EEgsT6vnXF_gIZ9z-XtCs',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_access)

def download_data(file):
    body = client_access.get_object(Bucket='jernbanedirektoratetreportsall-donotdelete-pr-ndqtmmoqyxzeyb',Key=file)['Body']
    if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )
    df = pd.read_csv(body)
    return df


### Read data ####
df = download_data("lda_topic_words.csv")           ##### 14 topics with 30 words
df.fillna("", inplace=True)                 
df2 = download_data("All_clean_sentiment_DT.csv")   ##### Sentiment value, hashtag, emoji, dominant topics
df2 = df2.drop(columns=['text', 'clean_text'])
new_words_data = download_data('new_words-2022-05-04.csv') 
new_hashtags = new_words_data['New hashtags']
new_words = new_words_data['New words list']

def update_datasets():
    global df, df2, new_words, new_hashtags
    df = download_data("lda_topic_words.csv")           ##### 14 topics with 30 words
    df.fillna("", inplace=True)                 
    df2 = download_data("All_clean_sentiment_DT.csv")   ##### Sentiment value, hashtag, emoji, dominant topics
    df2 = df2.drop(columns=['text', 'clean_text'])
    new_words_data = download_data('new_words-2022-05-04.csv') 
    new_hashtags = new_words_data['New hashtags']
    new_words = new_words_data['New words list']


## The API key is used to authenticate - it can be found in the Watson Discovery instance 
authenticator = IAMAuthenticator('3RyxMSL-pOAzR-wZy-lzwhoHCNyudRAzeudyd4BzxEt5')
discovery = DiscoveryV1(
    version='2020-09-25',
    authenticator=authenticator
)

# The service URL is found from the Watson Discovery instance
discovery.set_service_url('https://api.eu-de.discovery.watson.cloud.ibm.com/instances/e7652353-13f6-40a5-9868-b3205d8201ca')

pio.templates.default = "ggplot2" # Template for all graphs

#Locate within the query
def locate_in_query_other_time_sentiment(Jsonfile, response):
    #finding relevant information in json
    aggregations_top_level = response['aggregations']
    aggregations_to_iterate_over = aggregations_top_level[0]['aggregations'][0]['results']
    return aggregations_to_iterate_over
    
def locate_in_query_time_sentiment(Jsonfile,response):
    #finding relevant information in json
    aggregations_to_iterate_over = response['aggregations'][0]['results']
    return aggregations_to_iterate_over

#Create df from the json, document+sentment
def query2df(aggregations_to_iterate_over):
    #creating dataframes: the_df = date and matching aricles, sentiment_df = date + sentiment
    sentiment_dict = {}
    date_list = []
    docs_list = []
    for i in range(len(aggregations_to_iterate_over)):
        date = aggregations_to_iterate_over[i]['key_as_string']
        d = datetime.fromisoformat(date[:-1])
        d.strftime('%Y-%m-%d')
        date_list.append(d)
        docs_list.append(aggregations_to_iterate_over[i]['matching_results'])
        pos= 0
        neu = 0
        neg = 0
        for j in range(len(aggregations_to_iterate_over[i]['aggregations'][0]['results'])):
            sentiment = aggregations_to_iterate_over[i]['aggregations'][0]['results'][j]
            if sentiment['key'] == 'positive':
                pos = sentiment['matching_results']
            elif sentiment['key'] == 'neutral':
                neu = sentiment['matching_results']
            elif sentiment['key'] == 'negative':
                neg = sentiment['matching_results']
        sentiment_dict[d] = {'positive': pos, 'negative': neg, 'neutral' : neu}

    #the_df created
    the_dict = {}
    the_dict["date"] = date_list
    the_dict["docs"] = docs_list
    # dict_articles
    the_df =pd.DataFrame.from_dict(the_dict)
    # the_df
    
    #sentimen_df created
    sentiment_df = pd.DataFrame.from_dict(sentiment_dict).T
    sentiment_df = sentiment_df.reset_index()
    sentiment_df = sentiment_df.rename(columns={'index': 'date'})
    sentiment_df
    
    #merged df
    complete_df = pd.merge(the_df, sentiment_df, how='outer', on = 'date')
    return(complete_df)

#The Query
def query(natural_language_query):
    ### Articles over time cathegory + sentiment
    response = discovery.query('system',
                                 'news-en',
                                 natural_language_query=natural_language_query,
                                 aggregation='timeslice(publication_date,1weeks).term(enriched_text.sentiment.document.label,count:3)',
                                 count=0).get_result()

    json_data = json.dumps(response, indent=5)
    return json_data, response

def plot_lines(df, natural_language_query):
    df.plot('date', 'docs', color='#000099', title = 'Documents over time for: "'+natural_language_query+'"')
    df.plot('date', ['positive', 'negative', 'neutral'], color=['#009900','#FF0000','#FFCC00'], title = 'Sentiment over time for: "'+natural_language_query+'"')
    
def plot_pie(df, natural_language_query):
    fig, ax = plt.subplots(figsize=(6, 6))

    x = [df['positive'].sum(),df['negative'].sum(),df['neutral'].sum()]
    labels = ['positive', 'negative', 'neutral']

    patches, texts, pcts = ax.pie(
        x, labels=labels, autopct='%.1f%%',
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'size': 'x-large'},
        startangle=90)
    # For each wedge, set the corresponding text label color to the wedge's
    # face color.
    for i, patch in enumerate(patches):
          texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='white')
    plt.setp(texts, fontweight=600)
    ax.set_title('Sentiment for query: "'+natural_language_query+'"' , fontsize=18)
    plt.tight_layout()    

def get_data(natural_language_query):
    json_data, response = query(natural_language_query)
    result = locate_in_query_time_sentiment(json_data, response)
    df = query2df(result)
    year_df = df[-9:-1]
    plot_lines(year_df, natural_language_query)
    plot_pie(year_df, natural_language_query)
    return year_df

#### Function for emoji for specific topic #####

def emoji_for_topic(df_dict, topic_nr):
    emoji = df_dict[topic_nr]['emoji_list'].str.split(' ')
    emoji_cleaned = []

    removal_list = ['➡️'.encode("utf-8"), '👉'.encode("utf-8"), '👇'.encode("utf-8"),'▶️'.encode("utf-8"), '⤵️'.encode("utf-8"),'⏩'.encode("utf-8"),'🔽'.encode("utf-8"),'⏯️'.encode("utf-8"), '🔜'.encode("utf-8"), '↪️'.encode("utf-8"),'🔝'.encode("utf-8"),'🗓️'.encode("utf-8"),'📅'.encode("utf-8"),'⬇️'.encode("utf-8"),'⬇️'.encode("utf-8"),'↘️'.encode("utf-8")]

    for text in emoji:
        text = [x.strip(string.punctuation) for x in text]
        if text != ['']:
            emoji_cleaned.append(text)

    text_join = [" ".join(text) for text in emoji_cleaned]
    final_emoji = " ".join(text_join)

    # counted emoji
    emoji_list = []
    filtered_emoji = [word for word in final_emoji.split()]
    for i in filtered_emoji:
        if i.encode("utf-8") not in removal_list:
            emoji_list.append(i)
    # print(emoji_list)

    emoji_counted_words = Counter(emoji_list)
    emoji_word_count = {}
    top_emoji_list = []
    for letter, count in emoji_counted_words.most_common(10):
        emoji_word_count[letter] = count
    #print('Counted Emoji:\n')
    for i,j in emoji_word_count.items():
        top_emoji_list.append(i)
    #     print('Emoji: {0}, count: {1}'.format(i,j))
    emoji_str = ' '.join(top_emoji_list)
    return emoji_str

def emoji_topic(topic_nr):
    final_emoji_topic = emoji_for_topic(df_dict, topic_nr)
    fig = EmojiCloud().generate(final_emoji_topic)
    def fix_wc(wc):
        wc.update_layout(margin=dict(l=0, r=0, t=30, b=0), height= 400)
        wc.update_xaxes(visible=False)
        wc.update_yaxes(visible=False)
        return wc
    
    fig_emoji = px.imshow(fig)
    fig_emoji = fix_wc(fig_emoji)
    return fig_emoji  

###### Function for Tab 2 content - Watson Discovery ########

def get_data_query(natural_language_query):
    json_data, response = query(natural_language_query)
    result = locate_in_query_time_sentiment(json_data, response)
    df = query2df(result)
    year_df = df[-9:-1]
    return year_df

def plot_line_docs(df, natural_language_query):
    fig_line = px.line(df, x=df['date'], y=df['docs'], title = '<b>Documents over time for: <br>"'+natural_language_query+'"',
                       color_discrete_sequence=['darkblue'])
    fig_line.update_xaxes(title=None)
    fig_line.update_yaxes(title=None)
    fig_line.update_layout(margin=dict(l=5, r=5, b=90))
    return fig_line


def plot_line_sentiment(df, natural_language_query):
    fig_sent = px.line(df, x='date', y= ['positive', 'negative','neutral'], 
                       title = '<b>Sentiment over time for: <br>"'+natural_language_query+'"',
                       labels={"variable":""},
                       color_discrete_map={
                            "positive": "#248f24",
                            "negative": "#ff9900",
                            "neutral": "#0099cc"})
    fig_sent.update_xaxes(title=None)
    fig_sent.update_yaxes(title=None)
    fig_sent.update_layout(legend=dict(
                        orientation='h',
                        yanchor="bottom",
                        y=-0.35,
                        xanchor="center",
                        x=0.5
                        ),
                        margin=dict(l=5, r=5))
    return fig_sent

    
def plot_pie(df, natural_language_query):
    x = [df['positive'].sum(),df['negative'].sum(),df['neutral'].sum()]
    labels = ['Positive', 'Negative', 'Neutral']    
    fig_pie = px.pie(names=labels, values=x, color=labels, 
                     title='<b>Sentiment overall for: <br>"'+natural_language_query+'"',
                    color_discrete_map={
                            "Positive": "#248f24",
                            "Negative": "#ff9900",
                            "Neutral": "#0099cc"})
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(l=5, r=5), height= 400, showlegend=False)
    return fig_pie


#### Function for pop-up ####
def popup_pie(df2, topic_nr):
    dff = df2.copy() 
    pos = dff['sentiment_label']=='POSITIVE'
    neu = dff['sentiment_label']=='NEUTRAL'
    neg = dff['sentiment_label']=='NEGATIVE'
    topic = dff['dominant_topic']== topic_nr
    
    pos = len(dff[pos & topic])
    neu = len(dff[neu & topic])
    neg = len(dff[neg & topic])
    
    labels = ['Positive', 'Negative', 'Neutral']
    fig_pie = px.pie(names=labels, values=[pos, neu, neg], color=labels, title='<b>Sentiment Overall', 
                    color_discrete_map={
                            "Positive": "#248f24",
                            "Negative": "#ff9900",
                            "Neutral": "#0099cc"})
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(l=5, r=5, t=30, b=20), height= 400, showlegend=False)

    return fig_pie

def popup_sent(df2, topic_nr):
    dff = df2[(df2['dominant_topic']== topic_nr)]
    dff = dff.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
    fig_sent = px.line(dff, title = '<b>Sentiment over time', labels={"sentiment_label":""},
                color_discrete_map={
                "Positive": "#248f24",
                "Negative": "#ff9900",
                "Neutral": "#0099cc"},
                category_orders={"sentiment_label": ["Positive", "Negative", "Neutral"]},)
    fig_sent.update_xaxes(title=None)
    fig_sent.update_yaxes(title=None)
    return fig_sent

class EmojiCloud:
    def __init__(self, font_path='./Symbola.ttf'):
        self.font_path = font_path
        self.word_cloud = self.initialize_wordcloud()
        self.emoji_probability = None
        
    def initialize_wordcloud(self):
        return WordCloud(font_path=self.font_path,
                               width=2000,
                               height=1000,
                               background_color='white',
                               random_state=42,
                               collocations=False)
    
    def generate(self, text):
        emoji_frequencies = Counter(emojis.iter(text))
        total_count = sum(emoji_frequencies.values())
        
        self.emoji_probability = {emoji: count/total_count for emoji, count in emoji_frequencies.items()}
        wc = self.word_cloud.generate_from_frequencies(emoji_frequencies)
        return wc


#df for specific dominant topic
def DT_df(df):
    column = df["dominant_topic"]
    max_value = int(column.max()) 
    df_dict={}
    for topic in range(max_value+1):
        df_topic = df[df.dominant_topic == topic]
        df_dict[topic] = df_topic
    return df_dict
df_dict = DT_df(df2)

# ## Hashtags & Emoji handling

# In[8]:

hashtag = df2['hashtag'].str.lower().str.split(' ')
hashtag_cleaned = []

for text in hashtag:
    text = [x.strip(string.punctuation) for x in text]
    hashtag_cleaned.append(text)

text_join = [" ".join(text) for text in hashtag_cleaned]
final_hashtag = " ".join(text_join)

# In[9]:

emoji = df2['emoji_list'].str.split(' ')
emoji_cleaned = []
removal_list = ['➡️'.encode("utf-8"), '👉'.encode("utf-8"), '👇'.encode("utf-8"),'▶️'.encode("utf-8"), '⤵️'.encode("utf-8"),'⏩'.encode("utf-8"),'🔽'.encode("utf-8"),'⏯️'.encode("utf-8"), '🔜'.encode("utf-8"), '↪️'.encode("utf-8"),'🔝'.encode("utf-8"),'🗓️'.encode("utf-8"),'📅'.encode("utf-8"),'⬇️'.encode("utf-8"),'⬇️'.encode("utf-8"),'↘️'.encode("utf-8")]

for text in emoji:
    text = [x.strip(string.punctuation) for x in text]
    emoji_cleaned.append(text)

text_join = [" ".join(text) for text in emoji_cleaned]
final_emoji = " ".join(text_join)

# counted emoji
emoji_list = []
filtered_emoji = [word for word in final_emoji.split()]
for i in filtered_emoji:
    if i.encode("utf-8") not in removal_list:
        emoji_list.append(i)

emoji_counted_words = Counter(emoji_list)

emoji_word_count = {}
top_emoji_list = []
for letter, count in emoji_counted_words.most_common(50):
    emoji_word_count[letter] = count

for i,j in emoji_word_count.items():
    top_emoji_list.append(i)
    emoji_str = ' '.join(top_emoji_list)
final_emoji = emoji_str

# In[10]:

####### Horizon Scanning function ###########
def logo():
    return dbc.Col([dbc.Card([dbc.CardImg(src="assets/wsp_logo.png")], 
                             outline = False, className='mb-3 mt-3 ml-3 border-0')], width=2)


def wc_horizon():
    return dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='wc_horizon', figure={})
                        ])
                    ]),
                ], width={'size': 6, 'order':1})

def hashtag_horizon():
    return dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='hashtag_horizon', figure={})
                        ])
                    ]),
                ], width={'size': 6, 'order':2})

def scenario_impact():
    return dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='scenario_impact', figure={})
                        ])
                    ]),
                ], width={'size': 12})


# List of interesting & related query to the transportation
related_list = ['work from home', 'remote work', 'home office','garden office', 'bike to work',
'bus to work', 'train to work', 'walk to work', 'commute to work', 'train travelling', 'interrail',
'electric cars', 'electric scooters', 'green transportation', 'clean energy', 'short trips',
'urbanisation reversing', 'de-urbanisation', 'sustainable travel', 'micro mobility',
'traffic jam', 'circular economy', 'train ticket', 'walkable city', 'city trip',
'railway travel', 'train vacation', 'staycation', 'night train', 'scenery train', 'urban mobility']

option = [{"label": col, "value": col} for col in related_list]


# Tab content and dashboard layout
tab_wordcloud = [
                dbc.Row([logo()]),
                
                dbc.Card(html.H4("Topic Wordclouds"),
                         body=True, color="light", outline = False, className='mb-2 mt-2'),

            # Section for Wordclouds
                # first row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud0', figure={}, config={'displayModeBar': False}),                                
                            ]),
                            dbc.Button("More info", id='click0', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]), 
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 0"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,0))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,0))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(0)))
                                        ]),
                            ]),
                        ], id='modal0', size='xl', is_open=False)
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud1', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click1', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 1"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,1))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,1))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(1)))
                                        ]),
                            ]),
                        ], id='modal1', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud2', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click2', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 2"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,2))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,2))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(2)))
                                        ]),
                            ]),
                        ], id='modal2', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud3', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click3', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 3"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,3))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,3))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(3)))
                                        ]),
                            ]),
                        ], id='modal3', size='xl', is_open=False)
                    ], width=3),
                ],align='center',className='mb-2'),

                # second row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud4', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click4', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 4"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,4))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,4))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(4)))
                                        ]),
                            ]),
                        ], id='modal4', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud5', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click5', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 5"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,5))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,5))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(5)))
                                        ]),
                            ]),
                        ], id='modal5', size='xl', is_open=False)
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud6', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click6', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 6"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,6))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,6))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(6)))
                                        ]),
                            ]),
                        ], id='modal6', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud7', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click7', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 7"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,7))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,7))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(7)))
                                        ]),
                            ]),
                        ], id='modal7', size='xl', is_open=False)
                    ], width=3),
                ],className='mb-1'),

                # third row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud8', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click8', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 8"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,8))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,8))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(8)))
                                        ]),
                            ]),
                        ], id='modal8', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud9', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click9', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 9"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,9))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,9))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(9)))
                                        ]),
                            ]),
                        ], id='modal9', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud10', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click10', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 10"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,10))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,10))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(10)))
                                        ]),
                            ]),
                        ], id='modal10', size='xl', is_open=False)
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud11', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click11', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 11"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,11))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,11))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(11)))
                                        ]),
                            ]),
                        ], id='modal11', size='xl', is_open=False)
                    ], width=3),
                ],className='mb-1'),

                # fourth row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud12', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click12', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 12"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,12))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,12))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(12)))
                                        ]),
                            ]),
                        ], id='modal12', size='xl', is_open=False)
                    ], width={'size': 3, 'order':1, 'offset': 3}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='wordcloud13', figure={}, config={'displayModeBar': False}),
                            ]),
                            dbc.Button("More info", id='click13', n_clicks=0, color="dark",
                                       size='sm', outline=True, className="mr-1"),
                        ]),
                        dbc.Modal([   
                            dbc.ModalHeader("Sentiment and Emoji from Topic 13"),
                            dbc.ModalBody([                 
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_sent(df2,13))),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(figure=popup_pie(df2,13))),
                                        dbc.Col(dcc.Graph(figure=emoji_topic(13)))
                                        ]),
                            ]),
                        ], id='modal13', size='xl', is_open=False)
                    ], width={'size': 3, 'order':2}), 
                ],className='mb-1'),

                # Section for Hashtags   
                dbc.Card(html.H4("Hashtags & Emoji Wordclouds"), body=True, color="light", outline = False,
                     className='mb-2 mt-2'),

                # Section for and Emojis
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='hashtag-chart', figure={}, config={'displayModeBar': False}),
                            ])
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='emoji-chart', figure={}, config={'displayModeBar': False}),
                            ])
                        ]),
                    ], width=6),
                ],className='mb-2'),
               ]

## Watson Discovery Tab ##
tab_discovery = [
                dbc.Row([logo()]),
    
                dbc.Card(html.H4("Query Search 1"), body=True, color="light", outline = False, className='mb-2 mt-2'),                    
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dcc.Dropdown(id='query1', options=option, value='work from home', className='mb-2')])
                ],width={'size': 2, 'order':1}),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='no-document1', figure={})
                            ])
                        ]),
                    ], width={'size': 4, 'order':1}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='sentiment1', figure={}, config={'displayModeBar': False}),
                            ])
                        ]),
                    ], width={'size': 5, 'order':2}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='pie-sentiment1', figure={}, config={'displayModeBar': False}, style={'height': 450})
                            ])
                        ]),
                    ], width={'size': 3, 'order':3}),
                ],className='mb-2'),
                

                dbc.Card(html.H4("Query Search 2"), body=True, color="light", outline = False, className='mb-2 mt-2'),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dcc.Dropdown(id='query2', options=option, value='train to work', className='mb-2')])
                    ],width={'size': 2, 'order':1}),       
                ]),
                dbc.Row([ 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='no-document2', figure={})
                            ])
                        ]),
                    ], width={'size': 4, 'order':1}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='sentiment2', figure={}, config={'displayModeBar': False}),
                            ])
                        ]),
                    ], width={'size': 5, 'order':2}), 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='pie-sentiment2', figure={}, config={'displayModeBar': False}, style={'height': 450})
                            ])
                        ]),
                    ], width={'size': 3, 'order':3}),
                ],className='mb-2'),


                dbc.Card(html.H4("Free Text Search"), body=True, color="light", outline = False, className='mb-2 mt-2'),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dcc.Input(id='query3', placeholder='Type the query', type='text', value='work from home'),
                            dbc.Button("Search", id='submit1', n_clicks=0, color="dark", outline=True, className='mb-2')])
                    ], width={'size': 2, 'order':1}),
                ]),
                dbc.Row([    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='no-document3', figure={})
                            ])
                        ]),
                    ], width={'size': 4, 'order':1}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='sentiment3', figure={}, config={'displayModeBar': False}),
                            ])
                        ]),
                    ], width={'size': 5, 'order':2}), 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='pie-sentiment3', figure={}, config={'displayModeBar': False}, style={'height': 450})
                            ])
                        ]),
                    ], width={'size': 3, 'order':3}),
                ],className='mb-2')
           ]
                            

tab_horizon = [
    dbc.Row([logo()]),
    dbc.Card(html.H4("Horizon Scanning"), body=True, color="light", outline = False, className='mb-2 mt-2'),
    
    dbc.Row([
        wc_horizon(),
        hashtag_horizon()
    ], className='mb-2'),

    dbc.Row([
        scenario_impact(),
    ]),
    
    ####### Interval for Live Updates ##########
    dcc.Interval(
            id='interval-component',
            interval=1000*60*60*24*7, # in milliseconds
            n_intervals=0
        )


]

tabs = dbc.Tabs([
    dbc.Tab(tab_wordcloud, label="Wordcloud"),
    dbc.Tab(tab_discovery, label="Watson Discovery"),
    dbc.Tab(tab_horizon, label="Horizon Scanning")],
    id='tabs') 

# In[13]:

# this is here for testing on jupyter notebook
# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.FLATLY], title='JDIR-WSP: Trend & Scenario Detection') 
# app.layout = dbc.Container([tabs],fluid=True)

# Uncomment here for deploying to cloud
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title='WSP: Horizon Scanning') 
server = app.server
app.layout = dbc.Container([tabs],fluid=True)

#### Callback Function ####
@app.callback(
    Output('wordcloud0','figure'),
    Output('wordcloud1','figure'),
    Output('wordcloud2','figure'),
    Output('wordcloud3','figure'),
    Output('wordcloud4','figure'),
    Output('wordcloud5','figure'),
    Output('wordcloud6','figure'),
    Output('wordcloud7','figure'),
    Output('wordcloud8','figure'),
    Output('wordcloud9','figure'),
    Output('wordcloud10','figure'),
    Output('wordcloud11','figure'),
    Output('wordcloud12','figure'),
    Output('wordcloud13','figure'),
    Input('interval-component', 'n_intervals')
)
def update_pie(id):
    dff = df.copy()

    def my_wordcloud(area): 
        words =  dff[area].to_list()
        words1 = [s.replace('[', "")for s in words]
        words2 = [s.replace(']', "")for s in words1]
        words3 = [s.replace('"', "")for s in words2]
        words4 = [s.replace("'", "")for s in words3]
        words = words4
        word_cloud_dict = Counter(words)
        wcloud = WordCloud(
        background_color='black',
        #height=275).generate(' '.join(dff[area].astype(str)))
        ).generate_from_frequencies(word_cloud_dict)
        return wcloud

    fig_wc0 = px.imshow(my_wordcloud('topic0'), title='Topic 0')
    fig_wc1 = px.imshow(my_wordcloud('topic1'), title='Topic 1')
    fig_wc2 = px.imshow(my_wordcloud('topic2'), title='Topic 2')
    fig_wc3 = px.imshow(my_wordcloud('topic3'), title='Topic 3')
    fig_wc4 = px.imshow(my_wordcloud('topic4'), title='Topic 4')
    fig_wc5 = px.imshow(my_wordcloud('topic5'), title='Topic 5')
    fig_wc6 = px.imshow(my_wordcloud('topic6'), title='Topic 6')
    fig_wc7 = px.imshow(my_wordcloud('topic7'), title='Topic 7')
    fig_wc8 = px.imshow(my_wordcloud('topic8'), title='Topic 8')
    fig_wc9 = px.imshow(my_wordcloud('topic9'), title='Topic 9')
    fig_wc10 = px.imshow(my_wordcloud('topic10'), title='Topic 10')
    fig_wc11 = px.imshow(my_wordcloud('topic11'), title='Topic 11')
    fig_wc12 = px.imshow(my_wordcloud('topic12'), title='Topic 12')
    fig_wc13 = px.imshow(my_wordcloud('topic13'), title='Topic 13')

    def fix_wc(wc):
        wc.update_layout(margin=dict(l=0, r=0, t=30, b=0), height= 200)
        wc.update_xaxes(visible=False)
        wc.update_yaxes(visible=False)
        return wc
    figures = [fig_wc0, fig_wc1,fig_wc2, fig_wc3, fig_wc4, fig_wc5, fig_wc6, fig_wc7, 
               fig_wc8, fig_wc9, fig_wc10, fig_wc11, fig_wc12, fig_wc13]
    for i in figures:
        i = fix_wc(i)

    return fig_wc0, fig_wc1, fig_wc2, fig_wc3, fig_wc4, fig_wc5, fig_wc6, fig_wc7, fig_wc8 , fig_wc9, fig_wc10, fig_wc11, fig_wc12, fig_wc13

@app.callback(
    Output('hashtag-chart','figure'),
    Input('interval-component', 'n_intervals'),
)
def update_hashtag(figure):
    final_hashtag_list = Counter(final_hashtag.split(' '))
    wcloud = WordCloud(background_color='white', width=2400, height=1200).generate_from_frequencies(final_hashtag_list)
    
    def fix_wc(wc):
        wc.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        wc.update_xaxes(visible=False)
        wc.update_yaxes(visible=False)
        return wc
    
    fig_hashtag = px.imshow(wcloud, title='<b>Hashtag #')   
    fig_hashtag = fix_wc(fig_hashtag)
    return fig_hashtag  
    
@app.callback(
    Output('emoji-chart','figure'),
    Input('interval-component', 'n_intervals'))
def update_emoji(figure):
    
    text = final_emoji
    emoji_cloud = EmojiCloud()
    wc = emoji_cloud.generate(text)  # this will call the function wordcloud.generate inside the emoji class
    
    def fix_wc(wc):
        wc.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        wc.update_xaxes(visible=False)
        wc.update_yaxes(visible=False)
        return wc
    
    fig_emoji = px.imshow(wc, title='<b>Emoji :)')
    fig_emoji = fix_wc(fig_emoji)
    return fig_emoji    

## Pop-up function
@app.callback(
    Output('modal0','is_open'),
    [Input('click0', 'n_clicks')],
    [State('modal0','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal1','is_open'),
    [Input('click1', 'n_clicks')],
    [State('modal1','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal2','is_open'),
    [Input('click2', 'n_clicks')],
    [State('modal2','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal3','is_open'),
    [Input('click3', 'n_clicks')],
    [State('modal3','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal4','is_open'),
    [Input('click4', 'n_clicks')],
    [State('modal4','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal5','is_open'),
    [Input('click5', 'n_clicks')],
    [State('modal5','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal6','is_open'),
    [Input('click6', 'n_clicks')],
    [State('modal6','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal7','is_open'),
    [Input('click7', 'n_clicks')],
    [State('modal7','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal8','is_open'),
    [Input('click8', 'n_clicks')],
    [State('modal8','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal9','is_open'),
    [Input('click9', 'n_clicks')],
    [State('modal9','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal10','is_open'),
    [Input('click10', 'n_clicks')],
    [State('modal10','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal11','is_open'),
    [Input('click11', 'n_clicks')],
    [State('modal11','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal12','is_open'),
    [Input('click12', 'n_clicks')],
    [State('modal12','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('modal13','is_open'),
    [Input('click13', 'n_clicks')],
    [State('modal13','is_open')])
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

## Watson discovery dropdown function
@app.callback(
    Output('no-document1','figure'),
    Output('sentiment1','figure'),
    Output('pie-sentiment1','figure'),
    [Input('query1', 'value')],)
def query_search1(value):
    df = get_data_query(value)
    fig_line = plot_line_docs(df,value)
    fig_sent = plot_line_sentiment(df,value)
    fig_pie = plot_pie(df, value)
    return fig_line, fig_sent, fig_pie

@app.callback(
    Output('no-document2','figure'),
    Output('sentiment2','figure'),
    Output('pie-sentiment2','figure'),
    [Input('query2', 'value')],)
def query_search2(value):
    df = get_data_query(value)
    fig_line = plot_line_docs(df,value)
    fig_sent = plot_line_sentiment(df,value)
    fig_pie = plot_pie(df, value)
    return fig_line, fig_sent, fig_pie

@app.callback(
    Output('no-document3','figure'),
    Output('sentiment3','figure'),
    Output('pie-sentiment3','figure'),
    [Input('submit1', 'n_clicks')],
    [State('query3','value')])
def query_search3(n_clicks,value):
    df = get_data_query(value)
    fig_line = plot_line_docs(df,value)
    fig_sent = plot_line_sentiment(df,value)
    fig_pie = plot_pie(df, value)
    return fig_line, fig_sent, fig_pie


@app.callback(
    Output('wc_horizon','figure'),
    Input('interval-component', 'n_intervals')
)
def update_wc_horizon(figure):
    update_datasets() #------ this function will download the new dataset according to refresh time (live update setting)
    new_words_list = new_words[0]
    wc = WordCloud(background_color='#D8E6F0', width=2400, height=1200, min_font_size=30, max_font_size=250).generate(new_words_list)
    fig = px.imshow(wc, title='<b>New trending words this week')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

    return fig 

@app.callback(
    Output('hashtag_horizon','figure'),
    Input('interval-component', 'n_intervals')
)
def update_wc_horizon(figure):
    new_hashtags_list = new_hashtags[0]
    wc = WordCloud(background_color='#EFECEA', width=2400, height=1200, min_font_size=30, max_font_size=250).generate(new_hashtags_list)
    fig = px.imshow(wc, title='<b>New hashtags this week')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

    return fig 
    
if __name__ == '__main__':
    app.run_server()

