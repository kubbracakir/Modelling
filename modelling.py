#Data Loading
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Modelling
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

#Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score

import os

df = pd.read_csv('/Users/mac/Desktop/Python/src/datascience/Modelling/Monkey_Pox_Cases_Worldwide.csv')

df.head()
df.info()
df.isnull().sum()


def top10plots(col=None):
    #Sorting the Dataset
    df_sorted = df.sort_values(by=col,ascending=False).reset_index()
    #Getting the Top10
    top10 = df_sorted[:10]
    # Plotting the Top10
    label_text = ' '.join(col.split('_'))
    labeldict = {'size':'15','weight':'3'}
    titledict = {'size':'20','weight':'3'}
    fig = px.bar(x='Country',
                 y=col,
                 data_frame=top10,
                 labels=['Country',label_text],
                 color=col,
                 color_continuous_scale='electric',
                 text_auto=True,
                 title=f'Top 10 Countries based on {label_text}')
    fig.show()

top10plots(col='Confirmed_Cases')
top10plots(col='Suspected_Cases')
top10plots(col='Hospitalized')
top10plots(col='Travel_History_Yes')
top10plots(col='Travel_History_No')

df[df['Country']=='Portugal']


def world_map(col=None,title=None):
    
    fig = px.choropleth(df,
                  locations='Country',
                  locationmode='country names',
                  hover_name='Country',
                  color=col,
                  color_continuous_scale='electric')

    fig.update_layout(title_text=title)
    fig.show()

world_map(col='Confirmed_Cases',title='Confirmed MonkeyPox Cases Across The Globe')
world_map(col='Suspected_Cases',title='Suspected MonkeyPox Cases Across The Globe')
