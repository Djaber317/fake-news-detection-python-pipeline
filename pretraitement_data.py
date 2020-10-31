#This file is dedicated to normalisation of data that we gonna use to
#to make our pipeline model learn from
#The dataset has a shape of (7796×4). The first column identifies the news, 
#the second and third are the title and text, and the fourth column has labels 
#denoting whether the news is REAL or FAKE.

import pandas as pd 
import os
import csv

def pretraitement_data():
    #first we need to read the data
    
    df = pd.read_csv("C:\\Users\\AEK INFO\\Documents\\GitHub\\fake-news-detection-python-pipelines\\news.csv")    
    #group the columns text and title
    df['text'] += df['title']
    #remove columns that we don't need 
    df.drop(df.columns[[0, 1]], axis = 1, inplace=True)
    list_ = df.values.tolist()
    return list_
