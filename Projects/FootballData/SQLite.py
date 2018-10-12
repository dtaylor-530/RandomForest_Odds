import numpy as np
import pandas as pd
import sqlite3
import Common
from pandas.io.sql import read_sql

def GetBetFrontDataFromSqlite():
    #if 'data' in locals():
    #    return data
    path='/media/declan/40D6-87EE/database.sqlite'
    conn=sqlite3.connect(path)
    data=read_sql('select * from BetFront',conn)
    return data

def GetFootballDataFromSqlite():
    #if 'data' in locals():
    #    return data
    path='/media/declan/40D6-87EE/database.sqlite'
    conn=sqlite3.connect(path)
    data=read_sql('select * from Football_Data',conn)
    return data

def TidyBetFrontData(data):
    #df=pd.DataFrame()
    date=pd.to_datetime(data['DATETIME'],errors='coerce').dt.weekday
    df=pd.get_dummies( date)
   # df.join(dummies) 
    #df.drop('date',axis=1)
    df['B365H']=data['HOME_CLOSING']
    df['B365A']=data['AWAY_CLOSING']
    df['B365D']=data['DRAW_CLOSING']
    y=np.where(data.FTG1>data.FTG2, df.B365H, 0)-1
    #df['UPD']=np.where(data.FTG1==data.FTG2, df.B365D, 0)-1
    #df['UPA']=np.where(data.FTG1<data.FTG2, df.B365A, 0)-1
    margin=1/(1/df.B365H+1/df.B365D+1/df.B365A)
    return y,df,margin


def TidyFootball_Data(data):
    #df=pd.DataFrame()
    date=pd.to_datetime(data['Date'],errors='coerce').dt.weekday
    df=pd.get_dummies( date)
    div=pd.get_dummies(data['Div'])
    df=df.join(div)
    datatypesdate=pd.to_datetime(data['Date'],errors='coerce')
    df['DayofSeason']=(datatypesdate-min(datatypesdate)).dt.days % 365
    #df['Month']=datatypesdate.month
    # df.join(dummies) 
    #df.drop('date',axis=1)
    df['B365H']=data['B365H'].astype(float)
    df['B365A']=data['B365A'].astype(float)
    df['B365D']=data['B365D'].astype(float)
    df['oug']=np.where(data.FTHG<data.FTAG, df.B365H, 0)-1
    Common.clean_dataset(df)
    
    y=df['oug']
    df=df.drop('oug',axis=1)
    #df['UPD']=np.where(data.FTG1==data.FTG2, df.B365D, 0)-1
    #df['UPA']=np.where(data.FTG1<data.FTG2, df.B365A, 0)-1

    margin=1/(1/df.B365H+1/df.B365D+1/df.B365A)
    return y,df,margin

#dummy=TidyBetFrontData(GetBetFrontDataFromSqlite())
dummy=TidyFootball_Data(GetFootballDataFromSqlite())

