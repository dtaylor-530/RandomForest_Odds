import numpy as np
import pandas as pd
import csv as csv
from os import listdir
from os.path import isfile, join,basename
import os
from sklearn.utils import shuffle


path =  r'/media/declan/40D6-87EE/CombinedLeagues/FootballDataClean/' 


def get_data(league,side,path,shuff=False):
    pathex=path + league + side +'.csv'
    ex=not os.path.exists(pathex)
    if not ex:
        print("missing file: " + pathex)
        df=GetDataFrameFromCsvByLeague(league)
        if shuff:
            df = shuffle(df)
        y=df['UP' + side]
        margin=df['margin']
        X=df.drop(['UPH','UPA','UPD','margin'],axis=1)
    else:
        df=CsvStringsFromPanda(pathex)
        if shuff:
            df = shuffle(df)
        unwantedvar="Unnamed: 0"
        if unwantedvar in df.columns.values:
            df=df.drop(unwantedvar,axis=1)
        y=df['UP' + side]
        margin=df['margin']
        X=df.drop(['UP'+side,'margin'],axis=1)
        
    return y,X,margin


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def CsvToNumpy(file,ignore):
    d = np.genfromtxt(file, delimiter=',', names = True)
    d_new = d[[b for b in list(d.dtype.names) if not b in ignore]]
    return d_new


def CsvStringsFromPanda(file):
    return pd.read_csv(file, error_bad_lines=False)

#CsvToNumpy(path,alist)



def FTRToUnitProfit(oddtype,df):
    xx=df['B365'+oddtype]
    return np.where(df.FTR==oddtype, xx.astype(float), 0)-1
   

def FromCsv(league):
    with open(path+league + '.csv', 'r') as f1:    #Open File as read by 'r'
        reader = csv.reader(f1)
        headers = next(reader)
        column = {h:[] for h in headers}
        for row in reader:
            if len(row)!=len(headers):
                print("row %d not same length as header row  %d" % (len(row),len(headers)))
                if(len(row)<len(headers)):
                    diff=len(headers)-len(row)
                    print("adding %d empty elements" % diff)
                    for i in range(0,diff):
                        row.append("")
            for h, v in zip(headers, row):
                column[h].append(v)
        return pd.DataFrame.from_dict(column)



def GetDataFrameFromCsvByLeague(league):
    #columns = “age sex bmi map tc ldl hdl tch ltg glu”.split() #
    #df=CsvStringsFromPanda(path+league+'.csv')
    df=FromCsv(league)
    return GetDataFrame(df)

def GetDataFrameFromCsvPath(path):
    #columns = “age sex bmi map tc ldl hdl tch ltg glu”.split() #
    df=CsvStringsFromPanda(path)
    return GetDataFrame(df)

def GetDataFrame(df):
    #columns = “age sex bmi map tc ldl hdl tch ltg glu”.split() #
    df=df[df['Date']!=""]
    #df['WeekDay'] = pd.to_datetime(df['Date'],errors='coerce').dt.weekday
    
    df['B365H']=pd.to_numeric(df['B365H'])
    df['B365D']=pd.to_numeric(df['B365D'])
    df['B365A']=pd.to_numeric(df['B365A'])
    df['UPH']=FTRToUnitProfit('H',df)
    df['UPD']=FTRToUnitProfit('D',df)
    df['UPA']=FTRToUnitProfit('A',df)
    df['margin']=1/(1/df.B365H.astype(float)+1/df.B365D.astype(float)+1/df.B365A.astype(float))
    df['Date']=pd.to_datetime(df['Date'],errors='coerce').dt.weekday
    xc=df[['Date','B365H','B365D','B365A','margin','UPH','UPD','UPA']]
    clean_dataset(xc)
    datatypesdate=pd.to_datetime(xc['Date'],errors='coerce')
    xc['DayofSeason']=(datatypesdate-min(datatypesdate)).dt.days % 365

    dummies=pd.get_dummies(xc['Date'])
    xc=xc.drop('Date',axis=1)
    xc=xc.join(dummies) 
    return xc


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

#utility
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value

# df[['WeekDay',',]
path=r'/home/declan/Documents/MachineLearning/DataSets/' 
yvalid,Xvalid,margin2=get_data('SP1','H',path)