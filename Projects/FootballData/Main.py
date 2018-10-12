import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join,basename
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV,  train_test_split
import Common
import SQLite
from sklearn import datasets, linear_model, preprocessing
from matplotlib import pyplot as plt
from sklearn.utils import shuffle



#mask = np.all(np.isnan(df) | np.equal(df, 0) | np.isinf(df), axis=1)
#df[~mask]

def generate_model(X_train, X_test, y_train, y_test):
  
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=100),
                          n_estimators=200, 
                          learning_rate=0.01
                         )
    '''
    model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls')
   
    model = ExtraTreesRegressor(
                          n_estimators=200
                         )
    '''
    model.fit(X_train,y_train)
    print("model score ", model.score(X_test, y_test))
    return model

path2=r'/media/declan/40D6-87EE/CombinedLeagues/'
path=r'/home/declan/Documents/MachineLearning/DataSets/FootballDataClean/'
path4=r'/media/declan/40D6-87EE/CombinedLeaguesCleanedData/'
_league='SP1'
_side='D'
_rate=0.4
#_margin=0.94
start=1000
'''
def fit_linear(margin):
    cumsum=np.cumsum(np.where((output>0) & (X_test.margin>margin), (output*y_test), 0))
    regr = linear_model.LinearRegression()
    XX = np.array(list(range(0,len(cumsum)))).reshape(-1,1)
    YY=cumsum.reshape(-1,1)
    regr.fit(XX,YY)

    print (" percentage profit {}".format(regr.coef_))
'''

def get_profit(target,output,margin,odds,start,minmargin):
    profit=[]
    for h, v,xt,odd in zip(target, output,margin,odds):
        if(v>0) & (xt>minmargin):
            start+=start*_rate*h*v/(odd*odd)
        profit.append(start)
        
    #plt.plot(np.arange(np.shape(profit)[0]),profit)

    return profit


def plot_log(profit,base):
    logbase=np.log(base)
    logprofit=np.log(profit)-logbase
    plt.plot(np.arange(np.shape(logprofit)[0]),logprofit)



def _main(league,side,rate):

    #y,X,margin=Common.get_data(league,side,path4,True)
    #data=SQLite.GetFootballDataFromSqlite()
    y,X,margin=SQLite.TidyFootball_Data(SQLite.GetFootballDataFromSqlite())
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None,shuffle=True)
    print(X_train[:2])
    model = generate_model(X_train, X_test, y_train, y_test )
    
    '''
    profits=list()
    for i in range(10):
    #plot_log(profit,start)
        marg=0.9+(i/100)
        #unitprofit=np.average(np.where(output>0 & (X_test.margin>marg), (output*y_test), 0))
        profit=get_profit(y_train, output,margin,X_train['B365' + side],start,marg)
        #print ("unitprofit: {}".format(unitprofit))
        profits.append((marg,profit[-1]))
    #plt.plot(np.arange(np.shape(y_test)[0]),cumsum)
    bestmargin=max(profits,key=lambda item:item[1])[0]
 
    '''
    output=model.predict(X_train)
    #profits=list()
  
    '''
    for i in range(7):
    #plot_log(profit,start)
        marg=0.9+(i/100)
        #mhead=np.where(margin.head(output.shape[0])>marg,True,False)
        total=list()
        i=0
        for io in margin.head(output.shape[0]):
            if io>marg:
                if output[i]>0: 
                    total.append(output[i]*y_train[i])
            i=i+1
                    
        profit=np.average( total)
        #profit=get_profit(y_test, output,margin[-np.shape(y_test)[0]:],X_test['B365' + side],start,marg)
        #print ("unitprofit: {}".format(unitprofit))
        profits.append((marg,profit))
    '''
        
    output=model.predict(X_test)

    #plt.plot(np.arange(np.shape(y_test)[0]),cumsum)
    #bestmargin=max(profits,key=lambda item:item[1])[0]
    bestprofits=list()
    #bestmargin=[item for item in profits if item[1]>0]
    for i in range(0,10):
        bm=0.9+(i/100)
        profit=get_profit(y_test, output,margin[-np.shape(y_test)[0]:],X_test['B365' + side],start,bm)
        print ("profit test " ,profit[-1], "margin ",bm )
        bestprofits.append(profit)
        #plot_log(profit,start)
        plt.plot(np.arange(np.shape(profit)[0]),profit)
    #if(len(bestmargin)>0):
    #    weighted_avg =np.average([bp[-1]*item[1] for bp,item in zip(bestprofits,bestmargin)])
        #plt.plot(np.arange(np.shape(weighted_avg)[0]),weighted_avg)
    '''
    yvalid,Xvalid,margin2=Common.get_data(league,side,path)
  
    output2=model.predict(Xvalid)
    #print(output2)
    
    #print ("error: {}".format(((y_test.values- output) ** 2).mean()))
    #unitprofit2=np.average(np.where(output2>0 & (margin2>bestmargin), output2*yvalid, 0))
    #print ("unitprofit2: {}".format(unitprofit2))
    

    for i in range(0,7):
        bm=0.9+(i/100)
        profit2=get_profit(yvalid, output2,margin2,Xvalid['B365' + side],start,bm)
        print ("profit valid " ,profit2[-1],"margin " ,bm)
        bestprofits.append(profit2)
        plot_log(profit2,start)

    #print ("profit valid {}".format(profit2[-1]))
    #plt.plot(np.arange(np.shape(profit2)[0]),profit2)

    return profit,profit2
    '''
def main_():
    onlyfiles = [f for f in listdir(path2) if isfile(join(path2, f))]
    for league in onlyfiles:
        for side in ['H','D','A']:
            l=os.path.splitext(league)[0]
            print(l + " " + side)
            print(_main(l,side,_rate))

def main2():
    for i in range(4):
        _main(_league,_side,_rate)
 
def main__():
    onlyfiles = [f for f in listdir(path2) if isfile(join(path2, f))]
    for league in onlyfiles:
        __sum=0
        l=os.path.splitext(league)[0]
        print(l)
        _sum=0
        yvalid,Xvalid,margin2=Common.get_data(l,'H',path)
        for xv,yv in zip(Xvalid.iterrows(),yvalid):
            if xv[0]>6 & xv[0]<8:
                _sum+=yv
        print(_sum) 
        __sum+=_sum
        print(__sum)
#_main(_league,_side,_rate)
main__()