from Common import GetDataFrameFromCsvPath
from os import listdir
from os.path import isfile, join,basename
import os

path= r'/media/declan/40D6-87EE/CombinedLeagues/' 
path3=r'/media/declan/40D6-87EE/CombinedLeaguesCleanedData/' 

def CreateCleanedData(league):
    pathex=path+ league +'.csv'
    df=GetDataFrameFromCsvPath(pathex)

    #uph=df.UPH
    upd=df.UPD
    upa=df.UPA
    df=df.drop('UPD', 1)
    df=df.drop('UPA', 1)

    pathH =  path3+ league + 'H.csv'
    pathD =  path3+ league + 'D.csv'
    pathA =  path3+ league + 'A.csv'

    df.to_csv(pathH, sep=',',index=False )
    df=df.drop('UPH', 1)
    df = df.assign(UPD=upd)
    df.to_csv(pathD, sep=',',index=False )
    df=df.drop('UPD', 1)
    df = df.assign(UPA=upa)
    df.to_csv(pathA, sep=',',index=False )
    print("finished " + league)


onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for league in onlyfiles:
    CreateCleanedData(os.path.splitext(league)[0])

    
