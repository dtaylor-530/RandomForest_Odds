from Common import GetDataFrameFromCsvPath
from os import listdir
from os.path import isfile, join,basename
import os

pathremovable= r'/media/declan/40D6-87EE/CombinedLeagues/' 

path=r'/home/declan/Documents/MachineLearning/DataSets/'
path3=r'/home/declan/Documents/MachineLearning/DataSets/FootballDataClean/'

def CreateValidation(league):
    pathex=path+ league +'.csv'
    if not os.path.exists(pathex):
        print ('%s does not exist' % pathex)
        return
    df=GetDataFrameFromCsvPath(pathex)
    df=df.dropna()
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


onlyfiles = [f for f in listdir(pathremovable) if isfile(join(pathremovable, f))]

for league in onlyfiles:
    CreateValidation(os.path.splitext(league)[0])

    
