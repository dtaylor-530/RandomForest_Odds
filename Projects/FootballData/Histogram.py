# -*- coding: utf-8 -*-
import numpy as np
from Common import get_data
import matplotlib.pyplot as plt
import numpy as np

path4=r'/media/declan/40D6-87EE/CombinedLeaguesCleanedData/'
_league='T1'
_side='H'
_rate=0.3
_margin=0.94
start=1000


y,X,margin=get_data(_league,_side,path4,True)
    
hist, bins = np.histogram(margin, bins=60, weights=y)

plt.hist(hist, normed=True, bins=60)
plt.ylabel('Probability');