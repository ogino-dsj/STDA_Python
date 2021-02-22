# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:52:02 2021

@author: Mikito Ogino Japan
Keio University, Dentsu ScienceJam Inc.
"""

import numpy as np
import STDA
X1 = np.random.uniform(-10.0, 10.0, (60, 50, 3000)) #60channelx50dimensionsx3000epochs
X2 = np.random.uniform(20.0, 30.0, (60, 50, 100)) #60channelx50dimensionsx3000epochs
Y1 = np.zeros(3000)
Y2 = np.ones(100)

X = np.concatenate([X1,X2],axis=2)
Y = np.concatenate([Y1,Y2])

clf = STDA.STDA()
itrmax = 200
fea_X = clf.fit(X, Y, itrmax)

probaY = clf.predict_proba(X)

predY = np.where(probaY[:,1] > 0.5, 1, 0)





