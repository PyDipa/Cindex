# -*- coding: utf-8 -*-
"""
This script provides the recursive calculation used to identify the best C-index.
See the Concept_note.pdf to read about the component index

@author: Arianna Di Paola 

MANDATORY: 
Input features must be expressed as Z-score (mean=0, one standard deviation =1).
In the example inputs data (text data in CSV format) are already standardized in Z-scores. 

        
"""
# # Define the directory path containing the data files

import os
path = os.getcwd()

import sys
sys.path.append(path) #adds (in queque) the "path" to the ENV. VARIABLE "PATH"
from utils import*

#------------------------------------------------------------------------------
# IMPORTANT:
# Input features must be standardized as Z-scores (mean=0, one standard deviation=1).
# The example input data (CSV format) are already standardized.


data = np.genfromtxt('Variables.csv', delimiter=',',dtype=str)
I = np.array(data[1::,:].astype(float)) 
feature_names = data[0,:]

cond_var =  np.genfromtxt('Conditional_variable.csv', delimiter=',' )  
pos_idx =  np.where(cond_var>1)[0]
neg_idx =  np.where(cond_var<-1)[0]

print('The input data contain %s features with %s observations each' % (np.shape(I)[1], np.shape(I)[0]))
print('Number of exceptionally high Z-scores: %s' % len(pos_idx))
print('Number of exceptionally low Z-scores: %s' % len(neg_idx))


#---------------------------------------------------
# Call the Cindex function
#  see help(Cindex) for details
D, Components, Vars, Direction = Cindex(I,feature_names,pos_idx,neg_idx,plot=True,verbose=True)

D = np.array(D)
# select the best result
d = []
for i in range(len(feature_names)):
    n = Components[i][-1]
    d.append(D[i][n])

print('the best C-index reached the distance (D) of %s'%(np.max(d)))

#  let's see the components and their stress directions
# ("negative stress direction": cond_var[neg_idx] occur under negative anomalies of the component.
#   The opposite is true in case of "positive stress direction" (i.e., cond_var[neg_idx] occur under positive anomalies of the component)))

best = np.argmax(d)
n = len(Components[best])
F = Vars[best][0:n]
O = Direction[best][0:n]

print('the components of the best C-index are:')
for f,o in zip(F,O):
    print(o+f)

#------------------------------------------------------


