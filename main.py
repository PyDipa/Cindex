# -*- coding: utf-8 -*-
"""
This script provides the recursive calculation used to identify the best C-index.
See the Concept_note.pdf to read about the component index

@author: Arianna Di Paola 

v1. First release December 2021 (deprecated)

v2. Updated in October 2022
    - graphical bug fixed (blue box == data under neg_flag)
    - variable selection bug fixed 
    - added new features:
        > A new library auxiliary.py has been created; 
        > the recursive calculation for C-index has been put into the function Cindex within auxiliary

v3. Updated in June 2023
    - corrected grammatical errors in comments
    - improved comments explaining some command lines
    - renamed some variables to improve the readability and comprehensibility of the code
    
NOTE: - the script was build on Spyder Environment.
      - running the script from terminal each figure (and the script as well) will be blocked until the figure have been closed;

# -------------------------------------------------------
Python 3.7.7
    requirements:
    - numpy 1.19.5    
    - matplotlib 3.3.4
# -----------------------------------------------------

MANDATORY: 
Input features must be expressed as Z-score (mean=0, one standard deviation =1).
In the example inputs data (text data in CSV format) are already standardized in Z-scores. 

        
"""

import numpy as np
from matplotlib import pyplot as plt
import os
# PUT HERE THE PATH OF THE DIRECTORY WITH DATA 

try:
    path = 'put_here_the_path_direcotry'
    os.chdir(path)
except FileNotFoundError:
    print('please insert the directory path of the file "main.py" on your workstation:')
    path = input()
    os.chdir(path)


import sys
sys.path.append(path) #adds (in queque) the "path" to the ENV. VARIABLE "PATH"
from auxiliary import*

#------------------------------------------------------------------------------
# Load the test data: 
# Input features must be expressed as Z-score (mean=0, one standard deviation =1).
# In the example inputs data (text data in CSV format) are already standardized in Z-scores. 

data = np.genfromtxt('Variables.csv', delimiter=',',dtype=str)
I = np.array(data[1::,:].astype(float)) 
feature_names = data[0,:]

cond_var =  np.genfromtxt('Conditional_variable.csv', delimiter=',' )  
pos_idx =  np.where(cond_var>1)[0]
neg_idx =  np.where(cond_var<-1)[0]

print('input data have %s features with %s observations each'%(np.shape(I)[1],np.shape(I)[0]))
print('exceptional high Z-scores are %s '%(len(pos_idx)))
print('exceptional low Z-scores are %s '%(len(neg_idx)))

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




