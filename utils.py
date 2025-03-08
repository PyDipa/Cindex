#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A library for the C-index calculation
by Arianna Di Paola
arianna.dipaola@ibe.cnr.it
"""


import numpy as np
from matplotlib import pyplot as plt


#---------------------------------------------------
# Define colors for the boxplot
colors = ['tab:blue', 'orange', 'darkgrey']
# Dictionary for storing data and metadata using key-value pairs
marker = dict(marker='.', markersize=6)


def monotonic(x):
    """
    Identifies the components in a monotonically increasing sequence

    Parameters
    ----------
    x : numpy array
        A one-dimensional array of numeric values.

    Returns
    -------
    components : list
        A list of indices corresponding to elements that maintain
        the monotonic increase.
    """
    components = [0]
    for i, val in enumerate(x[1:], start=1):
        if val >= x[i - 1]:
            components.append(i)
        else:
            break
    return components



def Cindex(I,feature_names,pos_idx,neg_idx,plot,verbose):
    """
    The Cindex calculates a composite index from a set of independent variables.
    This index aims to maximize the Euclidean distance between two subsets of data identified by indicator (or conditional) variables

    
        Parameters
        ----------
        I : numpy array (floats) of shape (N x M) where N are the number of observations and M is the number of the input variables;
            Input variables
            
        feature_names : numpy array (strings) of shape (M,) encoding the name of the variables; i.e., variables' short name
        
        pos_idx : numpy array (integers) of shape (p,);
            the indices of I (row-wise) under the negative condition
        
        neg_idx : numpy array (integers) of shape (n,);
                the indices of I (row-wise) under the negative  condition
         
        plot : boolean (True or False)
            set plot=True for producing the figures
        
        verbose : boolean (True or False)
            set verbose=True for printing the results

        Returns
        -------
        D : list of size M; each element is a numpy array of floats of shape (M,)
            D stores the cumulated Fisher distance obtained by the sequential addition of variables to the composite index;
        
        Vars : list of size M; each element is a numpy array of strings of shape (M,)
            Var stores the variables' label linked to the DD outputs
        
        Direction : as for Var;
            Direction specify the stress direction, and whether each variable has been subtracted or summed by the algorithm:
                '+' == positive stress direction; summed if the first variables has the same stress direction (consistency), otherwise it is subtracted
                '-' == negative stress direction; summed if consistent with the first variables, otherwise subtracted
        
        Components : list of size M; each element is a list of integers of variable length
            Components stores the  componentes that contribute to a monotonic increase in the Fisher distance for each composite index,
            where each composite index is obtained for each starting variable
    """
    # allocating the outputs:

    D = []
    Vars = []
    Direction =[]
    Components =[]
    cnt =0

    # The results depend on the initial variable selection;
    # The Cindex function operates using a nested loop structure:
    #      - The outer `for` loop iterates through each variable as the "initial variable".
    #      - The inner `while` loop sequentially tests all remaining variables, selecting the one
    #        that maximizes the Fisher Distance, and updating the composite index accordingly.
    #        (Note: at the beginning, the composite index equals the initial variable).
    #      - The innermost `for` loop computes distance metrics for both adding and subtracting
    #        each remaining variable, allowing the `while` loop to select the optimal choice at each iteration.

    for i in range(np.shape(I)[1]):
        # select a first variable from I (x)
        x = I[:,i]
        
        # make a copy of I and delete from it the starting variable (>> Inew)
        Inew = np.array(I,copy=True)
        Inew= np.delete(Inew,i,axis=1)
        # make a copy of "features" yet without the name of the selected variable (featnew)
        featnew = np.array(feature_names,copy = True)
        featnew = np.delete(featnew,i)
        # start to allocate the name of the selected variable in the "var" list

        # results from the outer loop are marked by "_i"
        var_i =[]
        var_i.append(feature_names[i])
        # estimate (and put in proper list D) the distance for the selected variable under conditional flag
        # Note: the distance is computed as a ratio betwen a numerator (num) and denominator (den)
        num =(  np.mean(x[ pos_idx ])    - np.mean(x[ neg_idx]) )**2
        den =   np.std (x[ pos_idx ])**2 + np.std (x[ neg_idx]  )**2
        D_i = []
        D_i.append(num/den)
        

        operator_i=[]   # operator: "+" when the variable is summed ; "-" if subtracted
        operator_i.append('+') # the first variables holds its original sign
        # stress direction and operator: specular information that are always computed for check
        # the stress direction of the first variable determines the order of the boxplot diagram:
        # blue box on the right-side of the panel if the first variable has a positive stress direction; (+)
        # blue box on the left-side of the panel  if the first variable has a negative stress direction (-)
        direction_i = []  # stress direction
        if np.mean(x[neg_idx ]) > np.mean(x[pos_idx]):
            direction_i.append('+')
        else:
            direction_i.append('-')
           
        if verbose==True:
            print("i-loop = %s: initial variable = %s, direction = %s" %(str(i),feature_names[i],str(direction_i[0]))  )
    
        #---------------------------------------------------------------------------
        # INNER loop: Now let's see which variables increase the Fisher Distance.
        while True: # an infinite loop that stops thanks to a 'break' condition.
            # the loop goes on until each variable is tested and selected
            # both addition(+) and subtraction(-) are tested. The best choice (i.e., '+' or '-') is stored in operatore_ii
            operator_ii=[]
            # results from the inner loop are marked by "_ii"
            D_ii = []
            direction_ii=[]
            # at each "ii" loop the script select a  variable among the remaining ones  and test its addition/subtraction
            for ii in range(np.shape(Inew)[1]):
                # select a variable among the remaining ones
                x2 = Inew[:,ii]
                # D1 is the result from addition
                vec1 = x + x2 # added
                num1 =(  np.mean(vec1[pos_idx])    - np.mean(vec1[neg_idx]))**2
                den1 =   np.std (vec1[pos_idx])**2 + np.std (vec1[neg_idx])**2
                D1 = np.round(num1/den1,4)
                # D2 is the result from subtraction
                vec2 = x - x2 # subtracted
                num2 =(np.mean(vec2[pos_idx])    - np.mean(vec2[neg_idx]))**2
                den2 = np.std (vec2[pos_idx])**2 + np.std (vec2[neg_idx])**2
                D2 = np.round(num2/den2,4)
                
                # Now let's see whether D1>D2 or vice versa. The largest result is selected and
                # stored in "D_i". The best operator is stored in "operator_ii"
                if D1>D2:
                    D_ii.append(D1)
                    operator_ii.append('+')
                    # the direction is determined by the coherence with the first variable
                    if direction_i[0]=='+':
                        direction_ii.append('+')
                    else:
                        direction_ii.append('-')
                elif D2>D1:
                    D_ii.append(D2)
                    operator_ii.append('-')
                    # the direction is determined by the coherence with the first variable
                    if direction_i[0]=='-':
                        direction_ii.append('+')
                    else:
                        direction_ii.append('-')
    
                
            D_ii =  np.array(D_ii)
            # at the end of each i-loop the best variable is selected (i.e., the variable that mostly increased the distance).
            # Such variable is stored in "var_i" and the related result in "D_i" before being deleted from Inew
            try: 
                D_i.append(np.max(D_ii))
                var_i.append(featnew[np.argmax(D_ii)])
                operator_i.append(operator_ii[np.argmax(D_ii)])
                direction_i.append(direction_ii[np.argmax(D_ii)])
              
                # x is the calculation  of the composite index as sum of Z-scores
                if operator_i[-1] =='+':
                    x = np.round(x + Inew[:,np.argmax(D_ii)],4)
                elif operator_i[-1] =='-':
                    x = np.round(x - Inew[:,np.argmax(D_ii)],4)


                # removing the selected variable from Inew
                Inew = np.delete(Inew, np.argmax(D_ii), axis=1)
                featnew = np.delete(featnew, np.argmax(D_ii))
            except ValueError: # at some point Inew will be empty
                break
        
        var_i=np.array(var_i)
        D_i = np.round(np.array(D_i),4)
        operator_i = np.array(operator_i)
        Components_i = monotonic(D_i)
        if verbose==True:
            print('selected components for the ii-loop (inner loops) defining the composite index:')
            for j in range(len(Components_i)):
                print('    %s %s %s' %(direction_i[Components_i[j]],var_i[Components_i[j]],np.round(D_i[Components_i[j]],2)))
            print('Distance reached: %s' %(np.round(D_i[Components_i[-1]],2))  )
            # LET'S PLOT THE RESULTS
            print('plotting the result in fig.'+str(i+1))
            print('*********************************')
        
        # PLOTTING 
        
        if plot==True:
            cnt=cnt+1
            fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]},num=cnt)
            # fig.suptitle('Horizontally stacked subplots')
            ax1.plot(np.arange(len(D_i)),D_i,'-o',color='grey')
            ax1.plot(np.arange(0,len(Components_i)),D_i[0:len(Components_i)],'-o',color='black')
            ax1.set_xticks(np.arange(len(D_i)))
            labels = [direction_i[j] + var_i[j] for j in range(len(var_i))]
            ax1.set_xticklabels(labels,rotation=90,fontsize=14)
            ax1.set_ylabel('D',fontsize=14)
            ax1.grid()
        # ------ BOXPLOT----------
       
        #--------------PARTE IN COMUNE ------------------------------
            # COMPONENTS added to the initial one:
            varid = [np.where(feature_names==var_i[j])[0][0] for j in Components_i[1::] ]
            A = np.array(I[:,i],copy=True)# initial variables plus...
            for j in range(len(varid)):
                if operator_i[j+1]=='+':
                    A = np.round(A + I[:,varid[j]],4)
                elif operator_i[j+1]=='-':
                    A = np.round(A - I[:,varid[j]],4)
                # skip the frst sigh of perator as it refers to the initial variable
                # exec( 'A = A '+operator[i+1]+' I[:,varid['+str(i)+']]')
            
            a = [A[neg_idx],A[pos_idx],A] # low,high and all data
            bp = ax2.boxplot(a, patch_artist = True,
                notch ='True', vert = False,flierprops=marker)
        
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            for median in bp['medians']:
                median.set(color ='brown', linewidth = 3)
            ax2.set_yticklabels([' ', ' ',' '])
            ax2.set_xlabel('composite z-score',fontsize=14)
            ax2.set_xlim([-15,15])
            plt.tight_layout()
            plt.show()
            
            
        D.append(np.array(D_i))
        Vars.append(np.array(var_i))
        Direction.append(np.array(direction_i))
        Components.append(monotonic(np.array(D_i)))
       
    return(D,Components,Vars,Direction)



