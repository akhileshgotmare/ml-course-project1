# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import math

def build_poly(tX, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    poly = np.zeros((tX.shape[0],1+(tX.shape[1]-1)*degree))
    poly[:,0] = np.ones((tX.shape[0],))
    for p in np.arange(1,degree+1):
        poly[:,1+(p-1)*(tX.shape[1]-1):1+p*(tX.shape[1]-1)] = tX[:,1:tX.shape[1]]**p
    
    return poly


#def build_poly(x, degree):
    
#    N = x.shape[0]
#    d = x.shape[1]-1 # no of features (d) is no of cols - 1 (since we have the cols of 1's for the bias                        #  term)
    
#    phi_x = np.zeros(( N, (d*degree)+1 ))
#    
#    for i in range(N):
#        for j in range( (d*degree)+1 ):
#            if j<=d:
#                phi_x[i,j]=x[i,j]
#                
#            else:
##                block=math.ceil(j/19)
 #               j_b = j - (block-1)*d
#               phi_x[i,j] = pow(x[i,j_b], block)
            
            
            
#    return phi_x
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #raise NotImplementedError