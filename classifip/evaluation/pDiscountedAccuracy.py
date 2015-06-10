'''
Created on 21 avr. 2015

@author: Gen
'''

import math as m
import numpy as np

class pDiscountedAccuracy:
    '''
    Calculate the p-discounted accuracy of a set of predictions.
    
    Parameters:
        - the (ndarray) of predictions classes with for format :
                        class1    class2    class3 
                           --------------------
                    row1 | 0        1        1
                    row2 | 0        1        0
                    
            ('0', non predicted class, '1' predicted class)
        
        - the set of truth (ndarray of the POSITION of the true class), 
        for example [2, 1] means "class3 is the true one in row 1, class2 is the
        true one in row2".
            
        - the cost matrix (cf. classifip.dataset.genCosts)
        
    '''


    def __init__(self, trueClass,predictions,costMatrix):
        '''
        We convert the cost matrix to a normalized reward matrix, so we can compute
        the accuracy later
        '''
        self.trueClasses = trueClass
        self.predictions = predictions
        #self.costMatrix = costMatrix/float(np.max(costMatrix))
        self.costMatrix = costMatrix
        
    
    def computeRow(self,r,truth,prediction,aversion):
        '''
        Compute the p-discounted accuracy of a row of the set
        '''
        
        cost = 0
        
        if prediction[truth] == 0 and aversion is True: 
        #truth not in the prediction and we want mistake aversion
            p = 1 + r
        else:
            p = 1 - r
        
        if p == 0:
            cost = 1
            nbPrediction = 0
            for ind,val in enumerate(prediction):
                if val <> 0:                
                    cost = cost * self.costMatrix[ind,truth]
                    nbPrediction += 1
            return m.pow(cost,1./nbPrediction)
        elif p > 0 :
            for ind,val in enumerate(prediction):
                if val <> 0:                
                    cost += m.pow(self.costMatrix[ind,truth],p)
            return m.pow(cost/(sum(prediction)), 1./p)
        else :
            raise Exception('negative values of p are not handled')
    
    def compute(self,r=0,aversion=False):
        '''
        compute the p-discounted accuracy score with r specified as parameter
        '''
        score = 0
        count = 0

        for ind,val in enumerate(self.trueClasses):
            score += self.computeRow(r, truth=val, prediction=self.predictions[ind],aversion=aversion)
            count += 1
            
        return score/count