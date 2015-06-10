'''
Created on 4 juin 2015

@author: aitech
'''

import numpy as np

class genFscore:
    '''
    Compute the generalized F-score 
        
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
        self.maxcost = float(np.max(costMatrix))
        
    def computeRow(self,beta,truth,prediction):
        '''
        Compute the p-discounted accuracy of a row of the set
        '''
        
        cost_denom = 0
        cost_num = 0
        
        if sum(prediction) == 1.:
            return self.costMatrix[np.where(prediction==1.),truth]
       
        for ind,val in enumerate(prediction):
            cost_denom += self.maxcost - self.costMatrix[ind,truth]
            if val <> 0:
                cost_num += self.maxcost - self.costMatrix[ind,truth]
                         
        return self.maxcost - ((1.+beta**2)*cost_num)/(sum(prediction)*self.maxcost + beta**2*cost_denom)
    
    def compute(self,beta=1.):
        '''
        compute the p-discounted accuracy score with p specified as parameter
        '''
        score = 0
        count = 0
        
        
        for ind,val in enumerate(self.trueClasses):
            score += self.computeRow(beta, truth=val, prediction=self.predictions[ind])
            count += 1
        
        return score/count    