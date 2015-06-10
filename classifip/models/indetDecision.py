'''
Created on 4 juin 2015

@author: aitech
'''
import numpy as np
import itertools
import classifip.evaluation.pDiscountedAccuracy as PDA
import classifip.evaluation.genFscore as GFS

class indetDecision:
    '''
    Classifier which minimize the expected cost over all possible predictions:
    determinate and indeterminate. 
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    
    def minimize(self,probaset,c,method='PDA',param=1):
        """
        
        :param probaset: list posterior probabilities of k instances of a classif problem
        :type probaset: list of (list of float)
        
        :param method: method used to compute the accuracy/cost score
        :type method: either 'PDA: p-discounted accuracy' or 'GFS: generalized F-score'
        
        :param param: the parameter required by the method: refer to either 'r' or PDA or 'beta' of 'GFS'
        :type parem: float
        
        :param c: the cost matrix with only determinate predictions
        :type c: numpy.ndarray
        
        :return ret: the prediction matrix
        """
        #Initialize the prediction matrix
        nbclass = len(probaset[0]) # number of class of the problem
        ret = np.zeros((len(probaset),nbclass))
        
        #Initialize the score computation method
        if method == 'PDA':
            scoreMethod = PDA.pDiscountedAccuracy
        elif method == 'GFS':
            scoreMethod = GFS.genFscore
        else : 
            raise Exception('Unknown score computation method')
        
        
        for k in range(0,len(probaset)):
            buff = float('inf') #the current minimum expected cost
            prediction = []
            for L in range(1, nbclass+1):
                for subset in itertools.combinations(range(0,nbclass), L):
                    score = 0 #score of each possible prediction
                    
                    #forming the current prediction
                    pred_buff = [0] * nbclass
                    for sub in subset:
                        pred_buff[sub] = 1
                    for truth in range(0,nbclass):
                        score += scoreMethod(trueClass=[truth],predictions=[pred_buff],costMatrix=c).compute(param)*probaset[k][truth]
                    if buff > score:
                        prediction = pred_buff
                        buff = score
            ret[k] = prediction
            
        return ret
                        
            