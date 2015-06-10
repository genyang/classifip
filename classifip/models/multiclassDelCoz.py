'''
Created on 11 fevr. 2015

@author: Gen
'''
import numpy as np

class multiclassDelCoz:
    '''
    Making set-valued predictions using the method based on the article 
    "Learning nondeterministic classifiers" of J. Del Coz and A. Bahamonde.
    
    The posterior probabilities computed by a base learner is required as parameter.

    The algorithm optimizes the output to a set of classes basing on these posterior 
    probabilities and the F-beta measure.
    
    
    '''


    def __init__(self):
        """Initialization
        """
        
        
    def optimize(self,probaset,beta=1.):
        """learn the NCC, mainly storing counts of feature/class pairs
        
        :param probaset: list posterior probabilities of k instances of a classif problem
        :type learndataset: list of (list of float)
        
        :param beta: parameter for the F_beta loss used for the loss minimization procedure, by default, the F1 measure is used
        :type beta: int
        
        :return ret: the prediction matrix
        """
        nbclass = len(probaset[0]) # number of class of the problem
        ret = np.zeros((len(probaset),nbclass))

        #Initializing the prediction matrix
        
        for k in range(0,len(probaset)):

            # get the descending ordering of porbabilities
            proba_ordered = np.argsort(-np.array(probaset[k]))
            
            #initialization of optimization
            i=1
            loss = 1.
            loss_buffer = 1. - (1.+beta*beta)/(beta*beta+i) * sum([probaset[k][ind] for ind in proba_ordered[0:i]])
            while (i<nbclass) and (loss_buffer < loss):
                i += 1
                loss = loss_buffer
                loss_buffer = 1. - (1.+beta*beta)/(beta*beta+i) * sum([probaset[k][ind] for ind in proba_ordered[0:i]])
                
                    

            if loss_buffer >= loss:
                for ind in proba_ordered[0:(i-1)]: 
                    ret[k][ind] = 1. 
            else :    
                for ind in range(0,nbclass):
                    ret[k][ind] = 1.        
        return ret
            
            
    