import numpy as np

class ordinalDelCoz:
    '''
    Making interval-valued predictions using the method based on the article 
    "Learning to Predict One or More Ranks inOrdinal Regression Tasks" of J. Del Coz.
    
    The posterior probabilities computed by a base learner is required as a parameter.
    The algorithm optimizes the output to a set of classes basing on these posterior probabilities.
    
    
    '''


    def __init__(self):
        """Initialization
        """
        
        
    def optimize(self,probaset,beta=1):
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
            max_proba = np.zeros((nbclass,2)) 
            
            # we start by finding out the max probabilities when the length of the interval is fixed
            # i is the length of class intervals
            for i in range(1,nbclass+1):
                
                # j is the j-th class 
                for j in range(0,nbclass - i +1) :
                        buffer_proba = sum(probaset[k][j:(j+i)])

                        if buffer_proba > max_proba[i-1][1] :
                            max_proba[i-1][0] = j #starting class
                            max_proba[i-1][1] = buffer_proba # new highest probability of the intervals of length i starting at j
            # when the length of the interval is nbclass, i.e. we take the whole class, and the proba is 1        
#             max_proba[nbclass-1][0] = 0 
#             max_proba[nbclass-1][1] = 1.
            
            # now we minimize the F_beta loss depending on the length i of the class
            # i is the length of class intervals
            min_loss = 1.
            min_i = 1
            for i in range(1,nbclass):
                buffer_min = 1. - (1.+beta*beta)/(beta*beta+i) * max_proba[i-1][1]

                if buffer_min < min_loss :
                    min_loss = buffer_min
                    min_i = i
            
            # we predict every class situated in the interval defined previously with min_i and max_proba[0]
            start_indice = int(max_proba[min_i-1][0]) 
            for indice in range(start_indice,start_indice+min_i):
                ret[k][indice] = 1.
            
            
        return ret
            
            
    