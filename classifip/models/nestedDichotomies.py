'''
Created on mai 2016

@author: Gen Yang
'''
from classifip.representations import binaryTree as bt
import pickle, copy
from classifip.representations.binaryTree import BinaryTree

    
class NestedDichotomies(bt.BinaryTree):
    """ NestedDichotomies is a classifier that implements the nested dichotomy 
    binary decomposition technique [#Fox1997,#Frank2004] allowing for the 
    transformation of a multi-classification problem into a set of binary ones. 
    The specific feature of this implementation is that, we allow for the 
    treatment of interval-valued probabilities. 
     
    :param learner: the base (binary) classifier 
    
    """         
    
    def __init__(self,classifier, label=None,load=None, node=None):
        """
        Initialization of the Nested Dichotomies classifier
        :param classifier: one instance of the base binary classifier to be used.
        The classifier should have a "learn" and a "evaluate" methods available.
        The final output of the classifier should be a list of 
        :class:`~classifip.representations.intervalProbabilities.IntervalProbabilities`
        """
        if classifier is not None:
            self.classifier = classifier
        else :
            raise Exception('Must specify a binary classifier to be used')
        
        if load is not None:
            with open(load + '.pkl', 'rb') as f:
                tree = pickle.load(f)
                self.node = tree.node
                self.left = tree.left
                self.right = tree.right
                self.nbDecision = tree.nbDecision
                self.classifier = tree.classifier
        else :
            if node is None:
                if label is None:
                    raise Exception('No initialization information provided')
                self.node = NestedDichotomies.Node(label=label)
                self.nbDecision = self.node.count()
            else :    
                self.node = node
                self.nbDecision = node.count()
                 
            self.left = None
            self.right = None
            
    
    
    def learnCurrent(self,dataset,**kwargs):
        """
        Learn the underlying binary classification problem associated with the 
        current node of the dichotomy tree, and store the models in the Nested 
        Dichotomy tree.

        (See :func:`~classifip.models.nestedDichotomies.learn` for the detail of parameters)
        """
        
        if self.left.node.isEmpty() or self.right.node.isEmpty() :
            raise Exception("Current node has no left or/and right child node.")
        
        data = dataset.select_class_binary(positive=self.left.node.label, 
                                       negative=self.right.node.label)
        
        # Apply the base binary classifier for the current node of the tree     
        self.classifier.learn(data,**kwargs)

    
    def learn(self,dataset,**kwargs):
        """
        Recursive learning process (see :func:`~classifip.models.nestedDichotomies.learnCurrent`) 
        for the entire dichotomy tree structure and the entire dataset.
        
        :param dataset: learning data
        :type dataset: :class:`~classifip.dataset.ArffFile`
        
        :param **kwargs: the parameters available to the base binary classifier.
        
        .. warning:: no check is performed on the validity of the arguments.
        """
        if (self.left is not None) and (self.right is not None) :
            self.learnCurrent(dataset,**kwargs)
            '''
            we only try to learn the children nodes when there are more than one
            class value / label associated with them.
            '''
            if self.left.node.count() > 1:
                self.left.learn(dataset,**kwargs)
            if self.right.node.count() > 1:    
                self.right.learn(dataset,**kwargs)      
        
    
    def _evalCurrent(self,testdataset,out,**kwargs): 
        """
        Evaluation of the learnt local binary model on the test dataset for the 
        current node of the dichotomy tree.
        
        (See :func:`~classifip.models.nestedDichotomies.evaluate` for the detail of parameters)
        """            
        if self is not None :
            # for a single node
            #self.left.node.proba = None #we reset results of previous evaluations
            #self.right.node.proba = None

            self.node.proba = self.classifier.evaluate(testdataset, **kwargs)
            out.nbDecision = self.nbDecision
            out.node.label = self.node.label 
            out.node.proba = self.node.proba[0][0]
            
            out.left = bt.BinaryTree(self.left.node)
            out.right = bt.BinaryTree(self.right.node)
            #===================================================================
            # if type(result[0][0]) is probadis.ProbaDis:
            #     for ip in result:
            #         self.left.node.proba.append([ip[0].proba[0],ip[0].proba[0]])
            #         self.right.node.proba.append([ip[0].proba[1],ip[0].proba[1]])
            # else:
            #     for ip in result:
            #         self.left.node.proba.append([ip[0].lproba[1,0],ip[0].lproba[0,0]])
            #         self.right.node.proba.append([ip[0].lproba[1,1],ip[0].lproba[0,1]])
            #===================================================================
        
        # recursion
        if self.left.node.count() > 1:
            self.left._evalCurrent(testdataset,out.left,**kwargs)
        if self.right.node.count() > 1:
            self.right._evalCurrent(testdataset,out.right,**kwargs)
            
        #return out
    
    def evaluate(self,testdataset,**kwargs):
        """
        Recursively evaluate all local models (see :func:`~classifip.models.nestedDichotomies.evalCurrent`)
        of the entire dichotomy tree for the test dataset.
        
        :param testdataset: list of input features of instances to evaluate
        :type testdataset: list
        
        :param **kwargs: should contain any parameter available to the base 
        classifier.
        
        :returns: a set of probabilistic binary tree
        :rtype: lists of :class:`~classifip.representations.binaryTree.BinaryTree`
        
        .. warning:: no check is performed on the validity of the arguments.
        
        """
        
        if self is not None :
            results = []
            for item in testdataset:
                tree = bt.BinaryTree(label=['initialization'])
                self._evalCurrent([item],tree,**kwargs) 
                results.append(tree)
        return results
    
    
    def build(self, method="random"):
        """
        Build the structure of the entire binary tree by splitting the initial
        root node (the ensemble of class values) into children nodes.
        """
        if self.node.isEmpty() :
            raise Exception("Cannot split an empty root node")
        
        if (self.left is not None) or (self.right is not None) :
            raise Exception("The given root node already has child")
        
        if self.node.count() == 1 :
            '''
            When there is only one class value in the label of the current node,
            we stop splitting.
            '''
            self.left = None
            self.right = None
        
        else :
            l_node, r_node = self.node.splitNode(method)
            self.left = NestedDichotomies(copy.copy(self.classifier),node = l_node)
            self.right = NestedDichotomies(copy.copy(self.classifier),node = r_node)
            self.left.build(method)
            self.right.build(method) 
    