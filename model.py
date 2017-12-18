# some benchmark models

import numpy as np
import phyloinfer as pinf
        
    
####   Phylogenetic Inference Model
## This is a simple wrapper for easy usage of phyloinfer.

class PHY:
    def __init__(self, pden, subModel, data, scale=0.1):
        self.pden = pden  # stationary distribution of the continuous time Markov chain model.
        Qmodel, Qpara = subModel # substitution model 
        if Qmodel == "JC":
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompJC()
        if Qmodel == "HKY":
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompHKY(pden, Qpara)
        if Qmodel == "GTR":
            AG, AC, AT, GC, GT, CT = Qpara
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompGTR(pden, AG, AC, AT, GC, GT, CT)
        
        # initialized the conditional likelihood vector 
        self.L = pinf.Loglikelihood.initialCLV(data)
        self.nsites = len(data[0])
        
        self.scale = scale  # the branch length exponential prior parameter
        self.ntips = len(data)  # number of tips 
        
    def init_tree(self, branch='random'):
        tree = pinf.Tree()
        tree.populate(self.ntips)
        tree.unroot()
        pinf.tree.init(tree, branch=branch)
        return tree
                
    def loglikelihood(self, tree, branch, grad=False):
        return pinf.Loglikelihood.phyloLoglikelihood(tree, branch, self.D, self.U, self.U_inv, self.pden, self.L, grad=grad)    

    def setbranch(self, tree, branch):
        pinf.branch.set(tree, branch)
    
    def idx2nodeMAP(self, tree):
        return pinf.tree.idx2nodeMAP(tree)
    
    def saveTree(self, tree, filename, tree_format=9):
        pinf.result.saveTree(tree, filename, tree_format=tree_format)
    
    def check_llgrad(self, tree, branch, db):
        check_grad = []
        for i in range(len(branch)):
            new_branch = branch.copy()
            new_branch[i] += db
            check_grad.append((self.loglikelihood(tree, new_branch)-self.loglikelihood(tree, branch))/db)
        
        return np.array(check_grad)
