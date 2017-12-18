# non-bifurcating phylogenetic inference via adaptive LASSO

import numpy as np
import copy
from utils import prox_l1, prox_next, penalty_value, RaxmlCommandline
import phyloinfer as pinf
from Bio import MissingExternalDependencyError
from Bio.Application import _Option, AbstractCommandline

MAX_ITER = 1000
ABSTOL = 1e-04
EPS = np.finfo(float).eps
MIN_STEP_SIZE = 5e-08


def raxml(sequences, starting_tree="pars", model="JC", bootstrap=False, fixed_tree=False, bstrees=100, log=True):
    raxml_exe = None
    try:
        from Bio._py3k import getoutput
        output = getoutput("./raxml-ng -v")
        if "not found" not in output and "RAxML-NG" in output:
            raxml_exe = "./raxml-ng"
    except OSError:
        pass

    if not raxml_exe:
        raise MissingExternalDependencyError(
            "Install RAxML (binary raxml-ng) if you want to test the Bio.Phylo.Applications wrapper.")
    
    if not fixed_tree:
        if bootstrap:
            cmd = RaxmlCommandline(raxml_exe, bootstrap=None, sequences=sequences, model=model, starting_tree=starting_tree, bstrees=bstrees)
        else:
            cmd = RaxmlCommandline(raxml_exe, sequences=sequences, model=model, starting_tree=starting_tree)
    else:
        cmd = RaxmlCommandline(raxml_exe, fixtree=None, sequences=sequences, model=model, starting_tree=starting_tree)
      
    out, err = cmd()
    if log:
        print out      


def ista(model, tree, brlen, lam, gamma, beta, wts, maxiter=MAX_ITER, abstol=ABSTOL, minstepsz=MIN_STEP_SIZE):
    brlen_x = brlen
    objval_ll = []
    objval_lasso = []
    lam_ada = lam
    
    curr_ll = model.loglikelihood(tree, brlen_x)
    objval_ll.append(curr_ll)
    objval_lasso.append(curr_ll-gamma*np.linalg.norm(brlen_x*wts, 1))
    
    for k in range(maxiter):
        grad_brlen_x = -model.loglikelihood(tree, brlen_x, grad=True)
        if lam_ada < minstepsz:
            lam_ada = lam
        while True:
            brlen_z = np.maximum(prox_l1(brlen_x-lam_ada*grad_brlen_x, lam_ada*gamma*wts), 0.0)
            prop_ll = model.loglikelihood(tree, brlen_z)
            if -prop_ll <= -curr_ll + np.dot(grad_brlen_x.T, brlen_z-brlen_x) + (1.0/2/lam_ada)*np.sum((brlen_z-brlen_x)**2):
                break
            lam_ada *= beta
        brlen_x, curr_ll = brlen_z, prop_ll
        objval_ll.append(curr_ll)
        objval_lasso.append(curr_ll-gamma*np.linalg.norm(brlen_x*wts, 1))
        if abstol is not None and np.abs(objval_lasso[-1]-objval_lasso[-2]) < abstol:
            break
    
    return brlen_x, objval_ll, objval_lasso, lam_ada
    

def fista(model, tree, brlen, lam, gamma, beta, wts, prox='l1', 
          maxiter=MAX_ITER, abstol=ABSTOL, minstepsz=MIN_STEP_SIZE, stepsz_restart=True, monitor=False):
    brlen_x, brlen_prev_x = brlen, brlen
    objval_ll = []
    objval_penalized = []
    lam_ada = lam
    
    prop_ll = model.loglikelihood(tree, brlen_x)
    objval_ll.append(prop_ll)
    objval_penalized.append(prop_ll-penalty_value(brlen_x*wts, gamma, method=prox))
    
    restart = 0
    
    for k in range(maxiter):
        brlen_y = np.maximum(brlen_x + ((k-restart)/(k-restart+3.0)) * (brlen_x - brlen_prev_x), 0.0)
        curr_ll = model.loglikelihood(tree, brlen_y)
        if np.isinf(curr_ll) or (stepsz_restart and lam_ada < minstepsz):
            if monitor:
                print "iteration {}; step size {}; {}, restarting ...".format(k, lam_ada, ['step size too small', 'current loglikelihood is infinite'][np.isinf(curr_ll)])
            restart, lam_ada = k+1, lam
            continue
            
        grad_brlen_y = -model.loglikelihood(tree, brlen_y, grad=True)
        while True:
            brlen_z = np.maximum(prox_next(brlen_y-lam_ada*grad_brlen_y, lam_ada*gamma*wts, method=prox), 0.0)
                
            prop_ll = model.loglikelihood(tree, brlen_z)
            if -prop_ll <= -curr_ll + np.dot(grad_brlen_y.T, brlen_z-brlen_y) + (1.0/(2*lam_ada))*np.sum((brlen_z-brlen_y)**2):
                break
            lam_ada *= beta
        
        
        brlen_prev_x, brlen_x = brlen_x, brlen_z
        objval_ll.append(prop_ll)
        objval_penalized.append(prop_ll - penalty_value(brlen_x*wts, gamma, method=prox))
        
        if abstol is not None and np.abs(objval_penalized[-1]-objval_penalized[-2]) < abstol:
            break
    
    return brlen_x, objval_ll, objval_penalized, lam_ada, k+1
    

def adaLasso(model, tree, brlen, lam, gamma, beta, prox='l1', msteps=4, gamma_ada_penalized=1.0, maxiter=MAX_ITER,
             abstol=ABSTOL, minstepsz=MIN_STEP_SIZE, monitor=False, sparsity_monitor=False):
    wts = np.ones(2*model.ntips-3)
    brlen_x = brlen
    mean_wts, mean_wts_prev = 1.0, 1.0
    max_wts, max_wts_prev = 1.0, 1.0
    objval_ll = []
    objval_penalized = []
    brlen_x_msteps = []
    n_zeros = np.empty(msteps)
    init_lam = lam 
    

    for m in range(msteps):
        if m>0 and prox=='l2':
            brlen_x, objval_ll_step, objval_penalized_step, lam, niter = fista(model, tree, brlen_x, lam, gamma, beta, wts, maxiter=maxiter, abstol=abstol, minstepsz=minstepsz, monitor=monitor)
        else:
            brlen_x, objval_ll_step, objval_penalized_step, lam, niter = fista(model, tree, brlen_x, lam, gamma, beta, wts, prox=prox, maxiter=maxiter, abstol=abstol, minstepsz=minstepsz, monitor=monitor)
        
        n_zeros[m] = np.sum(brlen_x==0.0)
        if sparsity_monitor:
            print "cycle {}: [niter {}; # zeros: {}; step size: {}]".format(m+1, niter, n_zeros[m], lam)
            
        if lam < minstepsz:
            lam = init_lam
        
        wts = 1.0/(brlen_x+EPS)**gamma_ada_penalized
        
        mean_wts = 1.0/np.mean(1.0/wts)
        gamma *= mean_wts_prev/mean_wts
        mean_wts_prev = mean_wts
        
        
        brlen_x_msteps.append(brlen_x)
        objval_ll.extend(objval_ll_step)
        objval_penalized.extend(objval_penalized_step)
        
    return brlen_x_msteps, objval_ll, objval_penalized, n_zeros, lam

      

      