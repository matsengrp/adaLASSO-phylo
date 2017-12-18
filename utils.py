# some useful functions

import numpy as np
import re
import copy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from cStringIO import StringIO
import phyloinfer as pinf
from Bio import MissingExternalDependencyError, Phylo
from Bio.Application import _Option, AbstractCommandline
from ete3 import NodeStyle, TreeStyle, TextFace
import commands


def prox_l1(x, thr):
    return np.maximum(0.0, x-thr) - np.maximum(0.0, -x-thr)
    
    
def prox_l2(x, thr):
    return x/(1+2.0*thr)
    

def penalty_scad(x, gamma, a=3.7):
    upper_bound = np.minimum(np.abs(x), a*gamma)
    return np.sum(gamma*np.minimum(np.abs(x), gamma) + (a*gamma-.5*(upper_bound+gamma))/((a-1)*gamma)*np.maximum(upper_bound-gamma, 0.0))
    

def prox_scad(x, thr, a=3.7):
    upper_bound = np.minimum(x, a*thr) - np.minimum(x+a*thr,0.0)
    return np.maximum(0.0, x-thr) - np.maximum(0.0, -x-thr) + (1.0/(a-2))*(np.maximum(0.0, upper_bound-2*thr) - np.maximum(0.0, -upper_bound-2*thr))
    

def prox_next(x, thr, a=3.7, method='l1'):
    if method=='l1':
        return prox_l1(x, thr)
    if method=='l2':
        return prox_l2(x, thr)
    if method=='scad':
        return prox_scad(x, thr, a=a) 


def penalty_value(x, gamma, a=3.7, method='l1'):
    if method=='l1':
        return gamma*np.linalg.norm(x, 1)
    if method=='l2':
        return gamma*np.linalg.norm(x, 2)
    if method=='scad':
        return penalty_scad(x, gamma, a=a) 


class RaxmlCommandline(AbstractCommandline):
    def __init__(self, cmd='raxml-ng', **kwargs):
        self.parameters = [
            _Option(['--bootstrap', 'bootstrap'],
                    """
                    Run non-parametric bootstrap analysis. Number of bootstrap replicates can be 
                    specified with --bs--trees parameter (default 100).
                    """,
                    equate=False,
                    ),
            
            _Option(['--evaluate', 'fixtree'],
                    "Optimize model parameters and/or branch lengths on a fixed tree topology.",
                    equate=False,
                    ),
                    
            _Option(['--model', 'model'],
                    """Model of Nucleotide or Amino Acid Substitution:
                    DNA: 
                        JC, K80, F81, HKY, TN93ef, TN93, K81, K81uf, TPM2, TPM2uf, TPM3, TPM3uf, TIM1,
                        TIM1uf, TIM2, TIM2uf, TIM3, TIM3uf,TVMef, TVM, SYM, GTR
                        
                    Protein*: 
                        Dayhoff, LG, DCMut, JTT, mtREV, WAG, RtREV, CpREV, VT, Blosum62, MtMam, MtArt,
                        MtZoa, PMB, HIVb,HIVw, JTT-DCMut, FLU, StmtREV, LG4M (implies +G4),
                        LG4X (implies +R4), PROTGTR
                        
                    Fixed user-defined rates:
                        e.g. HKY{1.0/2.5} or GTR{0.5/2.0/1.0/1.2/0.1/1.0}
                    """,
                    equate=False,
                   ),
            
            _Option(['--msa','sequences'],
                    "Name of the alignment data file, in FASTA, non-interleaved PHYLIP and CATG formats.",
                    filename=True,
                    equate=False,
                   ),
            
            _Option(['--tree', 'starting_tree'],
                    "File name of a user starting tree, in Newick format.",
                    filename=True,
                    equate=False,
                   ),
                   
            _Option(['--bs-trees', 'bstrees'],
                    "Number of non-parametric bootstrap replicates.",
                    equate=False,
                    ),
                    
        ]
        AbstractCommandline.__init__(self, cmd, **kwargs)   
     
        
def readPara(filename):
    with open(filename,'r') as readin_file:
        id_line = readin_file.readline()
        ID = ''.join(re.split('\[|\]|ID:',id_line)).strip()
        result = []
        
        while True:
            line = readin_file.readline()
            if line == "": 
                break
            samp = []
            for stats in line.strip('\n').split('\t'):
                samp.append(float(stats))
            result.append(samp)
            
    return np.array(result), ID 
    

def readTree(filename, tree_format):
    with open(filename, 'r') as readin_file:
        tree_list = []
        while True:
            line = readin_file.readline()
            if line == "":
                break
            tree_list.append(pinf.Tree(line, format=tree_format))
    
    return tree_list


def saveTree(sampled_tree, filename, tree_format=5):
    if type(sampled_tree) is not list:
        sampled_tree = [sampled_tree]
        
    with open(filename,'w') as output_file:
        for tree in sampled_tree:
            tree_newick = tree.write(format=tree_format)
            output_file.write(tree_newick + '\n') 


def detection(zerobr_idx, brlen_est):
    hit_zeros = np.sum(brlen_est[zerobr_idx]==0.0)
    miss_zeros = len(zerobr_idx) - hit_zeros
    false_alarm = np.sum(brlen_est==0.0) - hit_zeros
    
    return miss_zeros, false_alarm
    
    
    
def readEdgeSupport(filename):
    EdgeSupport = {}
    with open(filename, 'r') as readin_file:
        while True:
            line = readin_file.readline()
            if not line:
                break
            line_stats = line.strip('\n\t').split('\t')
            EdgeSupport[re.findall(r"[-+]?\d*\.\d+|\d+", line_stats[0])[0]] = [float(EdgeSupp) for EdgeSupp in line_stats[1:]] 
            
    return EdgeSupport 


def addLabel(tree_list):
    for i, tree in enumerate(tree_list):
        if not tree.label:
            tree.label = 'tree_{}'.format(i+1)
            
def maptaxonname(taxon_namespace, taxa):
    for i, name in enumerate(taxon_namespace):
        name.label = taxa[i]

def Collapse(tree, threshold=1e-06):
    toCollapse = [n for n in tree.iterInternalsNoRoot() if n.br.len <=threshold]
    for n in toCollapse:
        tree.collapseNode(n)
        
def splitSupport(tree_list, consensus_dict, taxon_names, collapse_threshold = 0.0, skip=0.5):
    n_tree = len(tree_list)
    split_dict = {key:0.0 for key in consensus_dict}
    for i in range(int(n_tree*skip), n_tree):
        if collapse_threshold == 0.0:
            tree = tree_list[i]
        else:
            tree = copy.deepcopy(tree_list[i])
            Collapse(tree, collapse_threshold)
        
        tree.taxNames = taxon_names
        tree.makeSplitKeys()
        for n in tree.iterInternalsNoRoot():
            if n.br.splitKey in split_dict:
                split_dict[n.br.splitKey] += 1.0
    
    for split in split_dict:
        split_dict[split] = int('{:.0f}'.format(split_dict[split]/(n_tree-int(n_tree*skip)) * 100))
    
    return split_dict   