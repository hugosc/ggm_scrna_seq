import numpy as np
import graph_ecp as gcp
from scipy.stats import poisson
from scipy.linalg import lu
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse import eye
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
import pickle

from scipy.sparse import csr_matrix, spmatrix, csc_matrix


def schur_complement(M, M_inv, idx):
    A_idx = idx
    B_idx = np.setdiff1d(np.arange(M.shape[0]), A_idx)
    AB_sub = M[np.ix_(A_idx, B_idx)]
    sub_inv = sym_submatrix_inverse(M_inv, A_idx)
    
    return M[np.ix_(A_idx, A_idx)] - (AB_sub @ sub_inv @ AB_sub.T)


def sym_submatrix_inverse(inv, exclude_idx):
    
    maintain_idx = np.setdiff1d(np.arange(inv.shape[0]), exclude_idx)
    W = inv[np.ix_(maintain_idx, maintain_idx)]
    X = inv[np.ix_(maintain_idx, exclude_idx)]
    Z = inv[np.ix_(exclude_idx, exclude_idx)]
    
    return W - np.matmul(np.matmul(X,np.linalg.inv(Z)), X.T)



def PSD_add_and_update_inverse(M, Inv, vals, idx):
    
    L, U = lu(vals, permute_l=True, overwrite_a=True)
    
    L_ = np.zeros((M.shape[0],vals.shape[1]))
    L_[np.ix_(idx, np.arange(vals.shape[1]))] = L 
    
    U_ = np.zeros((vals.shape[1], M.shape[0]))
    U_[np.ix_(np.arange(vals.shape[1]), idx)] = U 
    
    M[np.ix_(idx,idx)] += vals
    
    div_term = U_ @ Inv @ L_
    div_term[np.diag_indices(div_term.shape[0])] += 1
    
    Inv -= Inv @ L_ @ np.linalg.inv(div_term) @  U_ @ Inv
    
    
    return M, Inv, L_

def stable_conditional_decorrelation(M, idx):
    
    compl_idx = np.setdiff1d(np.arange(M.shape[0]), idx)
    
    S = np.linalg.lstsq(M[np.ix_(compl_idx, compl_idx)], M[np.ix_(compl_idx, idx)])[0]
    
    new_ = M[np.ix_(idx, compl_idx)] @ S
    new_[np.diag_indices_from(new_)] = M[idx, idx]
    
    delta = np.max(np.abs(M[np.ix_(idx,idx)] - new_))
    
    M[np.ix_(idx,idx)] = new_
    
    return M, delta

def stable_MLE_ggm(empirical_cov, G, max_iter, epsilon):
    
    non_edges = gcp.ind_set_nedge_partition(G.shape[0], G)
    
    Cov = empirical_cov.copy()
    it = 0

    while it < max_iter:
        
        smaller_than_epsilon = True
        
        for nedge in non_edges:
                        
            if nedge[0] == nedge[1]:
                continue
                
            
            _, delta = stable_conditional_decorrelation(Cov, nedge)
                 
            it += 1
            
            if delta > epsilon:
                smaller_than_epsilon = False
            if it >= max_iter:
                break
                
        if smaller_than_epsilon:
            break
            
    return Cov, it

def conditional_decorrelation(M, Inv, idx):
    
    Delta = -schur_complement(M, Inv, idx)
    
    Delta[np.diag_indices_from(Delta)] = 0
    
    #print(Delta)
    
    PSD_add_and_update_inverse(M, Inv, Delta, idx)
    
    return M, Inv, np.max(Delta)


def MLE_ggm(empirical_cov, G, max_iter, epsilon):
    
    non_edges = np.argwhere(G == False)
    
    Cov = empirical_cov.astype(np.longdouble)
    Prec = np.linalg.inv(empirical_cov).astype(np.longdouble)
    
    
    it = 0
    
    while it < max_iter:
        
        smaller_than_epsilon = True
        
        for nedge in non_edges:
                        
            if nedge[0] == nedge[1]:
                continue
                
            
            _, _, delta = conditional_decorrelation(Cov, Prec, nedge)
            
            
                    
            it += 1
            
            if delta > epsilon:
                smaller_than_epsilon = False
            if it >= max_iter:
                break
        print("Inverse error:", np.abs(1-np.sum(Cov @ Prec)/Cov.shape[0]))
        if smaller_than_epsilon:
            break
                
    return Cov, Prec, it


def gnp(n, p):
    adj = np.random.choice([False, True], p=[1-p,p],size=(n, n))
    adj = np.triu(adj) + np.triu(adj, k=1).T

    return adj
            
def random_ggm(n, p, num_samples, sparse_precision=False):
    """Produce a random gaussian multivariate distribution
    with a sparse conditional dependence structure, following a gnp.


    Args:
        n ([type]): [description]
        p ([type]): [description]
        num_samples ([type]): [description]
        sparse_precision (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """    
    adj = gnp(n, p)

    P = (np.random.rand(n,n) - 0.5)
    P = (P@P.T) * adj
    
    smallest = -np.abs(np.linalg.eigvalsh(P)[0])
    
    P[np.diag_indices(n)] -= smallest -0.1*(np.random.rand(n) + .5)
    
    Cov = np.linalg.inv(P)
    
    L = np.linalg.cholesky(Cov)
    
    X = np.random.normal(scale=1, size=(num_samples, n))
    
    if sparse_precision:
       P =  csr_matrix(P)

    return P, Cov, np.dot(L, X.T).T

def clique_edge_partition(G):
    
    edges = list(zip(*G.nonzero())) if isinstance(G, spmatrix) else np.argwhere(G != 0).tolist()
    
    return gcp.clique_edge_partition(G.shape[0], edges)
    
class GGMOptimizer:
    def __init__(self, empirical_cov: np.ndarray, G: np.ndarray, max_iter: int, epsilon: float, sparse=True):
        self.empirical_cov = empirical_cov
        self.G = G
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.sparse = sparse

        self._delta = 10e10
        self._iter = 0
        self._sub_iter = 0

    @staticmethod
    def complement_from_clique(K: np.ndarray, clique: List[Tuple[int, int]]) -> np.ndarray:
        #S = np.linalg.lstsq(M[np.ix_(compl_idx, compl_idx)], M[np.ix_(compl_idx, idx)])[0]
        A, B = clique, np.setdiff1d(np.arange(K.shape[0]), clique)
        sub = K[np.ix_(B, A)]
        return sub.T @ spsolve(K[np.ix_(B, B)], sub)

    def instantiate_cliques_and_vars(self):
        print("initializing variables...")
        if self.sparse:
            self.K = eye(self.G.shape[0]).tocsc()
            self.K_old = csc_matrix(self.K.shape)
        else:
            self.K = np.eye(self.G.shape[0])
            self.K_old = self.K.copy()

        print("Obtaining cliques...")
        self.cliques = clique_edge_partition(self.G)
        self._clique_iterator = tqdm(self.cliques).__iter__()

    def step(self):
        if (self._iter >= self.max_iter) or (self._delta <= self.epsilon):
            print("Reached solution")
            return False
        try:
            clique = next(self._clique_iterator)
            self._sub_iter += 1
        except:      
            self._reset_iteration()
            clique = next(self._clique_iterator)
            self._sub_iter = 1
        try:
            inverse_sub_cov = np.linalg.inv(self.empirical_cov[np.ix_(clique, clique)])
        except:
            print("error at", clique)
            return False
        self.K[np.ix_(clique, clique)] = inverse_sub_cov + GGMOptimizer.complement_from_clique(self.K, clique)

        return True

    def save(self, fpath):
        temp = self._clique_iterator
        self._clique_iterator = None
        with open(fpath, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._clique_iterator = temp

    @staticmethod
    def load(fpath):
        with open(fpath, 'rb') as handle:
            obj = pickle.load(handle)
        obj._clique_iterator = tqdm(obj.cliques).__iter__()
        for _ in range(obj._sub_iter):
            next(obj._clique_iterator)
        return obj
        
    def _reset_iteration(self):
        self._delta = np.max(np.abs(self.K_old - self.K))
        print("finished iteration", self._iter, "with delta =", self._delta)
        self._clique_iterator = tqdm(self.cliques).__iter__()
        self._iter += 1
        print("Starting iteration", self._iter, flush=True)
        self.K_old[:,:] = self.K