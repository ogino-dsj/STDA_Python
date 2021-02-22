# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:14:27 2021

@author: Mikito Ogino Japan
Keio University, Dentsu ScienceJam Inc.
"""

import numpy as np
import copy
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class STDA:    
    def fit(self, X, label, itrmax):
        self.STDAmode = 0
        Size=list(X.shape)
        y=label;

        nclass=2;

        n_ord=len(Size)-1
        order=list(range(0,n_ord))
        
        D = [2,2]
        W = []
        for i in order:
            W.append(np.eye(Size[i]))
            
        error = np.zeros([itrmax, n_ord])
            
        n_r=1
        for nrp in range(0,itrmax):
            preW=copy.deepcopy(W)
            for i in order:
                mpy_ord=copy.deepcopy(order)
                mpy_ord.remove(i)
                mpy_ord = mpy_ord[0]
                tns_i = np.tensordot(X, W[mpy_ord], ([mpy_ord],[1]))

                Size=[tns_i.shape[0+i*2], tns_i.shape[2-i*2]]
                tns_i = tns_i.transpose(0,2,1)
                                    
                Me_all=np.mean(tns_i, axis=2)
                Sb=np.zeros([Size[i],Size[i]])
                Sw=np.zeros([Size[i],Size[i]])
                Me=np.zeros([Size[i],tns_i.shape[1],nclass])
                N = []
                for c in range(0,nclass):
                    N.append(np.sum(y==c))
                    Xw=tns_i[:,:,y==c]
                    Me[:,:,c]=np.mean(Xw, axis=2)
                    Sb=Sb+N[c]*np.dot((Me[:,:,c]-Me_all),(Me[:,:,c]-Me_all).T)
                    for j in range(0, N[c]):
                        Sw=Sw+np.dot((Xw[:,:,j]-Me[:,:,c]),(Xw[:,:,j]-Me[:,:,c]).T)
                Sb=Sb/np.sum(N)
                Sw=Sw/np.sum(N)
                eigen_d, eigen_vec = scipy.sparse.linalg.eigs(Sb,k=Size[i],M=Sb+Sw)
                sorted_eigen_d = np.sort(eigen_d)[-1::-1]
                Id = np.argsort(eigen_d)[-1::-1]
                midW=eigen_vec[:,Id[0:D[i]]]
                                
                if nrp>0:
                    for r in range(0,D[i]):
                        midW[:,r]=midW[:,r]*np.sign(np.corrcoef(midW[:,r],preW[i][r,:].T)[0][1])
                
                W[i]=midW.T;
            
            n_break=0;
            if nrp>0:
                for j in range(0,n_ord):
                    error[n_r,j]=np.linalg.norm(abs(W[j])-abs(preW[j]), ord=2)
                    if error[n_r,j]<0.00001:
                        n_break=n_break+1
                n_r=n_r+1;
            if n_break==n_ord:
                break
            
        if nrp>=itrmax:
            print('Not perfectly converged. You may try a larger number of iteration.')
        else:
            print('Converged at iteration: ',nrp)
        
        self.STDAmode = W
        #STDAprojection
        W1 = self.STDAmode[0]
        W2 = self.STDAmode[1]
        X = X.transpose(1,2,0)
        W1 = W1.transpose(1,0)
        projSTDA1=np.dot(X, W1)
        projSTDA1 = projSTDA1.transpose(1,2,0)
        W2 = W2.transpose(1,0)
        projSTDA2 = np.dot(projSTDA1, W2)
        projSTDA2 = projSTDA2.transpose(2,1,0)
        Size = list(projSTDA2.shape)
        projSTDA2 = projSTDA2.reshape(np.prod(Size[0:-1]),-1)
        
        self.lda_clf = LinearDiscriminantAnalysis()
        self.lda_clf.fit(projSTDA2.T.real, y)
        
        return projSTDA2.T.real
        
    def predict_proba(self, X):
        W1 = self.STDAmode[0]
        W2 = self.STDAmode[1]
        X = X.transpose(1,2,0)
        W1 = W1.transpose(1,0)
        projSTDA1=np.dot(X, W1)
        projSTDA1 = projSTDA1.transpose(1,2,0)
        W2 = W2.transpose(1,0)
        projSTDA2 = np.dot(projSTDA1, W2)
        projSTDA2 = projSTDA2.transpose(2,1,0)
        Size = list(projSTDA2.shape)
        projSTDA2 = projSTDA2.reshape(np.prod(Size[0:-1]),-1)
        
        y = self.lda_clf.predict_proba(projSTDA2.T.real)
        return y, projSTDA2