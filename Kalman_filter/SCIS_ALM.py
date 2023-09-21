from math import inf
import numpy as np

class Opts:
    def __init__(self,mxitr,mu0,muf,rmu,tol_gap,tol_frel,tol_Xrel,tol_Yrel,numDG,record,sigma):
        self.mxitr = mxitr
        self.mu0 = mu0
        self.muf = muf
        self.rmu = rmu
        self.tol_gap = tol_gap
        self.tol_frel = tol_frel
        self.tol_Xrel = tol_Xrel
        self.tol_Yrel = tol_Yrel
        self.numDG = numDG
        self.record = record
        self.sigma = sigma
    
class Out:
    def __init__(self,X,Y,itr,pinf,obj,gapX,gapY,gap):
        self.X = X
        self.Y = Y
        self.itr = itr
        self.obj = obj
        self.gapX = gapX
        self.gapY = gapY
        self.pinf = pinf
        self.gap = gap

    
def SCIS_ALM(S,rho,opts):
    n = S.shape[0]
    X = np.zeros((n,n))
    Y = np.zeros((n,n))
    gradgY = np.zeros((n,n))
    mu = opts.mu0
    fc = 1e10
    dualgap = 1e10
    sigma = opts.sigma
    for itr in range(opts.mxitr):
        Xp = X
        W = Y/mu - gradgY - S
        W = (W+W.T)/2
        d,V=np.linalg.eig(W)
        gamma = (mu*d+np.sqrt((mu*d)**2+4*mu))/2
        X = V@np.diag(gamma)@V.T
        gradfX = -V@np.diag(1/gamma)@V.T+S

        Yp = Y
        Y = X-mu*gradfX-mu*np.minimum(rho,np.maximum(-rho,(X-mu*gradfX)/(sigma+mu)))
        gradgY = np.minimum(rho,np.maximum(-rho,Y/sigma))

        dualgap_plus_n_X = np.sum(np.sum(S@X)) + rho* np.sum(np.sum(np.abs(X)))
        dualgap_X = dualgap_plus_n_X - n

        dualgap_plus_n_Y = np.sum(np.sum(S@Y)) + rho* np.sum(np.sum(np.abs(Y)))
        dualgap_Y = dualgap_plus_n_Y - n

        
        nrmXmY = np.linalg.norm(X-Y,'fro')
        fp = fc
        fc = -(np.sum(np.log(gamma)) - dualgap_plus_n_X)
        pinf = nrmXmY/np.max([np.linalg.norm(X,'fro'),np.linalg.norm(Y,'fro')])
        frel = np.abs(fp - fc)/np.max(np.abs([fp,fc,1]))
        Xrel = np.linalg.norm(X-Xp,'fro')/np.max([1,np.linalg.norm(X,'fro'),np.linalg.norm(Xp,'fro')])
        Yrel = np.linalg.norm(Y-Yp,'fro')/np.max([1,np.linalg.norm(Y,'fro'),np.linalg.norm(Yp,'fro')])   

        stop = ( frel < opts.tol_frel ) and ( Xrel < opts.tol_Xrel ) and ( Yrel<opts.tol_Yrel )

        if np.mod(itr,opts.numDG) == 0:
            Lambda = gradgY
            d,V = np.linalg.eig(S+Lambda)
            dualgap = -np.sum(np.log(gamma))+np.sum(np.sum(S@X))+rho*np.sum(np.sum(abs(X))) - np.sum(np.log(d))-n
            stop = stop or (dualgap < opts.tol_gap)
            mu = np.max([mu * opts.rmu, opts.muf])

        if stop:
            return Out(X,Y,itr,pinf,np.sum(np.log(gamma)) - dualgap_plus_n_X,dualgap_X,dualgap_Y,dualgap)
      

