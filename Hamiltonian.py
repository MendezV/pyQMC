#1
import numpy as np
from scipy.linalg import circulant
from scipy.linalg import eigh
import Auxfield
import sys
class Hopping:

    def __init__(self,Nsites,dtau,mu):

        self.Nsites=Nsites
        self.dtau=dtau
        self.mu=mu
        self.t=-1
        self.T=self.t*self.genT(Nsites)
        
        #exp of the hopping matrices
        self.expT=self.expTf(dtau)
        self.expTmu=np.exp(dtau*mu)*self.expT

        #inverses
        self.expTinv=self.expTf(-dtau)
        self.expTmuinv=np.exp(-dtau*mu)*self.expTinv
    
    def genT(self,Nsites):
        vecinit=np.zeros(Nsites)
        vecinit[1]=1
        vecinit[-1]=1
        return circulant(vecinit) #nearest-neighbour chain

    def expTf(self,dtau):
        [lam,V]=eigh(self.T)
        return V@(np.diag(np.exp(dtau*lam))@V.T)
    

class Vint:

    def __init__(self, dtau, U):
        self.U=U
        self.dtau=dtau
        self.alpha=np.arccosh(np.exp(dtau*U/2))
        

def main()->int:
    Nsites=4
    dtau=0.01
    mu=0
    ht=Hopping(Nsites,dtau,mu)
    
    return 0

if __name__=='__main__':
    sys.exit(main())