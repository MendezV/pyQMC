import numpy as np
import Hamiltonian
import Auxfield
import sys
import time
class Btau_s:
    def __init__(self, ht, hv, gamma, sigma ):
        self.ht=ht
        self.hV=hv
        self.gamma=gamma
        self.sigma=sigma

        At=self.Amat()
        self.Bs=self.genB(At)

    def Amat(self):
        A=[]
        for tau in range(self.gamma.Ntau):
            A.append(np.exp(self.sigma*self.hV.alpha*self.gamma.fields[tau,:])) #has to be changed for different interactions
        return A

    def genB(self,At):
        B=[]
        for tau in range(self.gamma.Ntau):
            Bta=(At[tau]*((self.ht.expTmu).T)).T
            # Bta=np.diag(At[tau])@self.ht.expTmu Slower way scales worse
            B.append(Bta)

        return B

def main()->int:
    Nsites=4
    Ntau=3
    dtau=0.01
    mu=0
    U=1
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Btau_s(ht, hv, gamma, 1)
    Bm=Btau_s(ht, hv, gamma, -1)
    print(Bp.Bs)
    return 0

if __name__=='__main__':
    sys.exit(main())