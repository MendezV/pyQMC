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

        self.At=self.Amat()
        self.Bs=self.genB()

        self.At_inv=self.Amat_inv()
        self.Bs_inv=self.genB_inv()


    def Amat(self):
        A=[]
        for tau in range(self.gamma.Ntau):
            A.append(np.exp( (self.sigma*self.hV.alpha)*(self.gamma.fields[tau,:]) )) #has to be changed for different interactions
        return A

    def genB(self):
        B=[]
        for tau in range(self.gamma.Ntau):
            Bta=(self.At[tau]*((self.ht.expTmu).T)).T
            # Bta=np.diag(At[tau])@self.ht.expTmu Slower way scales worse
            B.append(Bta)

        return B
    
    def Amat_inv(self):
        A=[]
        for tau in range(self.gamma.Ntau):
            A.append(np.exp(-self.sigma*self.hV.alpha*self.gamma.fields[tau,:])) #has to be changed for different interactions
        return A

    def genB_inv(self):
        B=[]
        for tau in range(self.gamma.Ntau):
            Bta=self.At_inv[tau]*((self.ht.expTmuinv))
            B.append(Bta)

        return B

    def flip_field_update_B(self, tau, site):
        self.gamma.flipfield(tau, site)
        
        self.At[tau][site]=np.exp(self.sigma*self.hV.alpha*self.gamma.fields[tau,site])
        self.At_inv[tau][site]=np.exp(-self.sigma*self.hV.alpha*self.gamma.fields[tau,site])
            
        self.Bs[tau]=(self.At[tau]*((self.ht.expTmu).T)).T
        self.Bs_inv[tau]=self.At_inv[tau]*((self.ht.expTmuinv))
        return None

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
    print(Bp.gamma.fields)
    Bp.flip_field_update_B(1,0)
    print(Bp.Bs[1]@Bp.Bs_inv[1])
    return 0

if __name__=='__main__':
    sys.exit(main())