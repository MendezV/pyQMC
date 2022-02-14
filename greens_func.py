import numpy as np
from scipy.linalg import svd
from scipy.linalg import inv
import sys
import Propagators
import Hamiltonian
import Auxfield
import stab
import time
class Geq:
    def __init__(self, stringops ):
        self.stringops=stringops
        self.Glist=self.construct_G()

    def inver_IpB(self,BB):
        [U,D,V]=svd(BB)
        second_mat=np.conj(U.T)@np.conj(V.T)+np.diag(D)
        [Up,Dp,Vp]=svd(second_mat)
        return (np.conj((Vp@V).T)/Dp)@np.conj((U@Up).T)

    def naive_inver_IpB(self,BB):
        nainv=inv(np.eye(np.shape(BB)[0])+BB)
        return nainv 

    def construct_G(self):
        Glist=[]
        Ntau=self.stringops.Ops.gamma.Ntau
        for tau in range(Ntau):
            B1=self.stringops.Opmult_stab_LtoR(tau,0)
            B2=self.stringops.Opmult_stab_LtoR(Ntau-1,tau)
            BB=B1@B2
            a2= np.eye(4)+np.reshape(np.arange(16),[4,4])
            # print(np.min( np.abs(eigvals(a2))) )
            print(np.mean(np.abs(self.inver_IpB(a2)-self.naive_inver_IpB(a2))))

            Glist.append(BB)
        return Glist




def main()->int:
    Nsites=25
    Beta=5
    dtau=0.1
    Ntau=int(Beta/dtau)
    print("ntau is",Ntau)
    
    Nwrap=10
    mu=0
    U=4
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1)
    # Bm=Propagators.Btau_s(ht, hv, gamma, -1)

    Bsp=stab.StringOps(Bp,Nwrap)
    
    s=time.time()
    Geqp=Geq(Bsp)
    e=time.time()
    print(e-s)

    return 0

if __name__=='__main__':
    sys.exit(main())