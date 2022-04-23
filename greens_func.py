#4
import numpy as np
from scipy.linalg import svd
from scipy.linalg import inv
import sys
import Propagators
import Hamiltonian
import Auxfield
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

    def construct_G_tau(self,tau):

        Ntau=self.stringops.gamma.Ntau
        B1=self.stringops.Opmult_stab_LtoR(tau,0) #both sides are inclusive, second argument is rightmost index, first argument is leftmost index
        B2=self.stringops.Opmult_stab_LtoR(Ntau-1,tau+1) #both sides are inclusive
        BB=B1@B2

        return self.inver_IpB(BB)

    def AdvanceG_multistep(self, tau, steps, Gtau):
        Ntau=self.stringops.gamma.Ntau
        init=int((tau)%Ntau)
        fini=int((tau+steps)%Ntau)
        Gtau_ad=Gtau
        for t in range(init,fini):
            print("i advanced ", t)
            Gtau_ad=self.stringops.Bs[t]@(Gtau_ad@self.stringops.Bs_inv[t])
        return Gtau_ad
    
    def AdvanceG(self, tau, steps, Gtau):
        Gtau_ad=self.stringops.Bs[tau]@(Gtau@self.stringops.Bs_inv[tau])
        return Gtau_ad

    def construct_G(self):
        Glist=[]
        for tau in range(self.stringops.gamma.Ntau):
            Glist.append(self.construct_G_tau(tau))
        return Glist

    def Ratio(self, tau, site ):
        x_new=self.stringops.gamma.propose_field( tau, site)
        x_old=self.stringops.gamma.fields[tau,site]
        alpha=self.stringops.hV.alpha
        Delta=np.exp(alpha*(x_new-x_old))-1
        Ratio=1+Delta*(1-self.Glist[tau][site,site])
        return Ratio,Delta

    def update_G_tau(self,tau,site, Ratio,Delta):
        modG=(Delta/Ratio)*np.outer(self.Glist[tau][:,site],self.Glist[tau][site,:])
        self.Glist[tau]=self.Glist[tau]-modG
        return None



def main()->int:
    
    Nsites=25
    Beta=5
    dtau=0.1
    Ntau=int(Beta/dtau)
    print("ntau is",Ntau)
    
    Nwrap=10
    mu=0
    U=4
    t=-1
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu,t)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1, Nwrap)
    # Bm=Propagators.Btau_s(ht, hv, gamma, -1)

    
    s=time.time()
    Geqp=Geq(Bp)
    e=time.time()
    print(e-s)

    return 0

if __name__=='__main__':
    sys.exit(main())