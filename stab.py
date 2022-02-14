import numpy as np
from scipy.linalg import svd
import sys
import Propagators
import Hamiltonian
import Auxfield
import time

class StringOps:
    def __init__(self, Ops, Nwrap):
        self.Ops=Ops
        self.Nwrap=Nwrap

    def stabmult_svd(self,A,B):
        [U,D,V]=svd(B)
        temp=(A@U)*D
        [Up,Dp,Vp]=svd(temp)
        VpV=Vp@V
        UpDp=Up*Dp
        return UpDp@VpV

        
    def Opmult_stab_LtoR(self, in_tau, fin_tau):
        
        if in_tau<fin_tau:
            intau=in_tau
            fintau=fin_tau
        elif(in_tau>fin_tau):
            intau=fin_tau
            fintau=in_tau
        else:
            return self.Ops.Bs[in_tau]

        shape_mat=np.shape(self.Ops.Bs[intau])
        Res=np.eye(shape_mat[0])
        tautot=fintau-intau
        groups=tautot//self.Nwrap
        residue=tautot%self.Nwrap
        # print('groups',groups,groups*self.Nwrap, tautot,)
        
        for group in range(0,groups):
            # print('group_c',group)
            temp_ind=intau+(group)*self.Nwrap
            temp_prod=self.Ops.Bs[temp_ind]
            for tau in range(intau+1+(group)*self.Nwrap,intau+(group+1)*self.Nwrap):
                    temp_prod=self.Ops.Bs[tau]@temp_prod
            Res=self.stabmult_svd(temp_prod,Res)
            

        if residue>0:
            temp_prod=self.Ops.Bs[groups*self.Nwrap]
            for tau in range(groups*self.Nwrap+1,fintau+1): #fintau+1 since we want to include the last matrix
                    temp_prod=self.Ops.Bs[tau]@temp_prod
            Res=self.stabmult_svd(temp_prod,Res)
        return Res
                        
        
    def Opmult_LtoR(self, intau, fintau):
        
        if intau<fintau:
            Res=self.Ops.Bs[intau]
            for tau in range(intau+1,fintau+1):
                Res=self.Ops.Bs[tau]@Res
  
        else:
            print(f'setting initial time as {fintau} and final time as {intau}')
            Res=self.Ops.Bs[fintau]
            for tau in range(fintau+1,intau):
                Res=self.Ops.Bs[tau]@Res
                
        return Res


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

    Bsp=StringOps(Bp,Nwrap)
    
    nostab=Bsp.Opmult_LtoR(0,Ntau)
    
    Bsp=StringOps(Bp,Nwrap)
    stab=Bsp.Opmult_stab_LtoR(0,Ntau)

    for Nw in range(1,Ntau):
        Bsp=StringOps(Bp,Nw)
        s=time.time()
        stab=Bsp.Opmult_stab_LtoR(0,Ntau)
        e=time.time()
        print(Nw,e-s, np.mean(np.abs(stab)))



    print(1,np.mean(np.abs(stab)))
    print(2,np.mean(np.abs(nostab)))
    print('dif',np.mean(np.abs(nostab-stab)))

    #testing equal index
    i=10
    stab=Bsp.Opmult_stab_LtoR(i,i)
    print(np.mean(np.abs(stab)))
    print(np.mean(np.abs(Bp.Bs[i])))


    return 0



if __name__=='__main__':
    sys.exit(main())