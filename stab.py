import numpy as np
from scipy.linalg import svd
import sys
import Propagators
import Hamiltonian
import Auxfield

class StringOps:
    def __init__(self, Ops, Nwrap):
        self.Ops=Ops
        self.Nwrap=Nwrap

    def Opmult_stab2_LtoR(self, intau, fintau):
        
        if intau<fintau:
            Res=self.Ops.Bs[intau]
            for tau in range(intau+1,fintau):
                if tau%self.Nwrap!=0:
                    Res=self.Ops.Bs[tau]@Res
                else:
                    [U,D,V]=svd(Res)
                    temp=(self.Ops.Bs[tau]@U)*D
                    [Up,Dp,Vp]=svd(temp)
                    VpV=Vp@V
                    UpDp=Up*Dp
                    Res=UpDp@VpV
        else:
            print(f'setting initial time as {fintau} and final time as {intau}')
            Res=self.Ops.Bs[fintau]
            for tau in range(fintau+1,intau):
                if tau%self.Nwrap!=0:
                    Res=self.Ops.Bs[tau]@Res
                else:
                    [U,D,V]=svd(Res)
                    temp=(self.Ops.Bs[tau]@U)*D
                    [Up,Dp,Vp]=svd(temp)
                    VpV=Vp@V
                    UpDp=Up*Dp
                    Res=UpDp@VpV
        return Res
        
    def Opmult_stab_LtoR(self, in_tau, fin_tau):
        
        if in_tau<fin_tau:
            intau=in_tau
            fintau=fin_tau
        else:
            intau=fin_tau
            fintau=in_tau

        shape_mat=np.shape(self.Ops.Bs[intau])
        Res=np.eye(shape_mat[0])
        tautot=fintau-intau
        groups=tautot//self.Nwrap
        residue=tautot%self.Nwrap
        print('groups',groups,groups*self.Nwrap, tautot,)
        
        for group in range(0,groups):
            # print('group_c',group)
            temp_ind=intau+(group)*self.Nwrap
            temp_prod=self.Ops.Bs[temp_ind]
            for tau in range(intau+1+(group)*self.Nwrap,intau+(group+1)*self.Nwrap):
                    temp_prod=self.Ops.Bs[tau]@temp_prod
            [U,D,V]=svd(Res)
            temp=(temp_prod@U)*D
            [Up,Dp,Vp]=svd(temp)
            VpV=Vp@V
            UpDp=Up*Dp
            Res=UpDp@VpV

        if residue>0:
            temp_prod=self.Ops.Bs[groups*self.Nwrap]
            for tau in range(groups*self.Nwrap+1,fintau):
                    temp_prod=self.Ops.Bs[tau]@temp_prod
            [U,D,V]=svd(Res)
            temp=(temp_prod@U)*D
            [Up,Dp,Vp]=svd(temp)
            VpV=Vp@V
            UpDp=Up*Dp
            Res=UpDp@VpV
        return Res
                        
        
    def Opmult_LtoR(self, intau, fintau):
        
        if intau<fintau:
            Res=self.Ops.Bs[intau]
            for tau in range(intau+1,fintau):
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
    print(Ntau)
    
    Nwrap=10
    mu=0
    U=4
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1)
    Bm=Propagators.Btau_s(ht, hv, gamma, -1)

    Bs=StringOps(Bp,Nwrap)
    
    stab=Bs.Opmult_stab2_LtoR(0,Ntau)
    nostab=Bs.Opmult_LtoR(0,Ntau)
    
    Bs=StringOps(Bp,Nwrap)
    stab2=Bs.Opmult_stab_LtoR(0,Ntau)


    print(1,np.mean(np.abs(stab)))
    print(2,np.mean(np.abs(nostab)))
    print(3,np.mean(np.abs(stab2)))
    print('dif',np.mean(np.abs(nostab-stab)))
    print('dif',np.mean(np.abs(stab2-stab)))
    print('dif',np.mean(np.abs(stab2-nostab)))
    return 0



if __name__=='__main__':
    sys.exit(main())