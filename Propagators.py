import numpy as np
from scipy.linalg import svd
import Hamiltonian
import Auxfield
import sys
import time
class Btau_s:
    def __init__(self, ht, hv, gamma, sigma, Nwrap ):
        self.ht=ht
        self.hV=hv
        self.gamma=gamma
        self.sigma=sigma
        self.Nwrap=Nwrap

        self.At=self.Amat()
        self.Bs=self.genB()

        self.At_inv=self.Amat_inv()
        self.Bs_inv=self.genB_inv()

    #constructing methods
    def Amat(self):
        A=[]
        for tau in range(self.gamma.Ntau):
            A.append(np.exp( (self.sigma*self.hV.alpha)*(self.gamma.fields[tau,:]) )) #has to be changed for different interactions
        return A

    def genB(self):
        B=[] #each matrix corresponds to B(tau,tau-1)
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

    #updating methods
    def flip_field_update_B(self, tau, site):
        self.gamma.flipfield(tau, site)
        
        self.At[tau][site]=np.exp(self.sigma*self.hV.alpha*self.gamma.fields[tau,site])
        self.At_inv[tau][site]=np.exp(-self.sigma*self.hV.alpha*self.gamma.fields[tau,site])
            
        self.Bs[tau]=(self.At[tau]*((self.ht.expTmu).T)).T
        self.Bs_inv[tau]=self.At_inv[tau]*((self.ht.expTmuinv))
        return None

    #stable matrix multiplication methods
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
            return self.Bs[in_tau]

        #initializing matrix that stores the result of the mult
        shape_mat=np.shape(self.Bs[intau])
        Res=np.eye(shape_mat[0])

        #getting the number of matrices in the product, the number of bunches of stabilization free products for a given Nwrap 
        #and the remaining ops if Ntau is not divisible by Nwrap
        tautot=fintau-intau
        groups=tautot//self.Nwrap
        residue=tautot%self.Nwrap

        #multiplication of the Nwrap groups
        for group in range(0,groups):

            temp_ind=intau+(group)*self.Nwrap
            temp_prod=self.Bs[temp_ind]

            initpos=temp_ind+1
            finpos=temp_ind+self.Nwrap
            for tau in range(initpos,finpos):
                    temp_prod=self.Bs[tau]@temp_prod
            Res=self.stabmult_svd(temp_prod,Res)
            

        if residue>0:

            temp_ind=intau+(groups)*self.Nwrap
            temp_prod=self.Bs[temp_ind]

            initpos=temp_ind+1
            finpos=temp_ind+residue

            for tau in range(temp_ind+1,temp_ind+residue):  
                    temp_prod=self.Bs[tau]@temp_prod
            Res=self.stabmult_svd(temp_prod,Res)

        return Res
                        
        

def main()->int:
    Nsites=4
    Ntau=20
    Nwrap=10 #Nwrap-1 is the number of matrix multiplications that are stable. Mult of Nwrap matrices
    dtau=0.01
    mu=0
    U=1
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Btau_s(ht, hv, gamma, 1, Nwrap)
    Bm=Btau_s(ht, hv, gamma, -1, Nwrap)
    print(Bp.gamma.fields)
    print("\n")
    Bp.flip_field_update_B(1,0)
    print(Bp.gamma.fields)
    print(Bp.Bs[1]@Bp.Bs_inv[1])
    return 0

if __name__=='__main__':
    sys.exit(main())