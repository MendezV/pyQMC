import numpy as np
from scipy.linalg import svd
import sys
import Propagators
import Hamiltonian
import Auxfield
import stab

class Geq:
    def __init__(self, stringops ):
        pass



def main()->int:
    Nsites=20
    Ntau=20
    dtau=0.01
    Nwrap=1
    mu=0
    U=10
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1)
    Bm=Propagators.Btau_s(ht, hv, gamma, -1)
    Bsp=stab.StringOps(Bp,Nwrap)
    Bsm=stab.StringOps(Bm,Nwrap)
    stab=Bsp.Opmult_stab_LtoR(0,Ntau-1)
    stab=Bsm.Opmult_stab_LtoR(0,Ntau-1)

    Geqp=Geq()
    return 0

if __name__=='__main__':
    sys.exit(main())