#7
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg.blas as bl
import sys
import Propagators
import Hamiltonian
import Auxfield
import greens_func
import Update
import time

#TODO: precision checks
#TODO: Corr as a function of distance

def read_input(filename):
    variables = {}

    with open(filename) as f:
        for line in f:
            if line[0]!="#":

                name, value = line.split("=")
                variables[name] = float(value)

    return variables

def main()->int:
    print("hos")
    variables=read_input('parameters')
    Ntau=int(variables["Ntau"])
    Nsites=int(variables["Nsites"])
    Nbins=int(variables["Nbins"])
    Nsweeps=int(variables["Nsweeps"])
    Nwrap=20
    
    t=int(variables["ham_hop"])
    U=int(variables["ham_U"])
    beta=int(variables["beta"])
    mu=int(variables["ham_mu"])
    dtau=beta/Ntau


    #initializing
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1, Nwrap)
    Bm=Propagators.Btau_s(ht, hv, gamma, -1, Nwrap)
    Geq_pl=greens_func.Geq(Bp)
    Geq_min=greens_func.Geq(Bm)

    Obs_store=[]
    for _ in range(Nbins):
        Obs_store_sweep=[]
        for _ in range(Nsweeps):
            Obs_store_sweep.append(Update.MC_step(Geq_pl,Geq_min))
        Obs_store.append(np.sum(Obs_store_sweep, axis=0)/(Nsweeps))   
    
    meanO=np.sum(Obs_store, axis=0)/(Nbins)

    # plt.imshow(meanO)
    # plt.colorbar()
    # plt.savefig("ZZcorr.png")
    plt.plot(Obs_store)
    plt.show()

    

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
