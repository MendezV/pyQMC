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
    Nskip=int(variables["Nskip"])
    Nwrap=2
    
    t=float(variables["ham_hop"])
    U=float(variables["ham_U"])
    beta=float(variables["beta"])
    mu=float(variables["ham_mu"])
    dtau=beta/Ntau


    #initializing
    np.random.seed(1)
    ht=Hamiltonian.Hopping(Nsites,dtau,mu,t)
    hv=Hamiltonian.Vint(dtau,U)
    gamma=Auxfield.AuxField(Ntau,Nsites)
    Bp=Propagators.Btau_s(ht, hv, gamma, 1, Nwrap)
    Bm=Propagators.Btau_s(ht, hv, gamma, -1, Nwrap)
    Geq_pl=greens_func.Geq(Bp)
    Geq_min=greens_func.Geq(Bm)
    sim=Update.Simulation(Geq_pl,Geq_min, Nsweeps)

    Obs_store=[]
    for bin in range(Nbins):
        Obs_store_sweep=[]
        print('bin', bin)
        sim.MC_step(Obs_store_sweep)
        Obs_store.append(np.sum(Obs_store_sweep, axis=0)/(Nsweeps))   
    
    meanO=np.sum(Obs_store[Nskip:], axis=0)/(Nbins-Nskip)
    print('meanop', meanO, np.size(Obs_store[Nskip:]),(Nbins-Nskip))
    # plt.imshow(meanO)
    # plt.colorbar()
    # plt.savefig("ZZcorr.png")
    plt.plot(Obs_store)
    plt.show()

    

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
