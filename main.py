import numpy as np
import matplotlib as plt
import scipy.linalg.blas as bl
import sys
import time
import Hamiltonian

def main()->int:
    Ntau=sys.argv[1]
    Nsites=sys.argv[2]
    Nbins=sys.argv[3]
    Nsweeps=sys.argv[4]
    Nstab=10
    t=1
    U=1
    beta=0.1
    mu=0
    dtau=beta/Ntau




    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
