import numpy as np
import matplotlib as plt
import scipy.linalg.blas as bl



def Measure_ZZ(Geq_pl, Geq_min,tau):
    Nsites=Geq_pl.stringops.gamma.Nsites
    Gpl=Geq_pl.Glist[tau]
    Gmin=Geq_min.Glist[tau]
    Gpl_diag=np.diag(Gpl)
    Gmin_diag=np.diag(Gmin)

    niupnjdown=np.outer(1-Gpl_diag,1-Gmin_diag)
    nidownnjup=niupnjdown.T
    niupnjup=np.outer(1-Gpl_diag,1-Gpl_diag)+(np.eye(Nsites)-Gpl.T)@Gpl
    nidownnjsown=np.outer(1-Gmin_diag,1-Gmin_diag)+(np.eye(Nsites)-Gmin.T)@Gmin

    return niupnjdown+nidownnjup+niupnjup+nidownnjsown


    