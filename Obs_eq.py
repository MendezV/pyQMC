#5
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



def Measure_U_op(Geq_pl, Geq_min,tau):
    
    Gpl=Geq_pl.Glist[tau]
    Gmin=Geq_min.Glist[tau]
    Gpl_diag=np.diag(Gpl)
    Gmin_diag=np.diag(Gmin)

    niupnidown=np.diag(np.outer(1-Gpl_diag,1-Gmin_diag))
    U=Geq_min.stringops.hV.U

    return U*np.sum(niupnidown)

def Measure_T_op(Geq_pl, Geq_min,tau):
    
    Nsites=Geq_pl.stringops.gamma.Nsites
    Gpl=Geq_pl.Glist[tau]
    Tpl=Geq_pl.stringops.ht.T
    Gmin=Geq_min.Glist[tau]
    Tmin=Geq_min.stringops.ht.T

    hopmat=(np.eye(Nsites)-Gpl.T)*Tpl+(np.eye(Nsites)-Gmin.T)*Tmin
    
    t=Geq_min.stringops.ht.t
    return t*np.sum(hopmat)

def Measure_Etot(Geq_pl, Geq_min,tau):
    
    T=Measure_T_op(Geq_pl, Geq_min,tau)
    U=Measure_U_op(Geq_pl, Geq_min,tau)
    
    print(T,U)
    return T