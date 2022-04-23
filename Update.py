#6
import numpy as np
import Obs_eq


def MC_step(Geq_pl,Geq_min):

    Ntau=Geq_pl.stringops.gamma.Ntau
    Nsites=Geq_pl.stringops.gamma.Nsites
    Obs_store=[]
    for tau in range(Ntau):
        if tau%Geq_pl.stringops.Nwrap==0:
            Geq_pl.Glist[tau]=Geq_pl.construct_G_tau(tau)
            Geq_min.Glist[tau]=Geq_min.construct_G_tau(tau)
        for site in range(Nsites):
            [R_pl, Delta_pl]=Geq_pl.Ratio( tau, site )
            [R_min, Delta_min]=Geq_min.Ratio( tau, site )
            R=R_pl*R_min
            xi=np.random.random()
            if R>xi:
                Geq_pl.update_G_tau(tau,site, R_pl,Delta_pl) #updating Green's functions
                Geq_min.update_G_tau(tau,site, R_min,Delta_min) #updating Green's functions
                Geq_pl.stringops.flip_field_update_B( tau, site) #updating B's
                Geq_min.stringops.flip_field_update_B( tau, site) #updating B's
        indtau=int((tau+1)%Ntau)
        Geq_pl.Glist[indtau]=Geq_pl.AdvanceG(tau, 1, Geq_pl.Glist[tau])
        Geq_min.Glist[indtau]=Geq_min.AdvanceG(tau, 1, Geq_min.Glist[tau])

        # Obs_store.append(Obs_eq.Measure_ZZ(Geq_pl, Geq_min,tau))
        Obs_store.append(Obs_eq.Measure_Etot(Geq_pl, Geq_min,tau))
    
    return np.sum(Obs_store, axis=0)/Ntau
       