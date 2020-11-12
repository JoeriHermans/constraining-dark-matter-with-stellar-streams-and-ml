import MWPotential2014Likelihood
import astropy.units as u
import gd1_util
import hypothesis
import numpy
import numpy as np
import numpy as np
import pickle
import torch

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, turn_physical_off, vcirc
from galpy.util import bovy_conversion, bovy_coords, save_pickles, bovy_plot
from gd1_util_MWhaloshape import lb_to_phi12
from scipy import integrate, interpolate
from scipy.integrate import quad
from torch.distributions.uniform import Uniform



def allocate_prior_stream_age():
    lower = torch.tensor(3).float().to(hypothesis.accelerator)
    upper = torch.tensor(7).float().to(hypothesis.accelerator)

    return Uniform(lower, upper)


def allocate_prior_wdm_mass():
    lower = torch.tensor(1).float().to(hypothesis.accelerator)
    upper = torch.tensor(50).float().to(hypothesis.accelerator)

    return Uniform(lower, upper)


def load_observed_gd1(path, phi, degree=1):
    data = np.genfromtxt(path, names=True)
    phi_max = max(phi) + 5 # For stability in fitting the splines
    phi_min = min(phi) - 5 # For stability in fitting the splines
    phi_data = data["phi1mid"]
    if phi_min < min(phi_data) or phi_max > max(phi_data):
        raise ValueError("Angles not supported by observation.")
    indices = (phi_data <= phi_max) & (phi_data >= phi_min)
    phi_data = phi_data[indices]
    linear_density = data["lindens"][indices]
    error = data["e_lindens"][indices]
    trend = np.polyfit(phi_data, linear_density, deg=degree)
    fitted = np.poly1d(trend)(phi_data)
    error /= fitted
    linear_density /= fitted
    # Fit a spline and extract the requested values
    l = np.array(linear_density)
    fit_density = interpolate.InterpolatedUnivariateSpline(phi_data, linear_density)
    fit_error = interpolate.InterpolatedUnivariateSpline(phi_data, error)
    linear_density = fit_density(phi)
    trend = np.polyfit(phi, linear_density, deg=degree)
    fitted = np.poly1d(trend)(phi)
    linear_density /= fitted
    error = fit_error(phi)
    error /= fitted

    return linear_density, error


h=0.6774
ro=8.
vo=220.


def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
                for ti in np.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
            for ti in times.split(',')]


def parse_mass(mass):
    return [float(m) for m in mass.split(',')]


def nsubhalo(m):
    return 0.3*(10.**6.5/m)


def rs(m,plummer=False,rsfac=1.):
    if plummer:
        #print ('Plummer')
        return 1.62*rsfac/ro*(m/10.**8.)**0.5
    else:
        return 1.05*rsfac/ro*(m/10.**8.)**0.5


def alpha(m_wdm):
    return (0.048/h)*(m_wdm)**(-1.11) #in Mpc , m_wdm in keV


def lambda_hm(m_wdm):
    nu=1.12
    return 2*numpy.pi*alpha(m_wdm)/(2**(nu/5.) - 1.)**(1/(2*nu))


def M_hm(m_wdm):
    Om_m=0.3089
    rho_c=1.27*10**11 #Msun/Mpc^3
    rho_bar=Om_m*rho_c
    return (4*numpy.pi/3)*rho_bar*(lambda_hm(m_wdm)/2.)**3


def Einasto(r):
    al=0.678 #alpha_shape
    rm2=199 #kpc, see Erkal et al 1606.04946 for scaling to M^1/3
    return numpy.exp((-2./al)*((r/rm2)**al -1.))*4*numpy.pi*(r**2)


def dndM_cdm(M,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #c0kpc=2.02*10**(-13) #Msun^-1 kpc^-3 from Denis' paper
    m0=2.52*10**7 #Msun from Denis' paper
    return c0kpc*((M/m0)**mf_slope)


def fac(M,m_wdm):
    beta=-0.99
    gamma=2.7
    return (1.+gamma*(M_hm(m_wdm)/M))**beta


def dndM_wdm(M,m_wdm,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    return fac(M,m_wdm)*dndM_cdm(M,c0kpc=c0kpc,mf_slope=mf_slope)


def nsub_cdm(M1,M2,r=20.,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #number density of subhalos in kpc^-3
    m1=10**(M1)
    m2=10**(M2)
    return integrate.quad(dndM_cdm,m1,m2,args=(c0kpc,mf_slope))[0]*integrate.quad(Einasto,0.,r)[0]*(8.**3.)/(4*numpy.pi*(r**3)/3) #in Galpy units


def nsub_wdm(M1,M2,m_wdm,r=20.,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    m1=10**(M1)
    m2=10**(M2)
    return integrate.quad(dndM_wdm,m1,m2,args=(m_wdm,c0kpc,mf_slope))[0]*integrate.quad(Einasto,0.,r)[0]*(8.**3)/(4*numpy.pi*(r**3)/3) #in Galpy units


def simulate_subhalos_mwdm(sdf_pepper,m_wdm,mf_slope=-1.9,c0kpc=2.02*10**(-13),r=20.,Xrs=5.,sigma=120./220.):

    Mbin_edge=[5.,6.,7.,8.,9.]
    Nbins=len(Mbin_edge)-1
    #compute number of subhalos in each mass bin
    nden_bin=np.empty(Nbins)
    rate_bin=np.empty(Nbins)
    for ll in range(Nbins):
        nden_bin[ll]=nsub_wdm(Mbin_edge[ll],Mbin_edge[ll+1],m_wdm=m_wdm,r=r,c0kpc=c0kpc,mf_slope=mf_slope)
        Mmid=10**(0.5*(Mbin_edge[ll]+Mbin_edge[ll+1]))
        rate_bin[ll]=sdf_pepper.subhalo_encounters(sigma=sigma,nsubhalo=nden_bin[ll],bmax=Xrs*rs(Mmid,plummer=True))

    rate = np.sum(rate_bin)

    Nimpact= numpy.random.poisson(rate)

    norm= 1./quad(lambda M : fac(M,m_wdm)*((M)**(mf_slope +0.5)),10**(Mbin_edge[0]),10**(Mbin_edge[Nbins]))[0]

    def cdf(M):
        return quad(lambda M : norm*fac(M,m_wdm)*(M)**(mf_slope +0.5),10**Mbin_edge[0],M)[0]

    MM=numpy.linspace(Mbin_edge[0],Mbin_edge[Nbins],10000)

    cdfl=[cdf(i) for i in 10**MM]
    icdf= interpolate.InterpolatedUnivariateSpline(cdfl,10**MM,k=1)
    timpact_sub= numpy.array(sdf_pepper._uniq_timpact)[numpy.random.choice(len(sdf_pepper._uniq_timpact),size=Nimpact,
                                p=sdf_pepper._ptimpact)]
    # Sample angles from the part of the stream that existed then
    impact_angle_sub= numpy.array([sdf_pepper._icdf_stream_len[ti](numpy.random.uniform())
            for ti in timpact_sub])

    sample_GM=lambda: icdf(numpy.random.uniform())/bovy_conversion.mass_in_msol(vo,ro)
    GM_sub= numpy.array([sample_GM() for a in impact_angle_sub])
    rs_sub= numpy.array([rs(gm*bovy_conversion.mass_in_msol(vo,ro)) for gm in GM_sub])
    # impact b
    impactb_sub= (2.*numpy.random.uniform(size=len(impact_angle_sub))-1.)*Xrs*rs_sub
    # velocity
    subhalovel_sub= numpy.empty((len(impact_angle_sub),3))
    for ii in range(len(timpact_sub)):
        subhalovel_sub[ii]=sdf_pepper._draw_impact_velocities(timpact_sub[ii],sigma,impact_angle_sub[ii],n=1)[0]
    # Flip angle sign if necessary
    if not sdf_pepper._gap_leading: impact_angle_sub*= -1.

    return impact_angle_sub,impactb_sub,subhalovel_sub,timpact_sub,GM_sub,rs_sub


def compute_obs_density_no_interpolation(phi1, apars, dens_apar):
    apar_edge=[]
    phi1_edge=[]

    abw0=apars[1]-apars[0]
    apar_edge.append(apars[0]-(abw0/2.))

    phi1bw0=phi1[1]-phi1[0]
    phi1_edge.append(phi1[0]-(phi1bw0/2.))

    for ii in range(len(apars)-1):
        abw=apars[ii+1]-apars[ii]
        phi1bw=phi1[ii+1]-phi1[ii]
        apar_edge.append(apars[ii]+abw/2.)
        phi1_edge.append(phi1[ii]+phi1bw/2.)

    abw_last=apars[len(apars)-1]-apars[len(apars)-2]
    apar_edge.append(apars[len(apars)-1]+(abw_last/2.))

    phi1bw_last=phi1[len(phi1)-1]-phi1[len(phi1)-2]
    phi1_edge.append(phi1[len(phi1)-1]+(phi1bw_last/2.))

    #compute the Jacobian d(apar)/d(phi1) using finite difference method
    dapar_dphi1=np.fabs(numpy.diff(apar_edge)/numpy.diff(phi1_edge))
    density = dens_apar * dapar_dphi1

    return density


def compute_obs_density(phi1,apars,dens_apar,Omega):

    apar_edge=[]
    phi1_edge=[]

    abw0=apars[1]-apars[0]
    apar_edge.append(apars[0]-(abw0/2.))

    phi1bw0=phi1[1]-phi1[0]
    phi1_edge.append(phi1[0]-(phi1bw0/2.))

    for ii in range(len(apars)-1):
        abw=apars[ii+1]-apars[ii]
        phi1bw=phi1[ii+1]-phi1[ii]
        apar_edge.append(apars[ii]+abw/2.)
        phi1_edge.append(phi1[ii]+phi1bw/2.)

    abw_last=apars[len(apars)-1]-apars[len(apars)-2]
    apar_edge.append(apars[len(apars)-1]+(abw_last/2.))

    phi1bw_last=phi1[len(phi1)-1]-phi1[len(phi1)-2]
    phi1_edge.append(phi1[len(phi1)-1]+(phi1bw_last/2.))

    #compute the Jacobian d(apar)/d(phi1) using finite difference method
    dapar_dphi1=np.fabs(numpy.diff(apar_edge)/numpy.diff(phi1_edge))
    #print (dapar_dphi1)

    #Interpolate dens(apar)
    ipdens_apar= interpolate.InterpolatedUnivariateSpline(apars,dens_apar)

    #Interpolate apar(phi1)
    if phi1[1] < phi1[0] : # ad-hoc way of checking whether increasing or decreasing
        ipphi1= interpolate.InterpolatedUnivariateSpline(phi1[::-1],apars[::-1])
        #Interpolate Jacobian
        ipdapar_dphi1=interpolate.InterpolatedUnivariateSpline(phi1[::-1],dapar_dphi1[::-1])
        #Interpolate density(phi1) by multiplying by jacobian
        dens_phi1=interpolate.InterpolatedUnivariateSpline(phi1[::-1],ipdens_apar(ipphi1(phi1[::-1]))*ipdapar_dphi1(phi1[::-1]))

    else :
        ipphi1= interpolate.InterpolatedUnivariateSpline(phi1,apars)
        #Interpolate Jacobian
        ipdapar_dphi1=interpolate.InterpolatedUnivariateSpline(phi1,dapar_dphi1)
        #Interpolate density(phi1) by multiplying by jacobian
        dens_phi1=interpolate.InterpolatedUnivariateSpline(phi1,ipdens_apar(ipphi1(phi1))*ipdapar_dphi1(phi1))

    return (dens_phi1(phi1))
