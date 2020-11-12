import copy
import numpy
from scipy import interpolate
from galpy import potential
from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from galpy.actionAngle import actionAngleTorus
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_conversion, bovy_coords
import MWPotential2014Likelihood
from MWPotential2014Likelihood import _REFR0, _REFV0
# Coordinate transformation routines
_TKOP= numpy.zeros((3,3))
_TKOP[0,:]= [-0.4776303088,-0.1738432154,0.8611897727]
_TKOP[1,:]= [0.510844589,-0.8524449229,0.111245042]
_TKOP[2,:]= [0.7147776536,0.4930681392,0.4959603976]
@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0,1],[0,1])
def lb_to_phi12(l,b,degree=False):
    """
    NAME:
       lb_to_phi12
    PURPOSE:
       Transform Galactic coordinates (l,b) to (phi1,phi2)
    INPUT:
       l - Galactic longitude (rad or degree)
       b - Galactic latitude (rad or degree)
       degree= (False) if True, input and output are in degrees
    OUTPUT:
       (phi1,phi2) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    #First convert to ra and dec
    radec= bovy_coords.lb_to_radec(l,b)
    ra= radec[:,0]
    dec= radec[:,1]
    XYZ= numpy.array([numpy.cos(dec)*numpy.cos(ra),
                      numpy.cos(dec)*numpy.sin(ra),
                      numpy.sin(dec)])
    phiXYZ= numpy.dot(_TKOP,XYZ)
    phi2= numpy.arcsin(phiXYZ[2])
    phi1= numpy.arctan2(phiXYZ[1],phiXYZ[0])
    phi1[phi1<0.]+= 2.*numpy.pi
    return numpy.array([phi1,phi2]).T

@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0,1],[0,1])
def phi12_to_lb(phi1,phi2,degree=False):
    """
    NAME:
       phi12_to_lb
    PURPOSE:
       Transform (phi1,phi2) to Galactic coordinates (l,b)
    INPUT:
       phi1 - phi longitude (rad or degree)
       phi2 - phi latitude (rad or degree)
       degree= (False) if True, input and output are in degrees
    OUTPUT:
       (l,b) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    #Convert phi1,phi2 to l,b coordinates
    phiXYZ= numpy.array([numpy.cos(phi2)*numpy.cos(phi1),
                         numpy.cos(phi2)*numpy.sin(phi1),
                         numpy.sin(phi2)])
    eqXYZ= numpy.dot(_TKOP.T,phiXYZ)
    #Get ra dec
    dec= numpy.arcsin(eqXYZ[2])
    ra= numpy.arctan2(eqXYZ[1],eqXYZ[0])
    ra[ra<0.]+= 2.*numpy.pi
    #Now convert to l,b
    return bovy_coords.radec_to_lb(ra,dec)

@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([2,3],[])
def pmllpmbb_to_pmphi12(pmll,pmbb,l,b,degree=False):
    """
    NAME:
       pmllpmbb_to_pmphi12
    PURPOSE:
       Transform proper motions in Galactic coordinates (l,b) to (phi1,phi2)
    INPUT:
       pmll - proper motion Galactic longitude (rad or degree); contains xcosb
       pmbb - Galactic latitude (rad or degree)
       l - Galactic longitude (rad or degree)
       b - Galactic latitude (rad or degree)
       degree= (False) if True, input (l,b) are in degrees
    OUTPUT:
       (pmphi1,pmphi2) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    #First go to ra and dec
    radec= bovy_coords.lb_to_radec(l,b)
    ra= radec[:,0]
    dec= radec[:,1]
    pmradec= bovy_coords.pmllpmbb_to_pmrapmdec(pmll,pmbb,l,b,degree=False)
    pmra= pmradec[:,0]
    pmdec= pmradec[:,1]
    #Now transform ra,dec pm to phi1,phi2
    phi12= lb_to_phi12(l,b,degree=False)
    phi1= phi12[:,0]
    phi2= phi12[:,1]
    #Build A and Aphi matrices
    A= numpy.zeros((3,3,len(ra)))
    A[0,0]= numpy.cos(ra)*numpy.cos(dec)
    A[0,1]= -numpy.sin(ra)
    A[0,2]= -numpy.cos(ra)*numpy.sin(dec)
    A[1,0]= numpy.sin(ra)*numpy.cos(dec)
    A[1,1]= numpy.cos(ra)
    A[1,2]= -numpy.sin(ra)*numpy.sin(dec)
    A[2,0]= numpy.sin(dec)
    A[2,1]= 0.
    A[2,2]= numpy.cos(dec)
    AphiInv= numpy.zeros((3,3,len(ra)))
    AphiInv[0,0]= numpy.cos(phi1)*numpy.cos(phi2)
    AphiInv[0,1]= numpy.cos(phi2)*numpy.sin(phi1)
    AphiInv[0,2]= numpy.sin(phi2)
    AphiInv[1,0]= -numpy.sin(phi1)
    AphiInv[1,1]= numpy.cos(phi1)
    AphiInv[1,2]= 0.
    AphiInv[2,0]= -numpy.cos(phi1)*numpy.sin(phi2)
    AphiInv[2,1]= -numpy.sin(phi1)*numpy.sin(phi2)
    AphiInv[2,2]= numpy.cos(phi2)
    TA= numpy.dot(_TKOP,numpy.swapaxes(A,0,1))
    #Got lazy...
    trans= numpy.zeros((2,2,len(ra)))
    for ii in range(len(ra)):
        trans[:,:,ii]= numpy.dot(AphiInv[:,:,ii],TA[:,:,ii])[1:,1:]
    return (trans*numpy.array([[pmra,pmdec],[pmra,pmdec]])).sum(1).T

@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([2,3],[])
def pmphi12_to_pmllpmbb(pmphi1,pmphi2,phi1,phi2,degree=False):
    """
    NAME:
       pmllpmbb_to_pmphi12
    PURPOSE:
       Transform proper motions in (phi1,phi2) to Galactic coordinates (l,b)
    INPUT:
       pmphi1 - proper motion Galactic longitude (rad or degree); contains xcosphi2
       pmphi2 - Galactic latitude (rad or degree)
       phi1 - phi longitude (rad or degree)
       phi2 - phi latitude (rad or degree)
       degree= (False) if True, input (phi1,phi2) are in degrees
    OUTPUT:
       (pmll,pmbb) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    #First go from phi12 to ra and dec
    lb= phi12_to_lb(phi1,phi2)
    radec= bovy_coords.lb_to_radec(lb[:,0],lb[:,1])
    ra= radec[:,0]
    dec= radec[:,1]
    #Build A and Aphi matrices
    AInv= numpy.zeros((3,3,len(ra)))
    AInv[0,0]= numpy.cos(ra)*numpy.cos(dec)
    AInv[0,1]= numpy.sin(ra)*numpy.cos(dec)
    AInv[0,2]= numpy.sin(dec)
    AInv[1,0]= -numpy.sin(ra)
    AInv[1,1]= numpy.cos(ra)
    AInv[1,2]= 0.
    AInv[2,0]= -numpy.cos(ra)*numpy.sin(dec)
    AInv[2,1]= -numpy.sin(ra)*numpy.sin(dec)
    AInv[2,2]= numpy.cos(dec)
    Aphi= numpy.zeros((3,3,len(ra)))
    Aphi[0,0]= numpy.cos(phi1)*numpy.cos(phi2)
    Aphi[0,1]= -numpy.sin(phi1)
    Aphi[0,2]= -numpy.cos(phi1)*numpy.sin(phi2)
    Aphi[1,0]= numpy.sin(phi1)*numpy.cos(phi2)
    Aphi[1,1]= numpy.cos(phi1)
    Aphi[1,2]= -numpy.sin(phi1)*numpy.sin(phi2)
    Aphi[2,0]= numpy.sin(phi2)
    Aphi[2,1]= 0.
    Aphi[2,2]= numpy.cos(phi2)
    TAphi= numpy.dot(_TKOP.T,numpy.swapaxes(Aphi,0,1))
    #Got lazy...
    trans= numpy.zeros((2,2,len(ra)))
    for ii in range(len(ra)):
        trans[:,:,ii]= numpy.dot(AInv[:,:,ii],TAphi[:,:,ii])[1:,1:]
    pmradec= (trans*numpy.array([[pmphi1,pmphi2],[pmphi1,pmphi2]])).sum(1).T
    pmra= pmradec[:,0]
    pmdec= pmradec[:,1]
    #Now convert to pmll
    return bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec)

def convert_track_lb_to_phi12(track):
    """track = _interpolatedObsTrackLB"""
    phi12= lb_to_phi12(track[:,0],track[:,1],degree=True)
    phi12[phi12[:,0] > 180.,0]-= 360.
    pmphi12= pmllpmbb_to_pmphi12(track[:,4],track[:,5],
                                 track[:,0],track[:,1],
                                 degree=True)
    out= numpy.empty_like(track)
    out[:,:2]= phi12
    out[:,2]= track[:,2]
    out[:,3]= track[:,3]
    out[:,4:]= pmphi12
    return out

def phi12_to_lb_6d(phi1,phi2,dist,pmphi1,pmphi2,vlos):
    l,b= phi12_to_lb(phi1,phi2,degree=True)
    pmll, pmbb= pmphi12_to_pmllpmbb(pmphi1,pmphi2,phi1,phi2,degree=True)
    return [l,b,dist,pmll,pmbb,vlos]

def timeout_handler(signum, frame):
    raise Exception("Calculation timed-out")

def predict_gd1obs(pot_params,c,b=1.,pa=0.,
                   sigv=0.4,td=10.,
                   phi1=0.,phi2=-1.,dist=10.2,
                   pmphi1=-8.5,pmphi2=-2.05,vlos=-285.,
                   ro=_REFR0,vo=_REFV0,
                   isob=None,nTrackChunks=8,multi=None,
                   useTM=False,
                   logpot=False,
                   verbose=True):
    """
    NAME:
       predict_gd1obs
    PURPOSE:
       Function that generates the location and velocity of the GD-1 stream (similar to predict_pal5obs), its width, and its length for a given potential and progenitor phase-space position
    INPUT:
       pot_params- array with the parameters of a potential model (see MWPotential2014Likelihood.setup_potential; only the basic parameters of the disk and halo are used, flattening is specified separately)
       c- halo flattening
       b= (1.) halo-squashed
       pa= (0.) halo PA
       sigv= (0.365) velocity dispersion in km/s
       td= (10.) stream age in Gyr
       phi1= (0.) Progenitor's phi1 position
       phi2= (-1.) Progenitor's phi2 position
       dist= (10.2) progenitor distance in kpc
       pmphi1= (-8.5) progenitor proper motion in phi1 in mas/yr
       pmphi2= (-2.) progenitor proper motion in phi2 in mas/yr
       vlos= (-250.) progenitor line-of-sight velocity in km/s
       ro= (project default) distance to the GC in kpc
       vo= (project default) circular velocity at R0 in km/s
       nTrackChunks= (8) nTrackChunks input to streamdf
       multi= (None) multi input to streamdf
       logpot= (False) if True, use a logarithmic potential instead
       isob= (None) if given, b parameter of actionAngleIsochroneApprox
       useTM= (True) use the TorusMapper to compute the track
       verbose= (True) print messages
    OUTPUT:
    HISTORY:
       2016-08-12 - Written - Bovy (UofT)
    """
    # Convert progenitor's phase-space position to l,b,...
    prog= Orbit(phi12_to_lb_6d(phi1,phi2,dist,pmphi1,pmphi2,vlos),
                lb=True,ro=ro,vo=vo,solarmotion=[-11.1,24.,7.25])
    if logpot:
        pot= potential.LogarithmicHaloPotential(normalize=1.,q=c)
    else:
        pot= MWPotential2014Likelihood.setup_potential(pot_params,c,
                                                       False,False,ro,vo,
                                                       b=b,pa=pa)
    success= True
    this_useTM= useTM
    try:
        sdf= setup_sdf(pot,prog,sigv,td,ro,vo,multi=multi,
                       isob=isob,nTrackChunks=nTrackChunks,
                       verbose=verbose,useTM=useTM,logpot=logpot)
    except:
        # Catches errors and time-outs
        success= False
    # Check for calc. issues
    if not success:
        return (numpy.zeros((1001,6))-1000000.,False)
        # Try again with TM
        this_useTM= True
        this_nTrackChunks= 21 # might as well
        sdf= setup_sdf(pot,prog,sigv,td,ro,vo,multi=multi,
                       isob=isob,nTrackChunks=this_nTrackChunks,
                       verbose=verbose,useTM=this_useTM,logpot=logpot)
    else:
        success= not this_useTM
    # Compute the track and convert it to phi12
    track_phi= convert_track_lb_to_phi12(sdf._interpolatedObsTrackLB)
    return (track_phi,success)

def gd1_lnlike(posdata,distdata,pmdata,rvdata,
               track_phi):
    """
    Returns array [5] with log likelihood for each b) data set (phi2,dist,pmphi1,pmphi2,vlos)
    """
    out= numpy.zeros(5)-1000000000000000.
    if numpy.any(numpy.isnan(track_phi)):
        return out
    # Interpolate all tracks to evaluate the likelihood
    sindx= numpy.argsort(track_phi[:,0])
    # phi2
    ipphi2= \
        interpolate.InterpolatedUnivariateSpline(\
        track_phi[sindx,0],track_phi[sindx,1],k=1) # to be on the safe side
    out[0]= \
        -0.5*numpy.sum((ipphi2(posdata[:,0])-posdata[:,1])**2.\
                           /posdata[:,2]**2.)
    # dist
    ipdist= \
        interpolate.InterpolatedUnivariateSpline(\
        track_phi[sindx,0],track_phi[sindx,2],k=1) # to be on the safe side
    out[1]= \
        -0.5*numpy.sum((ipdist(distdata[:,0])-distdata[:,1])**2.\
                           /distdata[:,2]**2.)
    # pmphi1
    ippmphi1= \
        interpolate.InterpolatedUnivariateSpline(\
        track_phi[sindx,0],track_phi[sindx,4],k=1) # to be on the safe side
    out[2]= \
        -0.5*numpy.sum((ippmphi1(pmdata[:,0])-pmdata[:,1])**2.\
                           /pmdata[:,3]**2.)
    # pmphi2
    ippmphi2= \
        interpolate.InterpolatedUnivariateSpline(\
        track_phi[sindx,0],track_phi[sindx,5],k=1) # to be on the safe side
    out[3]= \
        -0.5*numpy.sum((ippmphi2(pmdata[:,0])-pmdata[:,2])**2.\
                           /pmdata[:,3]**2.)
    # vlos
    ipvlos= \
        interpolate.InterpolatedUnivariateSpline(\
        track_phi[sindx,0],track_phi[sindx,3],k=1) # to be on the safe side
    out[4]= \
        -0.5*numpy.sum((ipvlos(rvdata[:,0])-rvdata[:,2])**2.\
                           /rvdata[:,3]**2.)
    return out

def setup_sdf(pot,prog,sigv,td,ro,vo,multi=None,nTrackChunks=8,isob=None,
              trailing_only=False,verbose=True,useTM=True,logpot=False):
    """Simple function to setup the stream model"""
    if isob is None:
        if True or logpot:
            isob= 0.75
    if False:
        # Determine good one
        ts= numpy.linspace(0.,15.,1001)
        # Hack!
        epot= copy.deepcopy(pot)
        epot[2]._b= 1.
        epot[2]._b2= 1.
        epot[2]._isNonAxi= False
        epot[2]._aligned= True
        prog.integrate(ts,pot)
        estb= estimateBIsochrone(epot,
                                 prog.R(ts,use_physical=False),
                                 prog.z(ts,use_physical=False),
                                 phi=prog.phi(ts,use_physical=False))
        if estb[1] < 0.3: isob= 0.3
        elif estb[1] > 1.5: isob= 1.5
        else: isob= estb[1]
        if verbose: print(pot[2]._c, isob,estb)
    if not logpot and numpy.fabs(pot[2]._b-1.) > 0.05:
        aAI= actionAngleIsochroneApprox(pot=pot,b=isob,tintJ=1000.,
                                        ntintJ=30000)
    else:
        ts= numpy.linspace(0.,100.,10000)
        aAI= actionAngleIsochroneApprox(pot=pot,b=isob,tintJ=100.,
                                        ntintJ=10000,dt=ts[1]-ts[0])
    if useTM:
        aAT= actionAngleTorus(pot=pot,tol=0.001,dJ=0.0001)
    else:
        aAT= False
    try:
        sdf=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=True,nTrackChunks=nTrackChunks,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TKOP,
                     multi=multi,
                     nospreadsetup=True)
    except numpy.linalg.LinAlgError:
        sdf=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=True,nTrackChunks=nTrackChunks,
                     nTrackIterations=0,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TKOP,
                     multi=multi)
    return sdf

def gd1_data():
    #First the RV data from Table 1: phi1, phi2, Vrad, Vraderr
    rvdata= numpy.array([[-45.23,-0.04,28.8,6.9],
                         [-43.17,-0.09,29.3,10.2],
                         [-39.54,-0.07,2.9,8.7],
                         [-39.25,-0.22,-5.2,6.5],
                         [-37.95,0.00 ,1.1,5.6],
                         [-37.96,-0.00,-11.7,11.2],
                         [-35.49,-0.05,-50.4,5.2],
                         [-35.27,-0.02,-30.9,12.8],
                         [-34.92,-0.15,-35.3,7.5],
                         [-34.74,-0.08,-30.9,9.2],
                         [-33.74,-0.18,-74.3,9.8],
                         [-32.90,-0.15,-71.5,9.6],
                         [-32.25,-0.17,-71.5,9.2],
                         [-29.95,0.00,-92.7,8.7],
                         [-26.61,-0.11,-114.2,7.3],
                         [-25.45,-0.14,-67.8,7.1],
                         [-24.86,0.01 ,-111.2,17.8],
                         [-21.21,-0.02,-144.4,10.5],
                         [-14.47,-0.15,-179.0,10.0],
                         [-13.73,-0.28,-191.4,7.5],
                         [-13.02,-0.21,-162.9,9.6],
                         [-12.68,-0.26,-217.2,10.7],
                         [-12.55,-0.23,-172.2,6.6],
                         [-56.,-9999.99,122.,14.],#These are the SEGUE averages
                         [-47.,-9999.99,44.8,10.],#converted assuming (U,V,W)sun
                         [-28.,-9999.99,-77.8,6.],#10.,225.25,7.17
                         [-24.,-9999.99,-114.5,9.]])
    #Position data from Table 2: phi1, phi2, phi2err
    posdata= numpy.array([[-60.00,-0.64,0.15],
                          [-56.00,-0.89,0.27],
                          [-54.00,-0.45,0.15],
                          [-48.00,-0.08,0.13],
                          [-44.00,0.01,0.14],
                          [-40.00,-0.00,0.09],
                          [-36.00,0.04,0.10],
                          [-34.00,0.06,0.13],
                          [-32.00,0.04,0.06],
                          [-30.00,0.08,0.10],
                          [-28.00,0.03,0.12],
                          [-24.00,0.06,0.05],
                          [-22.00,0.06,0.13],
                          [-18.00,-0.05,0.11],
                          [-12.00,-0.29,0.16],
                          [-2.00 ,-0.87,0.07]])
    #Distance data from Table 3: phi1, Distance, Distance_err
    distdata= numpy.array([[-55.00,7.20,0.30],
                           [-45.00,7.59,0.40],
                           [-35.00,7.83,0.30],
                           [-25.00,8.69,0.40],
                           [-15.00,8.91,0.40],
                           [0.00,9.86,0.50]])
    #Proper motion data from Table 4: phi1, pmphi1, pmphi2, pmerr
    pmdata= numpy.array([[-55.00,-13.60 ,-5.70,1.30],
                         [-45.00,-13.10 ,-3.30,0.70],
                         [-35.00,-12.20 ,-3.10,1.00],
                         [-25.00,-12.60 ,-2.70,1.40],
                         [-15.00,-10.80 ,-2.80,1.00]])
    return (posdata,distdata,pmdata,rvdata)
