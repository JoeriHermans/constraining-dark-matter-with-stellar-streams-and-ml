import numpy
from scipy import integrate
import astropy.units as u
from galpy import potential
from galpy.util import bovy_plot, bovy_conversion, bovy_coords
from matplotlib import pyplot
_REFR0, _REFV0= 8., 220.
def like_func(params,c,surfrs,kzs,kzerrs,termdata,termsigma,fitc,fitvoro,
              dblexp,addpal5,addgd1,ro,vo,addgas):
    #Check ranges
    if params[0] < 0. or params[0] > 1.: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[1] < 0. or params[1] > 1.: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if (1.-params[0]-params[1]) < 0. or (1.-params[0]-params[1]) > 1.: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[2] < numpy.log(1./_REFR0) or params[2] > numpy.log(8./_REFR0):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[3] < numpy.log(0.05/_REFR0) or params[3] > numpy.log(1./_REFR0):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    if fitvoro and (params[7] <= 150./_REFV0 or params[7] > 290./_REFV0):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    if fitvoro and (params[8] <= 7./_REFR0 or params[8] > 9.4/_REFR0):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    if fitc and (params[7+2*fitvoro] <= 0. or params[7+2*fitvoro] > 4.):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    if fitvoro:
        ro, vo= _REFR0*params[8], _REFV0*params[7]
    #Setup potential
    pot= setup_potential(params,c,fitc,dblexp,ro,vo,fitvoro=fitvoro,
                         addgas=addgas)
    #Calculate model surface density at surfrs
    modelkzs= numpy.empty_like(surfrs)
    for ii in range(len(surfrs)):
        modelkzs[ii]= -potential.evaluatezforces(pot,
                                                 (ro-8.+surfrs[ii])/ro,
                                                 1.1/ro,
                                                 phi=0.)*bovy_conversion.force_in_2piGmsolpc2(vo,ro)
    out= 0.5*numpy.sum((kzs-modelkzs)**2./kzerrs**2.)
    #Add terminal velocities
    vrsun= params[5]
    vtsun= params[6]
    cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr= termdata
    #Calculate terminal velocities at data glon
    cl_vterm_model= numpy.zeros_like(cl_vterm)
    for ii in range(len(cl_glon)):
        cl_vterm_model[ii]= potential.vterm(pot,cl_glon[ii])
    cl_vterm_model+= vrsun*numpy.cos(cl_glon/180.*numpy.pi)\
        -vtsun*numpy.sin(cl_glon/180.*numpy.pi)
    mc_vterm_model= numpy.zeros_like(mc_vterm)
    for ii in range(len(mc_glon)):
        mc_vterm_model[ii]= potential.vterm(pot,mc_glon[ii])
    mc_vterm_model+= vrsun*numpy.cos(mc_glon/180.*numpy.pi)\
        -vtsun*numpy.sin(mc_glon/180.*numpy.pi)
    cl_dvterm= (cl_vterm-cl_vterm_model*vo)/termsigma
    mc_dvterm= (mc_vterm-mc_vterm_model*vo)/termsigma
    out+= 0.5*numpy.sum(cl_dvterm*numpy.dot(cl_corr,cl_dvterm))
    out+= 0.5*numpy.sum(mc_dvterm*numpy.dot(mc_corr,mc_dvterm))
    #Rotation curve constraint
    out-= logprior_dlnvcdlnr(potential.dvcircdR(pot,1.,phi=0.))
    #K dwarfs, Kz
    out+= 0.5*(-potential.evaluatezforces(pot,1.,1.1/ro,phi=0.)*bovy_conversion.force_in_2piGmsolpc2(vo,ro)-67.)**2./36.
    #K dwarfs, visible
    out+= 0.5*(visible_dens(pot,ro,vo)-55.)**2./25.
    #Local density prior
    localdens= potential.evaluateDensities(pot,1.,0.,phi=0.)*bovy_conversion.dens_in_msolpc3(vo,ro)
    out+= 0.5*(localdens-0.102)**2./0.01**2.
    #Bulge velocity dispersion
    out+= 0.5*(bulge_dispersion(pot,ro,vo)-117.)**2./225.
    #Mass at 60 kpc
    out+= 0.5*(mass60(pot,ro,vo)-4.)**2./0.7**2.
    #Pal5
    if addpal5:
        # q = 0.94 +/- 0.05 + add'l
        fp5= force_pal5(pot,23.46,ro,vo)
        out+= 0.5*(numpy.sqrt(2.*fp5[0]/fp5[1])-0.94)**2./0.05**2.
        out+= 0.5*(0.94**2.*(fp5[0]+0.8)+2.*(fp5[1]+1.82)+0.2)**2./0.6**2.
    #GD-1
    if addgd1:
        # q = 0.95 +/- 0.04 + add'l
        fg1= force_gd1(pot,ro,vo)
        out+= 0.5*(numpy.sqrt(6.675/12.5*fg1[0]/fg1[1])-0.95)**2./0.04**2.
        out+= 0.5*(0.95**2.*(fg1[0]+2.51)+6.675/12.5*(fg1[1]+1.47)+0.05)**2./0.3**2.
    # vc and ro measurements: vc=218 +/- 10 km/s, ro= 8.1 +/- 0.1 kpc
    out+= (vo-218.)**2./200.+(ro-8.1)**2./0.02
    if numpy.isnan(out): 
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        return out

def pdf_func(params,*args):
    return -like_func(params,*args)

def setup_potential(params,c,fitc,dblexp,ro,vo,fitvoro=False,b=1.,pa=0.,
                    addgas=False):
    pot= [potential.PowerSphericalPotentialwCutoff(normalize=1.-params[0]-params[1],
                                                   alpha=1.8,rc=1.9/ro)]
    if dblexp:
        if addgas:
            # add 13 Msun/pc^2 
            gp= potential.DoubleExponentialDiskPotential(\
                amp=0.03333*u.Msun/u.pc**3\
                    *numpy.exp(ro/2./numpy.exp(params[2])/_REFR0),
                hz=150.*u.pc,
                hr=2.*numpy.exp(params[2])*_REFR0/ro,
                ro=ro,vo=vo)
            gp.turn_physical_off()
            gprf= gp.Rforce(1.,0.)
            dpf= params[0]+gprf
            if dpf < 0.: dpf= 0.
            pot.append(\
                potential.DoubleExponentialDiskPotential(\
                    normalize=dpf,hr=numpy.exp(params[2])*_REFR0/ro,
                    hz=numpy.exp(params[3])*_REFR0/ro))
        else:
            pot.append(\
                potential.DoubleExponentialDiskPotential(\
                    normalize=params[0],hr=numpy.exp(params[2])*_REFR0/ro,
                    hz=numpy.exp(params[3])*_REFR0/ro))
    else:
        pot.append(\
            potential.MiyamotoNagaiPotential(normalize=params[0],
                                             a=numpy.exp(params[2])*_REFR0/ro,
                                             b=numpy.exp(params[3])*_REFR0/ro))
    if fitc:
        pot.append(potential.TriaxialNFWPotential(\
                normalize=params[1],a=numpy.exp(params[4])*_REFR0/ro,
                c=params[7+2*fitvoro],b=b,pa=pa))
    else:
        pot.append(potential.TriaxialNFWPotential(\
                normalize=params[1],a=numpy.exp(params[4])*_REFR0/ro,
                c=c,b=b,pa=pa))
    if addgas:
        pot.append(gp) # make sure it's the last
    return pot

def force_pal5(pot,dpal5,ro,vo):
    """Return the force at Pal5"""
    # First compute the location based on the distance
    l5, b5= bovy_coords.radec_to_lb(229.018,-0.124,degree=True)
    X5,Y5,Z5= bovy_coords.lbd_to_XYZ(l5,b5,dpal5,degree=True)
    R5,p5,Z5= bovy_coords.XYZ_to_galcencyl(X5,Y5,Z5,Xsun=ro,Zsun=0.025)
    return (potential.evaluateRforces(pot,R5/ro,Z5/ro,phi=p5,
                                      use_physical=True,ro=ro,vo=vo),
            potential.evaluatezforces(pot,R5/ro,Z5/ro,phi=p5,
                                      use_physical=True,ro=ro,vo=vo),
            potential.evaluatephiforces(pot,R5/ro,Z5/ro,phi=p5,
                                        use_physical=True,ro=ro,vo=vo))

def force_gd1(pot,ro,vo):
    """Return the force at GD-1"""
    # Just use R=12.5 kpc, Z= 6.675 kpc, phi=0
    R1= 12.5
    Z1= 6.675
    p1= 0.
    return (potential.evaluateRforces(pot,R1/ro,Z1/ro,phi=p1,
                                      use_physical=True,ro=ro,vo=vo),
            potential.evaluatezforces(pot,R1/ro,Z1/ro,phi=p1,
                                      use_physical=True,ro=ro,vo=vo),
            potential.evaluatephiforces(pot,R1/ro,Z1/ro,phi=p1,
                                        use_physical=True,ro=ro,vo=vo))

def mass60(pot,_REFR0,_REFV0):
    """The mass at 60 kpc in 10^11 msolar"""
    tR= 60./_REFR0
    # Average r^2 FR/G
    return -integrate.quad(lambda x: tR**2.*potential.evaluaterforces(pot,tR*x,tR*numpy.sqrt(1.-x**2.),phi=0.),
                           0.,1.)[0]\
                           *bovy_conversion.mass_in_1010msol(_REFV0,_REFR0)/10.

def bulge_dispersion(pot,_REFR0,_REFV0):
    """The expected dispersion in Baade's window, in km/s"""
    bar, baz= 0.0175, 0.068
    return numpy.sqrt(1./pot[0].dens(bar,baz)*integrate.quad(lambda x: -potential.evaluatezforces(pot,bar,x,phi=0.)*pot[0].dens(bar,x),baz,numpy.inf)[0])*_REFV0

def visible_dens(pot,_REFR0,_REFV0,r=1.):
    """The visible surface density at 8 kpc from the center"""
    if len(pot) == 4:
        return 2.*(integrate.quad((lambda zz: potential.evaluateDensities(pot[1],r,zz,phi=0.)),0.,2.)[0]+integrate.quad((lambda zz: potential.evaluateDensities(pot[3],r,zz,phi=0.)),0.,2.)[0])*bovy_conversion.surfdens_in_msolpc2(_REFV0,_REFR0)
    else:
        return 2.*integrate.quad((lambda zz: potential.evaluateDensities(pot[1],r,zz,phi=0.)),0.,2.)[0]*bovy_conversion.surfdens_in_msolpc2(_REFV0,_REFR0)

def logprior_dlnvcdlnr(dlnvcdlnr):
    sb= 0.04
    if dlnvcdlnr > sb or dlnvcdlnr < -0.5:
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    return numpy.log((sb-dlnvcdlnr)/sb)-(sb-dlnvcdlnr)/sb

#########################################PLOTS#################################
def plotRotcurve(pot):
    potential.plotRotcurve(pot,xrange=[0.,4.],color='k',lw=2.,yrange=[0.,1.4],
                           gcf=True)
    #Constituents
    line1= potential.plotRotcurve(pot[0],overplot=True,color='k',ls='-.',lw=2.)
    line2= potential.plotRotcurve(pot[1],overplot=True,color='k',ls='--',lw=2.)
    line3= potential.plotRotcurve(pot[2],overplot=True,color='k',ls=':',lw=2.)
    #Add legend
    pyplot.legend((line1[0],line2[0],line3[0]),
                  (r'$\mathrm{Bulge}$',
                   r'$\mathrm{Disk}$',
                   r'$\mathrm{Halo}$'),
                  loc='upper right',#bbox_to_anchor=(.91,.375),
                  numpoints=8,
                  prop={'size':16},
                  frameon=False)
    return None

def plotKz(pot,surfrs,kzs,kzerrs,_REFR0,_REFV0):
    krs= numpy.linspace(4./_REFR0,10./_REFR0,1001)
    modelkz= numpy.array([-potential.evaluatezforces(pot,kr,1.1/_REFR0)\
                               *bovy_conversion.force_in_2piGmsolpc2(_REFV0,_REFR0) for kr in krs])
    bovy_plot.bovy_plot(krs*_REFR0,modelkz,'-',color='0.6',lw=2.,
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$F_{Z}(R,|Z| = 1.1\,\mathrm{kpc})\ (2\pi G\,M_\odot\,\mathrm{pc}^{-2})$',
                        semilogy=True,
                        yrange=[10.,1000.],
                        xrange=[4.,10.],
                        zorder=0,gcf=True)
    pyplot.errorbar(_REFR0-8.+surfrs,
                    kzs,
                    yerr=kzerrs,
                    marker='o',
                    elinewidth=1.,capsize=3,zorder=1,
                    color='k',linestyle='none')  
    pyplot.errorbar([_REFR0],[69.],yerr=[6.],marker='d',ms=10.,
                    elinewidth=1.,capsize=3,zorder=10,
                    color='0.4',linestyle='none')
    #Do an exponential fit to the model Kz and return the scale length
    indx= krs < 9./_REFR0
    p= numpy.polyfit(krs[indx],numpy.log(modelkz[indx]),1)
    return -1./p[0]

def plotTerm(pot,termdata,_REFR0,_REFV0):
    mglons= numpy.linspace(-90.,-20.,1001)
    pglons= numpy.linspace(20.,90.,1001)
    mterms= numpy.array([potential.vterm(pot,mgl)*_REFV0 for mgl in mglons])
    pterms= numpy.array([potential.vterm(pot,pgl)*_REFV0 for pgl in pglons])
    bovy_plot.bovy_plot(mglons,mterms,'-',color='0.6',lw=2.,zorder=0,
                        xlabel=r'$\mathrm{Galactic\ longitude\, (deg)}$',
                        ylabel=r'$\mathrm{Terminal\ velocity}\, (\mathrm{km\,s}^{-1})$',
                        xrange=[-100.,100.],
                        yrange=[-150.,150.],
                        gcf=True)
    bovy_plot.bovy_plot(pglons,pterms,'-',color='0.6',lw=2.,zorder=0,
                        overplot=True)
    cl_glon,cl_vterm,cl_corr,mc_glon,mc_vterm,mc_corr= termdata
    bovy_plot.bovy_plot(cl_glon,cl_vterm,'ko',overplot=True)
    bovy_plot.bovy_plot(mc_glon-360.,mc_vterm,'ko',overplot=True)
    return None

def plotPot(pot):
    potential.plotPotentials(pot,rmin=0.,rmax=1.5,nrs=201,
                             zmin=-0.5,zmax=0.5,nzs=201,ncontours=21,
                             justcontours=True,gcf=True)
    return None
    
def plotDens(pot):
    potential.plotDensities(pot,rmin=0.01,rmax=1.5,nrs=201,
                            zmin=-0.5,zmax=0.5,nzs=201,ncontours=21,
                            log=True,justcontours=True,gcf=True)
    return None
    
def readClemens(dsinl=0.5/8.):
    data= numpy.loadtxt('../mwpot14data/clemens1985_table2.dat',delimiter='|',
                        comments='#')
    glon= data[:,0]
    vterm= data[:,1]
    #Remove l < 40 and l > 80
    indx= (glon > 40.)*(glon < 80.)
    glon= glon[indx]
    vterm= vterm[indx]
    if bin:
        #Bin in l=1 bins
        glon, vterm= binlbins(glon,vterm,dl=1.)
        #Remove nan, because 1 bin is empty
        indx= True-numpy.isnan(glon)
        glon= glon[indx]
        vterm= vterm[indx]
    #Calculate correlation matrix
    singlon= numpy.sin(glon/180.*numpy.pi)
    corr= calc_corr(singlon,dsinl)
    return (glon,vterm,numpy.linalg.inv(corr))

def readMcClureGriffiths(dsinl=0.5/8.,bin=True):
    data= numpy.loadtxt('../mwpot14data/McClureGriffiths2007.dat',
                        comments='#')
    glon= data[:,0]
    vterm= data[:,1]
    #Remove l > 320 and l > 80
    indx= (glon < 320.)*(glon > 280.)
    glon= glon[indx]
    vterm= vterm[indx]
    if bin:
        #Bin in l=1 bins
        glon, vterm= binlbins(glon,vterm,dl=1.)
    #Calculate correlation matrix
    singlon= numpy.sin(glon/180.*numpy.pi)
    corr= calc_corr(singlon,dsinl)
    return (glon,vterm,numpy.linalg.inv(corr))
    
def readMcClureGriffiths16(dsinl=0.5/8.,bin=True):
    data= numpy.loadtxt('../mwpot14data/McClureGriffiths2016.dat',
                        comments='#',delimiter='&')
    glon= data[:,0]
    vterm= data[:,1]
    #Remove l < 30 and l > 80
    indx= (glon > 40.)*(glon < 80.)
    glon= glon[indx]
    vterm= vterm[indx]
    if bin:
        #Bin in l=1 bins
        glon, vterm= binlbins(glon,vterm,dl=1.)
    #Calculate correlation matrix
    singlon= numpy.sin(glon/180.*numpy.pi)
    corr= calc_corr(singlon,dsinl)
    return (glon,vterm,numpy.linalg.inv(corr))
    
def calc_corr(singlon,dsinl):
    #Calculate correlation matrix
    corr= numpy.zeros((len(singlon),len(singlon)))
    for ii in range(len(singlon)):
        for jj in range(len(singlon)):
            corr[ii,jj]= numpy.exp(-numpy.fabs(singlon[ii]-singlon[jj])/dsinl)
    corr= 0.5*(corr+corr.T)
    return corr+10.**-10.*numpy.eye(len(singlon)) #for stability

def binlbins(glon,vterm,dl=1.):
    minglon, maxglon= numpy.floor(numpy.amin(glon)), numpy.floor(numpy.amax(glon))
    minglon, maxglon= int(minglon), int(maxglon)
    nout= maxglon-minglon+1
    glon_out= numpy.zeros(nout)
    vterm_out= numpy.zeros(nout)
    for ii in range(nout):
        indx= (glon > minglon+ii)*(glon < minglon+ii+1)
        glon_out[ii]= numpy.mean(glon[indx])
        vterm_out[ii]= numpy.mean(vterm[indx])
    return (glon_out,vterm_out)
