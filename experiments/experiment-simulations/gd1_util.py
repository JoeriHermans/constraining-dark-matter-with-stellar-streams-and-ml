from galpy.df import streamdf, streamgapdf
from streampepperdf_new import streampepperdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.util import bovy_conversion #for unit conversions
from gd1_util_MWhaloshape import phi12_to_lb_6d

R0, V0= 8., 220.
def setup_gd1model(leading=True,pot=MWPotential2014,
                   timpact=None,
                   hernquist=True,new_orb_lb=None,isob=0.8,
                   age=9.,sigv=0.5,
                   singleImpact=False,
                   length_factor=1.,
                   **kwargs):
    #lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=pot,b=isob)
    #obs= Orbit([1.56148083,0.35081535,-1.15481504,0.88719443,
    #            -0.47713334,0.12019596])
    #progenitor pos and vel from Bovy 1609.01298 and with corrected proper motion
    
    if new_orb_lb is None :
        obs=Orbit(phi12_to_lb_6d(0,-0.82,10.1,-8.5,-2.15,-257.),lb=True,solarmotion=[-11.1,24.,7.25],ro=8.,vo=220.)
        #sigv= 0.365/2.*(9./age) #km/s, /2 bc tdis x2, adjust for diff. age
        
    else:
        obs=Orbit(new_orb_lb,lb=True,solarmotion=[-11.1,24.,7.25],ro=8.,vo=220.)
    
    #
    if timpact is None:
        sdf= streamdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,leading=leading,
                      nTrackChunks=11,vsun=[-11.1,244.,7.25],
                      tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                      vo=V0,ro=R0)
    elif singleImpact:
        sdf= streamgapdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,
                         leading=leading,
                         nTrackChunks=11,vsun=[-11.1,244.,7.25],
                         tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                         vo=V0,ro=R0,
                         timpact=timpact,
                         spline_order=3,
                         hernquist=hernquist,**kwargs)
    else:
        sdf= streampepperdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,
                            leading=leading,
                            nTrackChunks=101,vsun=[-11.1,244.,7.25],
                            tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                            vo=V0,ro=R0,
                            timpact=timpact,
                            spline_order=1,
                            hernquist=hernquist,
                            length_factor=length_factor)
    sdf.turn_physical_off()  #original
    #obs.turn_physical_off()
    return sdf
