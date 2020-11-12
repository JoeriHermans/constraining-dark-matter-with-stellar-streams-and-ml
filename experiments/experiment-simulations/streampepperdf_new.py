# The DF of a tidal stream peppered with impacts
import copy
import hashlib
import numpy
from scipy import integrate, special, stats, optimize, interpolate, signal
from galpy.df import streamdf, streamgapdf
from galpy.df.streamgapdf import _rotation_vy
from galpy.util import bovy_conversion, bovy_coords
class streampepperdf(streamdf):
    """The DF of a tidal stream peppered with impacts"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the DF of a stellar stream peppered with impacts

        INPUT:

           streamdf args and kwargs

           Subhalo and impact parameters, for all impacts:

              timpact= time since impact ([nimpact]); needs to be set

              impact_angle= angle offset from progenitor at which the impact occurred (at the impact time; in rad) ([nimpact]); optional

              impactb= impact parameter ([nimpact]); optional

              subhalovel= velocity of the subhalo shape=(nimpact,3); optional

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed); all kicks need to chose the same option; optional keywords

                 1( GM= mass of the subhalo ([nimpact])

                    rs= size parameter of the subhalo ([nimpact])

                 2( subhalopot= galpy potential object or list thereof (should be spherical); list of len nimpact (if the individual potentials are lists, need to give a list of lists)

                 3( hernquist= (False) if True, use Hernquist kicks for GM/rs

           deltaAngleTrackImpact= (None) angle to estimate the stream track over to determine the effect of the impact [similar to deltaAngleTrack] (rad)

           nTrackChunksImpact= (floor(deltaAngleTrack/0.15)+1) number of chunks to divide the progenitor track in near the impact [similar to nTrackChunks]

           nTrackChunks= (floor(deltaAngleTrack/0.15)+1) number of chunks to divide the progenitor track in at the observation time

           nKickPoints= (10xnTrackChunksImpact) number of points along the stream to compute the kicks at (kicks are then interpolated)

           spline_order= (3) order of the spline to interpolate the kicks with

           length_factor= (1.) consider impacts up to length_factor x length of the stream

        OUTPUT:

           object

        HISTORY:

           2015-12-07 - Started based on streamgapdf - Bovy (UofT)

        """
        # Parse kwargs, everything except for timpact can be arrays
        timpact= kwargs.pop('timpact',[1.])
        impactb= kwargs.pop('impactb',None)
        if impactb is None:
            sim_setup= True # we're just getting ready to run sims
            impactb= [0. for t in timpact]
        else: sim_setup= False
        subhalovel= kwargs.pop('subhalovel',
                               numpy.array([[0.,1.,0.] for t in timpact]))
        impact_angle= kwargs.pop('impact_angle',
                                 numpy.array([0.0001 for t in timpact]))
        GM= kwargs.pop('GM',None)
        rs= kwargs.pop('rs',None)
        subhalopot= kwargs.pop('subhalopot',None)
        hernquist= kwargs.pop('hernquist',False)
        if GM is None and rs is None and subhalopot is None:
            # If none given, just use a small impact to get the coord.
            # transform set up (use 220 and 8 for now, switch to config later)
            GM= [10**-5./bovy_conversion.mass_in_1010msol(220.,8.)
                 for t in timpact]
            rs= [0.04/8. for t in timpact]
        if GM is None: GM= [None for b in impactb]
        if rs is None: rs= [None for b in impactb]
        if subhalopot is None: subhalopot= [None for b in impactb]
        deltaAngleTrackImpact= kwargs.pop('deltaAngleTrackImpact',None)
        nTrackChunksImpact= kwargs.pop('nTrackChunksImpact',None)
        nTrackChunks= kwargs.pop('nTrackChunks',None)
        nKickPoints= kwargs.pop('nKickPoints',None)
        spline_order= kwargs.pop('spline_order',3)
        length_factor= kwargs.pop('length_factor',1.)
        # Analytical Plummer or general potential?
        self._general_kick= GM[0] is None or rs[0] is None
        if self._general_kick and numpy.any([sp is None for sp in subhalopot]):
            raise IOError("One of (GM=, rs=) or subhalopot= needs to be set to specify the subhalo's structure")
        if self._general_kick:
            self._subhalopot= subhalopot
        else:
            self._GM= GM
            self._rs= rs
        # Now run the regular streamdf setup, but without calculating the
        # stream track (nosetup=True)
        kwargs['nosetup']= True
        super(streampepperdf,self).__init__(*args,**kwargs)
        # Adjust the angles
        if not self._leading and sim_setup: impact_angle*= -1.
        # Setup streamgapdf objects to setup the machinery to go between
        # (x,v) and (Omega,theta) near the impacts
        self._uniq_timpact= sorted(list(set(timpact)))
        self._sgapdfs_coordtransform= {}
        for ti in self._uniq_timpact:
            sgapdf_kwargs= copy.deepcopy(kwargs)
            sgapdf_kwargs['timpact']= ti
            sgapdf_kwargs['impact_angle']= impact_angle[0]#only affects a check
            if not self._leading: sgapdf_kwargs['impact_angle']= -1.
            sgapdf_kwargs['deltaAngleTrackImpact']= deltaAngleTrackImpact
            sgapdf_kwargs['nTrackChunksImpact']= nTrackChunksImpact
            sgapdf_kwargs['nKickPoints']= nKickPoints
            sgapdf_kwargs['spline_order']= spline_order
            sgapdf_kwargs['hernquist']= hernquist
            sgapdf_kwargs['GM']= GM[0] # Just to avoid error
            sgapdf_kwargs['rs']= rs[0]
            sgapdf_kwargs['subhalopot']= subhalopot[0]
            sgapdf_kwargs['nokicksetup']= True
            self._sgapdfs_coordtransform[ti]= \
                                            streamgapdf(*args,**sgapdf_kwargs)
        # Also setup coordtransform for the current time, for transforming
        if not 0. in self._uniq_timpact:
            ti= 0.
            sgapdf_kwargs= copy.deepcopy(kwargs)
            sgapdf_kwargs['timpact']= ti
            sgapdf_kwargs['impact_angle']= impact_angle[0]#only affects a check
            if not self._leading: sgapdf_kwargs['impact_angle']= -1.
            sgapdf_kwargs['deltaAngleTrackImpact']= deltaAngleTrackImpact
            sgapdf_kwargs['nTrackChunksImpact']= nTrackChunks
            sgapdf_kwargs['nKickPoints']= nKickPoints
            sgapdf_kwargs['spline_order']= spline_order
            sgapdf_kwargs['hernquist']= hernquist
            sgapdf_kwargs['GM']= 0. # Just to avoid error
            sgapdf_kwargs['rs']= rs[0]
            sgapdf_kwargs['subhalopot']= subhalopot[0]
            sgapdf_kwargs['nokicksetup']= True
            self._sgapdfs_coordtransform[ti]=\
                                            streamgapdf(*args,**sgapdf_kwargs)
            # Grab sgapdfs_coordtransform[0]'s coordinate transformation
            # for that for the current time
            self._sgapdfs_coordtransform[ti]._impact_angle= 0.
            self._sgapdfs_coordtransform[ti].\
                _interpolate_stream_track_kick()
            self._sgapdfs_coordtransform[ti].\
                _interpolate_stream_track_kick_aA()
            self._interpolatedObsTrack=\
                self._sgapdfs_coordtransform[ti]._kick_interpolatedObsTrack
            self._ObsTrack= self._sgapdfs_coordtransform[ti]._gap_ObsTrack
            self._interpolatedObsTrackXY= \
                self._sgapdfs_coordtransform[ti]._kick_interpolatedObsTrackXY
            self._ObsTrackXY= self._sgapdfs_coordtransform[ti]._gap_ObsTrackXY
            self._alljacsTrack=\
                self._sgapdfs_coordtransform[ti]._gap_alljacsTrack
            self._allinvjacsTrack=\
                self._sgapdfs_coordtransform[ti]._gap_allinvjacsTrack
            self._interpolatedObsTrackAA=\
                self._sgapdfs_coordtransform[ti]._kick_interpolatedObsTrackAA
            self._ObsTrackAA= self._sgapdfs_coordtransform[ti]._gap_ObsTrackAA
            self._interpolatedThetasTrack=\
                self._sgapdfs_coordtransform[ti]._kick_interpolatedThetasTrack
            self._thetasTrack=\
                self._sgapdfs_coordtransform[ti]._gap_thetasTrack
            self._nTrackChunks=\
                self._sgapdfs_coordtransform[ti]._nTrackChunksImpact
        self._gap_leading=\
            self._sgapdfs_coordtransform[timpact[0]]._gap_leading
        # Compute all kicks
        self._spline_order= spline_order
        self.hernquist= hernquist
        self._determine_deltaOmegaTheta_kicks(impact_angle,impactb,subhalovel,
                                              timpact,GM,rs,subhalopot)
        # Pre-compute probability of different times and angles ~ length
        self._setup_timeAndAngleSampling(length_factor)
        return None

    def _setup_timeAndAngleSampling(self,length_factor):
        """Function that computes the length of the stream at the different potential impact times, to determine relative probability of the stream being hit at those times (and the relative probability of different segments at that time)"""
        self._length_factor= length_factor
        # Loop through uniq_timpact, determining length of stream
        self._stream_len= {}
        self._icdf_stream_len= {}
        # Need to turn off timpact to get smooth density for length here
        store_timpact= self._timpact
        self._timpact= [] # reset, so density going into length is smooth
        for ti in self._uniq_timpact:
            # Determine total length in apar
            max_apar= self.length(tdisrupt=self._tdisrupt-ti)\
                *self._length_factor
            # Setup the interpolation at the impact time
            self._sgapdfs_coordtransform[ti]._impact_angle= 0.1 # dummy
            self._sgapdfs_coordtransform[ti]\
                ._interpolate_stream_track_kick()
            # Determine length of different segments
            dXda= self._sgapdfs_coordtransform[ti]._kick_interpTrackX\
                .derivative()
            dYda= self._sgapdfs_coordtransform[ti]._kick_interpTrackY\
                .derivative()
            dZda= self._sgapdfs_coordtransform[ti]._kick_interpTrackZ\
                .derivative()
            apars= numpy.linspace(0.,max_apar,
                                  self._sgapdfs_coordtransform[ti]._nKickPoints)
            cumul_stream_len= numpy.hstack((0.,numpy.array([\
                    integrate.quad(lambda da: numpy.sqrt(dXda(da)**2.\
                                                             +dYda(da)**2.\
                                                             +dZda(da)**2.),
                                   al,au)[0] for al,au in zip(apars[:-1],
                                                              apars[1:])])))
            cumul_stream_len= numpy.cumsum(cumul_stream_len)
            self._stream_len[ti]= cumul_stream_len[-1]
            self._icdf_stream_len[ti]=\
                interpolate.InterpolatedUnivariateSpline(\
                cumul_stream_len/cumul_stream_len[-1],apars,k=3)
        # Now determine relative probability of different times
        self._ptimpact= numpy.array([self._stream_len[ti]
                                     for ti in self._uniq_timpact])
        self._ptimpact/= numpy.sum(self._ptimpact)
        # Restore timpact
        self._timpact= store_timpact
        return None

############################## SIMULATION FUNCTIONS ###########################
    def simulate(self,nimpact=None,rate=1.,
                 sample_GM=None,sample_rs=None,
                 Xrs=3.,sigma=120./220.):
        """
        NAME:

           simulate

        PURPOSE:

           simulate a set of impacts

        INPUT:

           For the number of impacts, specify either:

              nimpact= (None) number of impacts; if not set, fall back on rate

              rate= (1.) expected number of impacts

           sample_GM= (None) function that returns a sample GM (no arguments)

           sample_rs= (None) function that returns a sample rs as a function of GM

           Xrs= (3.) consider impact parameters up to X rs

           sigma= (120/220) velocity dispersion of the DM subhalo population

        OUTPUT:

           (none; just sets up the instance for the new set of impacts)

        HISTORY

           2015-12-14 - Written - Bovy (UofT)

        """
        # Number of impacts
        if nimpact is None:
            nimpact= numpy.random.poisson(rate)
        # Sample times propto td-time
        timpacts= numpy.array(self._uniq_timpact)[\
            numpy.random.choice(len(self._uniq_timpact),size=nimpact,
                                p=self._ptimpact)]
        # Sample angles from the part of the stream that existed then
        impact_angles= numpy.array([\
                self._icdf_stream_len[ti](numpy.random.uniform())
                for ti in timpacts])
        # Keep GM and rs the same for now
        if sample_GM is None:
            raise ValueError("sample_GM keyword to simulate must be specified")
        else:
            GMs= numpy.array([sample_GM() for a in impact_angles])
        if sample_rs is None:
            raise ValueError("sample_rs keyword to simulate must be specified")
        else:
            rss= numpy.array([sample_rs(gm) for gm in GMs])
        # impact b
        impactbs= (2.*numpy.random.uniform(size=len(impact_angles))-1.)*Xrs*rss
        # velocity
        subhalovels= numpy.empty((len(impact_angles),3))
        for ii in range(len(timpacts)):
            subhalovels[ii]=\
                self._draw_impact_velocities(timpacts[ii],sigma,
                                             impact_angles[ii],n=1)[0]
        # Flip angle sign if necessary
        if not self._gap_leading: impact_angles*= -1.
        # Setup
        self.set_impacts(impact_angle=impact_angles,
                         impactb=impactbs,
                         subhalovel=subhalovels,
                         timpact=timpacts,
                         GM=GMs,rs=rss)
        return None

    def _draw_impact_velocities(self,timpact,sigma,impact_angle,n=1):
        """Draw impact velocities from the distribution relative to the stream
        at this time"""
        # Along the stream and tangential to the stream are Gaussian
        vy= numpy.random.normal(scale=sigma,size=n)
        vt= numpy.random.normal(scale=sigma,size=n)
        # cylindrical radial is minus Rayleigh
        vr= -numpy.random.rayleigh(scale=sigma,size=n)
        # build vx and vy
        theta= numpy.random.uniform(size=n)*2.*numpy.pi
        vx= vr*numpy.cos(theta)-vt*numpy.sin(theta)
        vz= vr*numpy.sin(theta)+vt*numpy.cos(theta)
        out= numpy.array([vx,vy,vz]).T
        # Now rotate wrt stream frame
        self._sgapdfs_coordtransform[timpact]._impact_angle= impact_angle
        self._sgapdfs_coordtransform[timpact]\
            ._interpolate_stream_track_kick()
        streamDir= numpy.array([self._sgapdfs_coordtransform[timpact]\
                                    ._kick_interpTrackvX(impact_angle),
                                self._sgapdfs_coordtransform[timpact]\
                                    ._kick_interpTrackvY(impact_angle),
                                self._sgapdfs_coordtransform[timpact]\
                                    ._kick_interpTrackvZ(impact_angle)])
        # Build the rotation matrices and their inverse
        rotinv= _rotation_vy(streamDir[:,numpy.newaxis].T,
                             inv=True)[0]
        out= numpy.sum(\
            rotinv*numpy.swapaxes(numpy.tile(out.T,(3,1,1)).T,1,2),axis=-1)
        return out

    def set_impacts(self,**kwargs):
        """
        NAME:

           set_impacts

        PURPOSE:

           Setup a new set of impacts

        INPUT:

           Subhalo and impact parameters, for all impacts:

              impactb= impact parameter ([nimpact])

              subhalovel= velocity of the subhalo shape=(nimpact,3)

              impact_angle= angle offset from progenitor at which the impact occurred (at the impact time; in rad) ([nimpact])

              timpact time since impact ([nimpact])

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed); all kicks need to chose the same option

                 1( GM= mass of the subhalo ([nimpact])

                    rs= size parameter of the subhalo ([nimpact])

                 2( subhalopot= galpy potential object or list thereof (should be spherical); list of len nimpact (if the individual potentials are lists, need to give a list of lists)


        OUTPUT:

           (none; just sets up new set of impacts)

        HISTORY:

           2015-12-14 - Written - Bovy (UofT)

        """
        # Parse kwargs, everything except for timpact can be arrays
        impactb= kwargs.pop('impactb',[1.])
        subhalovel= kwargs.pop('subhalovel',numpy.array([[0.,1.,0.]]))
        impact_angle= kwargs.pop('impact_angle',[1.])
        GM= kwargs.pop('GM',None)
        rs= kwargs.pop('rs',None)
        subhalopot= kwargs.pop('subhalopot',None)
        if GM is None: GM= [None for b in impactb]
        if rs is None: rs= [None for b in impactb]
        if subhalopot is None: subhalopot= [None for b in impactb]
        timpact= kwargs.pop('timpact',[1.])
        # Run through previous setup to determine delta Omegas
        self._determine_deltaOmegaTheta_kicks(impact_angle,impactb,subhalovel,
                                              timpact,GM,rs,subhalopot)
        return None

    def _determine_deltaOmegaTheta_kicks(self,impact_angle,impactb,subhalovel,
                                         timpact,GM,rs,subhalopot):
        """Compute the kicks in frequency-angle space for all impacts"""
        self._nKicks= len(impactb)
        self._sgapdfs= []
        # Go through in reverse impact order
        for kk in numpy.argsort(timpact):
            sgdf= copy.deepcopy(self._sgapdfs_coordtransform[timpact[kk]])
            # compute the kick using the pre-computed coordinate transformation
            sgdf._determine_deltav_kick(impact_angle[kk],impactb[kk],
                                        subhalovel[kk],
                                        GM[kk],rs[kk],subhalopot[kk],
                                        self._spline_order,
                                        self.hernquist)
            sgdf._determine_deltaOmegaTheta_kick(self._spline_order)
            self._sgapdfs.append(sgdf)
        # Store impact parameters
        sortIndx= numpy.argsort(numpy.array(timpact))
        self._timpact= numpy.array(timpact)[sortIndx]
        self._impact_angle= numpy.array(impact_angle)[sortIndx]
        self._impactb= numpy.array(impactb)[sortIndx]
        self._subhalovel= numpy.array(subhalovel)[sortIndx]
        self._GM= numpy.array(GM)[sortIndx]
        self._rs= numpy.array(rs)[sortIndx]
        self._subhalopot= [subhalopot[ii] for ii in sortIndx]
        # For _approx_pdf, also combine kicks that occur at same time
        self._sgapdfs_uniq= []
        self._uniq_timpact_sim= sorted(list(set(self._timpact)))
        for ti in self._uniq_timpact_sim:
            sgdfc= self._combine_deltav_kicks(ti)
            # compute the kick using the pre-computed coordinate transformation
            sgdfc._determine_deltaOmegaTheta_kick(self._spline_order)
            self._sgapdfs_uniq.append(sgdfc)
        return None

    def _combine_deltav_kicks(self,timpact):
        """Internal function to combine those deltav kicks that occur at the same impact time into the output sgdfc object, so they can be used to produce a combined delta Omega (Delta apar)"""
        # Find all the kicks that occur at timpact
        kickIndx= numpy.where(self._timpact == timpact)[0]
        # Copy the first one as the baseline
        sgdfc= copy.deepcopy(self._sgapdfs[kickIndx[0]])
        # Add deltavs from other kicks
        for kk in kickIndx[1:]:
            sgdfc._kick_deltav+= self._sgapdfs[kk]._kick_deltav
        return sgdfc

    def approx_kicks(self,threshold,relative=True):
        """
        NAME:
           approx_kicks
        PURPOSE:
           Remove parts of the interpolated kicks that can be neglected, for self._sgapdfs_uniq
        INPUT:
           threshold - remove parts of *individual* kicks in Opar below this threshold
           relative= (True) whether the threshold is relative or absolute (in dOpar)
        OUTPUT:
           (none; just adjusts the internal kicks; there is no way back)
        HISTORY:
           2016-01-16 - Written - Bovy (UofT)
        """
        for ii,ti in enumerate(self._uniq_timpact_sim):
            # Find all the kicks that occur at timpact
            kickIndx= numpy.where(self._timpact == ti)[0]
            keepIndx= numpy.zeros(\
                len(self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1]),
                dtype='bool')
            tx= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[:-1]
            range_tx= numpy.arange(len(tx))
            for kk in kickIndx:
                # Only keep those above threshold or between peaks, first
                # find peaks
                relKick=\
                    numpy.fabs(self._sgapdfs[kk]._kick_interpdOpar(tx))
                peakLeftIndx=\
                    numpy.argmax(relKick/numpy.amax(relKick)\
                                     *(tx <= self._sgapdfs[kk]._impact_angle))
                peakRightIndx=\
                    numpy.argmax(relKick/numpy.amax(relKick)\
                                     *(tx > self._sgapdfs[kk]._impact_angle))
                if relative:
                    relKick/= numpy.amax(relKick)
                keepIndx+= (relKick >= threshold)\
                    +((range_tx >= peakLeftIndx)*(range_tx <= peakRightIndx))
            # Only keep those in keepIndx
            self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,True-keepIndx]=\
                0.
            self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-2,True-keepIndx]=\
                0.
            c12= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1]\
                *self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-2]
            nzIndx= numpy.nonzero((c12 != 0.)+(numpy.roll(c12,1)-c12 != 0.)\
                                      +(range_tx == 0))[0]
            nzIndx= numpy.nonzero((c12 != 0.)+(numpy.roll(c12,1)-c12 != 0.)\
                                      +(numpy.roll(c12,2)-c12 != 0.)\
                                      +(c12-numpy.roll(c12,-1) != 0.)\
                                      +(range_tx == 0))[0]
            # Linearly interpolate over dropped ranges
            droppedIndx= numpy.nonzero(((c12 == 0.)\
                                           *(numpy.roll(c12,1)-c12 != 0.))\
                                           +((c12 == 0.)*(range_tx == 0)))[0]
            for dd in droppedIndx:
                if dd == 0:
                    prevx= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[0]
                    prevVal= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,0]
                else:
                    self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,dd]=\
                        self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,dd-1]+(self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[dd]-self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[dd-1])*self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-2,dd-1]
                    prevx= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[dd]
                    prevVal= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,dd]
                nextIndx= list(nzIndx).index(dd)
                if nextIndx == len(nzIndx)-1:
                    nextx= self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[-1]
                    nextVal= 0.
                else:
                    nextIndx= nzIndx[nextIndx+1]
                    nextx=\
                        self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[nextIndx]
                    nextVal=\
                        self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-1,nextIndx]
                self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[-2,dd]=\
                    (nextVal-prevVal)/(nextx-prevx)
            self._sgapdfs_uniq[ii]._kick_interpdOpar_poly= interpolate.PPoly(\
                self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.c[:,nzIndx],
                self._sgapdfs_uniq[ii]._kick_interpdOpar_poly.x[numpy.hstack((nzIndx,[len(tx)]))])
        return None

############################ PHASE-SPACE DF FUNCTIONS #########################
    def pOparapar(self,Opar,apar):
        """
        NAME:

           pOparapar

        PURPOSE:

           return the probability of a given parallel (frequency,angle) offset pair

        INPUT:

           Opar - parallel frequency offset (array)

           apar - parallel angle offset along the stream (scalar)

        OUTPUT:

           p(Opar,apar)

        HISTORY:

           2015-12-09 - Written - Bovy (UofT)

        """
        if isinstance(Opar,(int,float,numpy.float32,numpy.float64)):
            Opar= numpy.array([Opar])
        apar= numpy.tile(apar,(len(Opar)))
        out= numpy.zeros(len(Opar))
        # Need to rewind each to each impact sequentially
        current_Opar= copy.copy(Opar)
        current_apar= copy.copy(apar)
        current_timpact= 0.
        remaining_indx= numpy.ones(len(Opar),dtype='bool')
        for kk,timpact in enumerate(self._timpact):
            # Compute ts and where they were at current impact for all
            ts= current_apar/current_Opar
            # Evaluate those that have ts < (timpact-current_timpact)
            afterIndx= remaining_indx*(ts < (timpact-current_timpact))\
                *(ts >= 0.)
            out[afterIndx]=\
                super(streampepperdf,self).pOparapar(current_Opar[afterIndx],
                                                     current_apar[afterIndx],
                                                     tdisrupt=
                                                     self._tdisrupt\
                                                         -current_timpact)
            remaining_indx*= (True-afterIndx)
            if numpy.sum(remaining_indx) == 0: break
            # Compute Opar and apar just before this kick for next kick
            current_apar-= current_Opar*(timpact-current_timpact)
            dOpar_impact= self._sgapdfs[kk]._kick_interpdOpar(current_apar)
            current_Opar-= dOpar_impact
            current_timpact= timpact
        # Need one last evaluation for before the first kick
        if numpy.sum(remaining_indx) > 0:
            # Evaluate
            out[remaining_indx]=\
                super(streampepperdf,self).pOparapar(\
                current_Opar[remaining_indx],current_apar[remaining_indx],
                tdisrupt=self._tdisrupt-current_timpact)
        return out

    def _density_par(self,dangle,tdisrupt=None,approx=True,
                     force_indiv_impacts=False):
        """The raw density as a function of parallel angle
        approx= use faster method that directly integrates the spline
        representations
        force_indiv_impacts= (False) in approx, explicitly use each individual impact at a given time rather than their combined impact at that time (should give the same)"""
        if len(self._timpact) == 0:
            return super(streampepperdf,self)._density_par(dangle,
                                                           tdisrupt=tdisrupt)
        if tdisrupt is None: tdisrupt= self._tdisrupt
        if approx:
            return self._density_par_approx(dangle,tdisrupt,
                                            force_indiv_impacts)
        else:
            smooth_dens= super(streampepperdf,self)._density_par(dangle)
            return integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                                      *(1+T*T)/(1-T*T)**2.\
                                      *self.pOparapar(T/(1-T*T)\
                                                          *numpy.sqrt(self._sortedSigOEig[2])\
                                                          +self._meandO,dangle)/\
                                      smooth_dens,
                                  -1.,1.,
                                  limit=100,epsabs=1.e-06,epsrel=1.e-06)[0]\
                                  *smooth_dens
    def _densityAndOmega_par_approx(self,dangle):
        """Convenience function  to return both, avoiding doubling the computational time"""
        # First get the approximate PD
        ul,da,ti,c0,c1= self._approx_pdf(dangle,False)
        return (self._density_par_approx(dangle,self._tdisrupt,False,False,
                                    ul,da,ti,c0,c1),
                self._meanOmega_approx(dangle,self._tdisrupt,False,
                                       ul,da,ti,c0,c1))

    def _density_par_approx(self,dangle,tdisrupt,
                            force_indiv_impacts,
                            _return_array=False,
                            *args):
        """Compute the density as a function of parallel angle using the
        spline representations"""
        if len(args) == 0:
            ul,da,ti,c0,c1= self._approx_pdf(dangle,force_indiv_impacts)
        else:
            ul,da,ti,c0,c1= args
            ul= copy.copy(ul)
        # Find the lower limit of the integration interval
        lowbindx,lowx,edge= self.minOpar(dangle,False,force_indiv_impacts,True,
                                         ul,da,ti,c0,c1)
        ul[lowbindx-1]= ul[lowbindx]-lowx
        # Integrate each interval
        out= (0.5/c1*(special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                                      *(ul-c0-self._meandO))\
                          -special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])
                                       *(ul-c0-self._meandO
                                         -c1*(ul-numpy.roll(ul,1))))))
        if _return_array:
            return out
        out= numpy.sum(out[lowbindx:])
        # Add integration to infinity
        out+= 0.5*(1.+special.erf((self._meandO-ul[-1])\
                                      /numpy.sqrt(2.*self._sortedSigOEig[2])))
        # Add integration to edge if edge
        if edge:
            out+= 0.5*(special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                                       *(ul[0]-self._meandO))\
                           -special.erf(1./numpy.sqrt(2.*self._sortedSigOEig[2])
                                        *(dangle/tdisrupt-self._meandO)))
        return out

    def _approx_pdf(self,dangle,force_indiv_impacts=False):
        """Internal function to return all of the parameters of the (approximat) p(Omega,angle)"""
        # Use individual impacts or combination?
        if force_indiv_impacts:
            sgapdfs= self._sgapdfs
            timpact= self._timpact
        else:
            sgapdfs= self._sgapdfs_uniq
            timpact= self._uniq_timpact_sim
        # First construct the breakpoints for the last impact for this dangle,
        # and start on the upper limits
        Oparb= (dangle-sgapdfs[0]._kick_interpdOpar_poly.x)\
            /timpact[0]
        ul= Oparb[::-1]
        # Array of previous pw-poly coeffs and breaks
        pwpolyBreak= sgapdfs[0]._kick_interpdOpar_poly.x[::-1]
        pwpolyCoeff0= numpy.append(\
            sgapdfs[0]._kick_interpdOpar_poly.c[-1],0.)[::-1]
        pwpolyCoeff1= numpy.append(\
            sgapdfs[0]._kick_interpdOpar_poly.c[-2],0.)[::-1]
        # Arrays for propagating the lower and upper limits through the impacts
        da= numpy.ones_like(ul)*dangle
        ti= numpy.ones_like(ul)*timpact[0]
        do= -pwpolyCoeff0-pwpolyCoeff1*(da-pwpolyBreak)
        # Arrays for final coefficients
        c0= pwpolyCoeff0
        c1= 1.+pwpolyCoeff1*timpact[0]
        cx= numpy.zeros(len(ul))
        for kk in range(1,len(timpact)):
            ul, da, ti, do, c0, c1, cx, \
                pwpolyBreak, pwpolyCoeff0, pwpolyCoeff1=\
                self._update_approx_prevImpact(kk,ul,da,ti,do,c0,c1,cx,
                                               pwpolyBreak,
                                               pwpolyCoeff0,pwpolyCoeff1,
                                               sgapdfs,timpact)
        # Form final c0 by adding cx times ul
        c0-= cx*ul
        return (ul,da,ti,c0,c1)

    def _update_approx_prevImpact(self,kk,ul,da,ti,do,c0,c1,cx,
                                  pwpolyBreak,pwpolyCoeff0,pwpolyCoeff1,
                                  sgapdfs,timpact):
        """Update the lower and upper limits, and the coefficient arrays when
        going through the previous impact"""
        # Compute matrix of upper limits for each current breakpoint and each
        # previous breakpoint
        da_u= da-(timpact[kk]-timpact[kk-1])*do
        ti_u= ti+(timpact[kk]-timpact[kk-1])*c1
        xj= numpy.tile(sgapdfs[kk]._kick_interpdOpar_poly.x[::-1],
                       (len(ul),1)).T
        ult= (da_u-xj)/ti_u
        # Determine which of these fall within the previous set of limits,
        # allowing duplicates is easiest
        limitIndx= (ult >= numpy.roll(ul,1))
        limitIndx[:,0]= True
        limitIndx*= (ult <= ul)
        # Only keep those, flatten, add the previous set, and sort; this is the
        # new set of upper limits (barring duplicates, see below)
        ul_u= numpy.append(ult[limitIndx].flatten(),ul)
        # make sure to keep duplicates 2nd (important later when assigning
        # coefficients to the old limits)
        limitsIndx= numpy.argsort(\
            stats.rankdata(ul_u,method='ordinal').astype('int')-1)
        limitusIndx= numpy.argsort(limitsIndx) # to un-sort later
        ul_u= ul_u[limitsIndx]
        # Start updating the coefficient arrays
        tixpwpoly= ti*pwpolyCoeff1
        c0_u= c0+tixpwpoly*ul
        cx_u= cx+tixpwpoly
        # Keep other arrays in sync with the limits
        nNewCoeff= len(sgapdfs[kk]._kick_interpdOpar_poly.x)
        da_u= numpy.append(numpy.tile(da_u,(nNewCoeff,1))[limitIndx].flatten(),
                            da_u)[limitsIndx]
        ti_u= numpy.append(numpy.tile(ti_u,(nNewCoeff,1))[limitIndx].flatten(),
                           ti_u)[limitsIndx]
        do_u= numpy.append(numpy.tile(do,(nNewCoeff,1))[limitIndx].flatten(),
                           do)[limitsIndx]
        c0_u= numpy.append(numpy.tile(c0_u,(nNewCoeff,1))[limitIndx].flatten(),
                           c0_u)[limitsIndx]
        c1_u= numpy.append(numpy.tile(c1,(nNewCoeff,1))[limitIndx].flatten(),
                           c1)[limitsIndx]
        cx_u= numpy.append(numpy.tile(cx_u,(nNewCoeff,1))[limitIndx].flatten(),
                           cx_u)[limitsIndx]
        # Also update the previous coefficients, figuring out where old limits
        # were inserted
        insertIndx= numpy.sort(\
            limitusIndx[numpy.arange(numpy.sum(limitIndx),len(ul_u))])[:-1]
        pwpolyBreak_u= numpy.append(\
            xj[limitIndx].flatten(),numpy.zeros(len(ul)))[limitsIndx]
        pwpolyCoeff0_u= numpy.append(numpy.tile(\
                numpy.append(\
                    sgapdfs[kk]._kick_interpdOpar_poly.c[-1],0.)[::-1],
                (len(ul),1)).T[limitIndx].flatten(),\
                                       numpy.zeros(len(ul)))[limitsIndx]
        pwpolyCoeff1_u= numpy.append(numpy.tile(\
                numpy.append(\
                    sgapdfs[kk]._kick_interpdOpar_poly.c[-2],0.)[::-1],
                (len(ul),1)).T[limitIndx].flatten(),\
                                       numpy.zeros(len(ul)))[limitsIndx]
        # Need to do this sequentially, if multiple inserted in one range
        for ii in insertIndx[::-1]:
            pwpolyBreak_u[ii]= pwpolyBreak_u[ii+1]
            pwpolyCoeff0_u[ii]= pwpolyCoeff0_u[ii+1]
            pwpolyCoeff1_u[ii]= pwpolyCoeff1_u[ii+1]
        do_u-= pwpolyCoeff0_u+pwpolyCoeff1_u*(da_u-pwpolyBreak_u)
        # Now update the coefficient arrays
        c0_u+= pwpolyCoeff0_u
        c1_u+= pwpolyCoeff1_u*ti_u
        # Remove duplicates in limits
        dupIndx= numpy.roll(ul_u,-1)-ul_u != 0
        return (ul_u[dupIndx],da_u[dupIndx],ti_u[dupIndx],do_u[dupIndx],
                c0_u[dupIndx],c1_u[dupIndx],cx_u[dupIndx],
                pwpolyBreak_u[dupIndx],pwpolyCoeff0_u[dupIndx],
                pwpolyCoeff1_u[dupIndx])

    def minOpar(self,dangle,bruteforce=False,
                force_indiv_impacts=False,_return_raw=False,*args):
        """
        NAME:
           minOpar
        PURPOSE:
           return the approximate minimum parallel frequency at a given angle
        INPUT:
           dangle - parallel angle
           bruteforce= (False) if True, just find the minimum by evaluating where p(Opar,apar) becomes non-zero
           force_indiv_impacts= (False) if True, explicitly use the streamgapdf object of each individual impact; otherwise combine impacts at the same time (should give the same result)
        OUTPUT:
           minimum frequency that gets to this parallel angle
        HISTORY:
           2016-01-01 - Written - Bovy (UofT)
           2016-01-02 - Added bruteforce - Bovy (UofT)
        """
        if bruteforce:
            return self._minOpar_bruteforce(dangle)
        if len(args) == 0:
            ul,da,ti,c0,c1= self._approx_pdf(dangle,force_indiv_impacts)
        else:
            ul,da,ti,c0,c1= args
        # Find the lower limit of the integration interval
        lowx= ((ul-c0)*(self._tdisrupt-self._timpact[-1])+ul*ti-da)\
            /((self._tdisrupt-self._timpact[-1])*c1+ti)
        nlowx= lowx/(ul-numpy.roll(ul,1))
        nlowx[0]*= -1. #need to adjust sign bc of roll
        lowbindx= numpy.argmax((lowx >= 0.)*(nlowx <= 10.)) # not too crazy
        if lowbindx > 0 and -nlowx[lowbindx-1] < nlowx[lowbindx]:
            lowbindx-= 1
        edge= False
        if lowbindx == 0: # edge case
            lowbindx= 1
            lowx[lowbindx]= ul[1]-ul[0]
            edge= True
        if _return_raw:
            return (lowbindx,lowx[lowbindx],edge)
        elif edge:
            return dangle/self._tdisrupt
        else:
            return ul[lowbindx]-lowx[lowbindx]

    def _minOpar_bruteforce(self,dangle):
        nzguess= numpy.array([self._meandO,self._meandO])
        sig= numpy.array([numpy.sqrt(self._sortedSigOEig[2]),
                          numpy.sqrt(self._sortedSigOEig[2])])
        while numpy.all(self.pOparapar(nzguess,dangle) == 0.):
            nzguess+= sig/3.
        nzguess= nzguess[nzguess != 0.][0]
        nzval= self.pOparapar(nzguess,dangle)
        guesso= nzguess-numpy.sqrt(self._sortedSigOEig[2])
        while self.pOparapar(guesso,dangle)-10.**-6.*nzval > 0.:
            guesso-= numpy.sqrt(self._sortedSigOEig[2])
        return optimize.brentq(\
            lambda x: self.pOparapar(x*nzguess,dangle)-10.**-6.*nzval,
            guesso/nzguess,1.,xtol=1e-8,rtol=1e-8)*nzguess

    def meanOmega(self,dangle,oned=False,approx=True,
                  force_indiv_impacts=False,tdisrupt=None,norm=True):
        """
        NAME:

           meanOmega

        PURPOSE:

           calculate the mean frequency as a function of angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

           oned= (False) if True, return the 1D offset from the progenitor (along the direction of disruption)

           approx= (True) if True, compute the mean Omega by direct integration of the spline representation

           force_indiv_impacts= (False) if True, explicitly use the streamgapdf object of each individual impact; otherwise combine impacts at the same time (should give the same result)

        OUTPUT:

           mean Omega

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if len(self._timpact) == 0:
            out= super(streampepperdf,self).meanOmega(dangle,
                                                      tdisrupt=tdisrupt,
                                                      oned=oned)
            if not norm:
                return out*super(streampepperdf,self)._density_par(dangle,
                                                                   tdisrupt=tdisrupt)
            else:
                return out
        if tdisrupt is None: tdisrupt= self._tdisrupt
        if approx:
            dO1D= self._meanOmega_approx(dangle,tdisrupt,force_indiv_impacts)
        else:
            if not norm:
                denom= super(streampepperdf,self)._density_par(dangle)
            else:
                denom= self._density_par(dangle,approx=approx)
            dO1D=\
                integrate.quad(lambda T: (T/(1-T*T)\
                                              *numpy.sqrt(self._sortedSigOEig[2])\
                                              +self._meandO)\
                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                   *(1+T*T)/(1-T*T)**2.\
                                   *self.pOparapar(T/(1-T*T)\
                                                       *numpy.sqrt(self._sortedSigOEig[2])\
                                                       +self._meandO,dangle)\
                                   /self._meandO/denom,
                               -1.,1.,
                               limit=100,epsabs=1.e-06,epsrel=1.e-06)[0]\
                               *self._meandO
            if not norm: dO1D*= denom
        if oned: return dO1D
        else:
            return self._progenitor_Omega+dO1D*self._dsigomeanProgDirection\
                *self._sigMeanSign

    def _meanOmega_approx(self,dangle,tdisrupt,force_indiv_impacts,
                          *args):
        """Compute the mean frequency as a function of parallel angle using the
        spline representations"""
        if len(args) == 0:
            ul,da,ti,c0,c1= self._approx_pdf(dangle,force_indiv_impacts)
        else:
            ul,da,ti,c0,c1= args
            ul= copy.copy(ul)
        # Get the density in various forms
        dens_arr= self._density_par_approx(dangle,tdisrupt,force_indiv_impacts,
                                           True,
                                           ul,da,ti,c0,c1)
        dens= self._density_par_approx(dangle,tdisrupt,force_indiv_impacts,
                                       False,ul,da,ti,c0,c1)
        # Compute numerator using the approximate PDF
        # Find the lower limit of the integration interval
        lowbindx,lowx,edge= self.minOpar(dangle,False,force_indiv_impacts,True,
                                         ul,da,ti,c0,c1)
        ul[lowbindx-1]= ul[lowbindx]-lowx
        # Integrate each interval
        out= numpy.sum(((ul+(self._meandO+c0-ul)/c1)*dens_arr
                        +numpy.sqrt(self._sortedSigOEig[2]/2./numpy.pi)/c1**2.\
                            *(numpy.exp(-0.5*(ul-c0
                                              -c1*(ul-numpy.roll(ul,1))
                                              -self._meandO)**2.
                                         /self._sortedSigOEig[2])
                              -numpy.exp(-0.5*(ul-c0-self._meandO)**2.
                                          /self._sortedSigOEig[2])))\
                           [lowbindx:])
        # Add integration to infinity
        out+= 0.5*(numpy.sqrt(2./numpy.pi)*numpy.sqrt(self._sortedSigOEig[2])\
                       *numpy.exp(-0.5*(self._meandO-ul[-1])**2.\
                                       /self._sortedSigOEig[2])
                   +self._meandO
                   *(1.+special.erf((self._meandO-ul[-1])
                                    /numpy.sqrt(2.*self._sortedSigOEig[2]))))
        # Add integration to edge if edge
        if edge:
            out+= 0.5*(self._meandO*(special.erf(\
                    1./numpy.sqrt(2.*self._sortedSigOEig[2])\
                        *(ul[0]-self._meandO))
                                      -special.erf(\
                    1./numpy.sqrt(2.*self._sortedSigOEig[2])
                    *(dangle/tdisrupt-self._meandO)))
                       +numpy.sqrt(2.*self._sortedSigOEig[2]/numpy.pi)\
                           *(numpy.exp(-0.5*(dangle/tdisrupt-self._meandO)**2.
                                        /self._sortedSigOEig[2])
                              -numpy.exp(-0.5*(ul[0]-self._meandO)**2.
                                          /self._sortedSigOEig[2])))
        return out/dens

    def meanTrack(self,dangle,approx=True,force_indiv_impacts=False,
                  coord='XY',_mO=None):
        """
        NAME:

           meanTrack

        PURPOSE:

           calculate the mean track in configuration space as a function of angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

           approx= (True) if True, compute the mean Omega by direct integration of the spline representation

           force_indiv_impacts= (False) if True, explicitly use the streamgapdf object of each individual impact; otherwise combine impacts at the same time (should give the same result)

           coord= ('XY') coordinates to output:
                  'XY' for rectangular Galactocentric
                  'RvR' for cylindrical Galactocentric
                  'lb' for Galactic longitude/latitude
                  'custom' for custom sky coordinates

           _mO= (None) if set, this is the oned meanOmega at dangle (in case this has already been computed)

        OUTPUT:

           mean track (dangle): [6]

        HISTORY:

           2016-02-22 - Written - Bovy (UofT)

        """
        if _mO is None:
            _mO= self.meanOmega(dangle,oned=True,approx=approx,
                                force_indiv_impacts=force_indiv_impacts)
        # Go to 3D omega and 3D angle
        _mO= numpy.atleast_1d(_mO)
        dangle= numpy.atleast_1d(dangle)
        mO= numpy.tile(self._progenitor_Omega,(_mO.shape[0],1)).T\
            +_mO*numpy.tile(self._dsigomeanProgDirection,(_mO.shape[0],1)).T\
            *self._sigMeanSign
        da= numpy.tile(self._progenitor_angle,(_mO.shape[0],1)).T\
            +dangle\
            *numpy.tile(self._dsigomeanProgDirection,(_mO.shape[0],1)).T\
            *self._sigMeanSign
        # Convert to configuration space using Jacobian
        out= self._approxaAInv(mO[0],mO[1],mO[2],da[0],da[1],da[2])
        if coord.lower() == 'rvr':
            return out
        # Go Galactocentric rectangular
        XY= numpy.zeros_like(out)
        XY[0]= out[0]*numpy.cos(out[5])
        XY[1]= out[0]*numpy.sin(out[5])
        XY[2]= out[3]
        TrackvX, TrackvY, TrackvZ=\
            bovy_coords.cyl_to_rect_vec(out[1],out[2],out[4],out[5])
        XY[3]= TrackvX
        XY[4]= TrackvY
        XY[5]= TrackvZ
        if coord.lower() == 'xy':
            return XY
        # Go to Galactic spherical
        XYZ= bovy_coords.galcenrect_to_XYZ(XY[0]*self._ro,
                                           XY[1]*self._ro,
                                           XY[2]*self._ro,
                                           Xsun=self._R0,Zsun=self._Zsun).T
        vXYZ= bovy_coords.galcenrect_to_vxvyvz(XY[3]*self._vo,
                                               XY[4]*self._vo,
                                               XY[5]*self._vo,
                                               vsun=self._vsun,
                                               Xsun=self._R0,Zsun=self._Zsun).T
        lbd= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=True)
        vlbd= bovy_coords.vxvyvz_to_vrpmllpmbb(vXYZ[0],vXYZ[1],vXYZ[2],
                                               lbd[:,0],lbd[:,1],lbd[:,2],
                                               degree=True)
        if coord.lower() == 'lb':
            return numpy.hstack((lbd,vlbd)).T

################################# STATISTICS ##################################
    def _structure_hash(self,digit,apars):
        return hashlib.md5(numpy.hstack(([digit],apars,self._impact_angle)))\
            .hexdigest()
    def csd(self,d1='density',d2='density',
            apars=None):
        if d1.lower() == 'density' or d2.lower() == 'density':
            new_hash= self._structure_hash(1,apars)
            if hasattr(self,'_dens_hash') and new_hash == self._dens_hash:
                dens= self._dens
                dens_unp= self._dens_unp
            else:
                dens= numpy.array([self.density_par(a) for a in apars])
                dens_unp= numpy.array([\
                        super(galpy.df.streampepperdf.streampepperdf,self)\
                            ._density_par(a) for a in apars])
                # Store in case they are needed again
                self._dens_hash= new_hash
                self._dens= dens
                self._dens_unp= dens_unp
        if d1.lower() == 'meanomega' or d2.lower() == 'meanomega':
            new_hash= self._structure_hash(2,apars)
            if hasattr(self,'_mO_hash') and new_hash == self._mO_hash:
                mO= self._mO
                mO_unp= self._mO_unp
            else:
                mO= numpy.array([self.meanOmega(a,oned=True)
                                 for a in apars])
                mO_unp= numpy.array([\
                        super(galpy.df.streampepperdf.streampepperdf,self)\
                            .meanOmega(a,oned=True) for a in apars])
                # Store in case they are needed again
                self._mO_hash= new_hash
                self._mO= mO
                self._mO_unp= mO_unp
        if d1.lower() == 'density':
            x= dens/dens_unp/numpy.sum(dens)*numpy.sum(dens_unp)
        elif d1.lower() == 'meanomega':
            x= (self._progenitor_Omega_along_dOmega+mO)\
                /(self._progenitor_Omega_along_dOmega+mO_unp)
        if d2.lower() == 'density':
            y= dens/dens_unp/numpy.sum(dens)*numpy.sum(dens_unp)
        elif d2.lower() == 'meanomega':
            y= (self._progenitor_Omega_along_dOmega+mO)\
                /(self._progenitor_Omega_along_dOmega+mO_unp)
        return signal.csd(x,y,fs=1./(apars[1]-apars[0]),scaling='spectrum')

################################SAMPLE THE DF##################################
    def _sample_aAt(self,n):
        """Sampling frequencies, angles, and times part of sampling, for stream with gaps"""
        # Use streamdf's _sample_aAt to generate unperturbed frequencies,
        # angles
        Om,angle,dt= super(streampepperdf,self)._sample_aAt(n)
        # Now rewind angles to the first impact, then apply all kicks,
        # and run forward again
        dangle_at_impact= angle-numpy.tile(self._progenitor_angle.T,(n,1)).T\
            -(Om-numpy.tile(self._progenitor_Omega.T,(n,1)).T)\
            *self._timpact[-1]
        dangle_par_at_impact=\
            numpy.dot(dangle_at_impact.T,
                      self._dsigomeanProgDirection)\
                      *self._sgapdfs[-1]._gap_sigMeanSign
        dOpar= numpy.dot((Om-numpy.tile(self._progenitor_Omega.T,(n,1)).T).T,
                         self._dsigomeanProgDirection)\
                         *self._sgapdfs[-1]._gap_sigMeanSign
        for kk,timpact in enumerate(self._timpact[::-1]):
            # Calculate and apply kicks (points not yet released have
            # zero kick)
            dOr= self._sgapdfs[-kk-1]._kick_interpdOr(dangle_par_at_impact)
            dOp= self._sgapdfs[-kk-1]._kick_interpdOp(dangle_par_at_impact)
            dOz= self._sgapdfs[-kk-1]._kick_interpdOz(dangle_par_at_impact)
            Om[0,:]+= dOr
            Om[1,:]+= dOp
            Om[2,:]+= dOz
            if kk < len(self._timpact)-1:
                run_to_timpact= self._timpact[::-1][kk+1]
            else:
                run_to_timpact= 0.
            angle[0,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdar(dangle_par_at_impact)\
                +dOr*timpact
            angle[1,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdap(dangle_par_at_impact)\
                +dOp*timpact
            angle[2,:]+=\
                self._sgapdfs[-kk-1]._kick_interpdaz(dangle_par_at_impact)\
                +dOz*timpact
            # Update parallel evolution
            dOpar+=\
                self._sgapdfs[-kk-1]._kick_interpdOpar(dangle_par_at_impact)
            dangle_par_at_impact+= dOpar*(timpact-run_to_timpact)
        return (Om,angle,dt)
