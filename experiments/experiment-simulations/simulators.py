r"""
Code adapted from from Jo Bovy and Nilanjan Banik.
"""

import galpy
import gd1_util
import glob
import hypothesis
import numpy as np
import os
import pickle
import torch
import torch.multiprocessing as multiprocessing

from galpy.util import bovy_conversion
from hypothesis.simulation import Simulator as BaseSimulator
from util import compute_obs_density
from util import compute_obs_density_no_interpolation
from util import lb_to_phi12
from util import simulate_subhalos_mwdm



class GD1StreamSimulator(BaseSimulator):

    def __init__(self, hernquist_profile=True, max_subhalo_impacts=64):
        super(GD1StreamSimulator, self).__init__()
        self.hernquist_profile = hernquist_profile
        self.isob = 0.45
        self.chunk_size = 64
        self.length_factor = float(1)
        self.max_subhalo_impacts = int(max_subhalo_impacts)
        self.new_orb_lb = [188.04928416766532, 51.848594007807456, 7.559027173643999, 12.260258757214746, -5.140630283489461, 7.162732847549563]

    def _compute_impact_times(self, age):
        impact_times = []

        time_in_gyr = bovy_conversion.time_in_Gyr(vo=float(220), ro=float(8))
        for time in np.arange(1, self.max_subhalo_impacts + 1) / (self.max_subhalo_impacts):
            impact_times.append(time / time_in_gyr)

        return impact_times

    def _simulate_stream(self, age):
        sigv_age = (0.3 * 3.2) / age
        impacts = self._compute_impact_times(age)
        # Smooth stream.
        # stream_smooth_leading = gd1_util.setup_gd1model(age=age,
        #     isob=self.isob,
        #     leading=True,
        #     new_orb_lb=self.new_orb_lb,
        #     sigv=sigv_age)
        stream_smooth_trailing = gd1_util.setup_gd1model(age=age,
            isob=self.isob,
            leading=False,
            new_orb_lb=self.new_orb_lb,
            sigv=sigv_age)
        # Leading stream.
        # stream_leading = gd1_util.setup_gd1model(age=age,
        #     hernquist=self.hernquist_profile,
        #     isob=self.isob,
        #     leading=True,
        #     length_factor=self.length_factor,
        #     new_orb_lb=self.new_orb_lb,
        #     sigv=sigv_age,
        #     timpact=impacts)
        # Trailing stream.
        stream_trailing = gd1_util.setup_gd1model(age=age,
            hernquist=self.hernquist_profile,
            isob=self.isob,
            leading=False,
            length_factor=self.length_factor,
            new_orb_lb=self.new_orb_lb,
            sigv=sigv_age,
            timpact=impacts)

        # return stream_smooth_leading, stream_smooth_trailing, stream_leading, stream_trailing
        return None, stream_smooth_trailing, None, stream_trailing

    def forward(self, inputs):
        outputs = []

        for input in inputs:
            success = False
            while not success:
                try:
                    outputs.append(self._simulate_stream(input.item()))
                    success = True
                except:
                    pass

        return outputs



class WDMSubhaloSimulator(BaseSimulator):

    def __init__(self, streams, resolution=0.01, record_impacts=False, allow_no_impacts=True):
        super(WDMSubhaloSimulator, self).__init__()
        # Simulator parameters.
        self.record_impacts = record_impacts
        self.allow_no_impacts = allow_no_impacts # True -> allow observations with no impacts.
        # Subhalo parameters.
        self.Xrs = float(5)
        self.ravg = float(20)
        # Streams.
        self.streams = streams
        # Precomputed smooth stream properties.
        self.apars = np.arange(0.01, 1., resolution)
        self.dens_unp_leading = None
        self.omega_unp_leading = None
        self.dens_unp_trailing = None
        self.omega_unp_trailing = None
        self._precompute_smooth_stream_properties()

    def _precompute_smooth_stream_properties(self):
        # self._precompute_smooth_stream_leading()
        self._precompute_smooth_stream_trailing()

    def _precompute_smooth_stream_leading(self):
        stream_smooth = self.streams[0]
        self.dens_unp_leading = [stream_smooth._density_par(a) for a in self.apars]
        self.omega_unp_leading = [stream_smooth.meanOmega(a, oned=True) for a in self.apars]

    def _precompute_smooth_stream_trailing(self):
        stream_smooth = self.streams[1]
        self.dens_unp_trailing = [stream_smooth._density_par(a) for a in self.apars]
        self.omega_unp_trailing = [stream_smooth.meanOmega(a, oned=True) for a in self.apars]

    def _simulate_observation(self, wdm_mass, leading):
        if leading:
            return self._simulate_observation_leading(wdm_mass)
        else:
            return self._simulate_observation_trailing(wdm_mass)

    def _simulate_observation_leading(self, wdm_mass):
        success = False
        stream = self.streams[2] # Leading peppered stream
        dens_unp = self.dens_unp_leading
        omega_unp = self.omega_unp_leading
        while not success: # This is done to cover errors in the subhalo simulation code.
            try:
                output = self._simulate(wdm_mass, stream, dens_unp, omega_unp)
                success = True
            except Exception as e:
                print(e)

        return output

    def _simulate_observation_trailing(self, wdm_mass):
        success = False
        stream = self.streams[3] # Trailing peppered stream
        dens_unp = self.dens_unp_trailing
        omega_unp = self.omega_unp_trailing
        while not success:
            try:
                output = self._simulate(wdm_mass, stream, dens_unp, omega_unp)
                success = True
            except Exception as e:
                print(e)

        return output

    def _simulate(self, wdm_mass, stream, dens_unp, omega_unp):
        outputs = simulate_subhalos_mwdm(stream, m_wdm=wdm_mass, r=self.ravg, Xrs=self.Xrs)
        impact_angles = outputs[0]
        impactbs = outputs[1]
        subhalovels = outputs[2]
        timpacts = outputs[3]
        GMs = outputs[4]
        rss = outputs[5]
        # Check if subhalo hits were detected.
        num_impacts = len(GMs)
        has_impacts = num_impacts > 0
        if not self.allow_no_impacts and not has_impacts:
            raise ValueError("Only observations with impacts are allowed!")
        if has_impacts:
            stream.set_impacts(impactb=impactbs, subhalovel=subhalovels,
                impact_angle=impact_angles, timpact=timpacts, rs=rss, GM=GMs)
            densOmega = np.array([stream._densityAndOmega_par_approx(a) for a in self.apars]).T
            mO = densOmega[1]
        else:
            mO = omega_unp
        mT = stream.meanTrack(self.apars, _mO=mO, coord="lb")
        phi = lb_to_phi12(mT[0], mT[1], degree=True)[:, 0]
        phi[phi > 180] -= 360
        if has_impacts:
            density = compute_obs_density(phi, self.apars, densOmega[0], densOmega[1])
        else:
            density = compute_obs_density(phi, self.apars, dens_unp, omega_unp)
        # Check if the computed density has a nan-value.
        if np.isnan(density).sum() > 0:
            raise ValueError("nan values have been computed.")
        # Check if the impacts need to be recorded.
        if self.record_impacts:
            output = num_impacts, phi, density
        else:
            output = phi, density

        return output

    def _simulate_observations(self, wdm_mass):
        #leading = self._simulate_observation(wdm_mass, leading=True)
        trailing = self._simulate_observation(wdm_mass, leading=False)
        #output = leading, trailing

        #return output
        return trailing

    def forward(self, inputs):
        outputs = []

        inputs = inputs.view(-1, 1)
        for input in inputs:
            observation = self._simulate_observations(input.item())
            outputs.append(observation)

        return outputs



class PresimulatedStreamsWDMSubhaloSimulator(WDMSubhaloSimulator):

    def __init__(self, datadir, stream_model_index, resolution=0.01):
        streams = self._load_streams(datadir, stream_model_index)
        super(PresimulatedStreamsWDMSubhaloSimulator, self).__init__(
            streams, resolution=resolution)

    def _load_streams(self, datadir, index):
        # Stream models metadata.
        stream_models_path_query = datadir + "/stream-models/streams-*"
        directories = glob.glob(stream_models_path_query)
        directories.sort()
        stream_directories = directories
        stream_blocks = len(directories)
        streams_per_block = len(np.load(stream_directories[0] + "/inputs.npy"))
        streams_total = stream_blocks * streams_per_block
        # Load the streams.
        block_index = int(index / streams_per_block)
        stream_index_in_block = index % streams_per_block
        stream_directory = stream_directories[block_index]
        path = stream_directory + "/outputs.pickle"
        with open(path, "rb") as f:
            streams = pickle.load(f)[stream_index_in_block]

        return streams
