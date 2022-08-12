# Phase_space_refill
Code to repopulate phase space for given iniital distribution
Two different methods are used:
1 - Refill phase space using beam twiss parameters at the aperture and aperture information. This assumes a longitudinal distribution independent of the transverse coordinates, which is not complretely true in space charge dominated beams.
2 - Refill the phase space using neares neighbours strategy. This code is adapted from Jens Voelkers code in Matlab.
