#model parameters 
#this file stores the constants, initial conditions, switches, and parameters used in the model.

import numpy as np

# soil parameters
# dictionary containing hydraulic propoerts for several soil types 
# each soil type contains: 
# n = porosity 
# fraction of total soil volume occupied by pore space 
# sfc = relative soil saturation at field capacity
# soil water content after gravitational drainage has occurred 
# sw = relative soil saturation at wilting point
# below this value, plants cannot extract water from the soil
# sh = hygroscopic soil moisture threshold 
# very dry condition below which evaporation is effectively zero 
# ks = saturated hydraulic conductivity (m/day)
# controls how easily water moves through saturated soil 
# b = soil pore size distribution parameter 
# used in the Laio soil water retention relationships 
soil_types = {
    "sand":     {"n": 0.43, "sfc": 0.17, "sw": 0.05,  "sh": 0.02,  "Ks": 0.20,  "b": 4.05},
    "loam":     {"n": 0.43, "sfc": 0.27, "sw": 0.12,  "sh": 0.02,  "Ks": 0.03,  "b": 4.38},
    "clay":     {"n": 0.50, "sfc": 0.40, "sw": 0.27,  "sh": 0.02,  "Ks": 0.005, "b": 11.4},
    "silt":     {"n": 0.50, "sfc": 0.30, "sw": 0.15,  "sh": 0.02,  "Ks": 0.01,  "b": 7.75},
    "original": {"n": 0.40, "sfc": 0.30, "sw": 0.065, "sh": 0.02,  "Ks": 1.1,   "b": 4.38},
}

selected_soil = "original"
soil = soil_types[selected_soil]
#unpack the selected soil propoerties into individual variables so other modules can access them 
n, sfc, sw, sh, Ks, b = soil["n"], soil["sfc"], soil["sw"], soil["sh"], soil["Ks"], soil["b"]

# switches
USE_SEASONAL_PET    = True
PLOT_PAPER_FIGS     = True
PLOT_WATER_BLOCK    = False
PLOT_CARBON_BLOCK   = False
PLOT_NITROGEN_BLOCK = False

# PET
# small positive minimum used in the seasonal PET function 
# this prevents PET from becoming exactly zero during the lowest part of the seasonal cycle
PET_MIN     = 5e-5
#fraction of total PET allocated to transpiration
# .55 means 55% of PET is allocated to transpiration, the rest is allocated to soil evaporation
PET_SPLIT_T = 0.55
# phase shift is used in the sinusoidal seasonal PET function 
# 0 means there is currently no phase shift, the peak of the seasonal PET function occurs at day 180 (mid-summer)
PET_PHASE   = 0.0

# vegetation/root zone parameters 

# root zone depth (m)
# used in the carbon/nitrogen model to convert between volumetric concentrations and areal quantities
Zr = 0.80
#soil saturation threshold above which transpiration is assumed to occur at its maximum potential rate
s_star = 0.17
# PET target
#target average annual potential ET (m/year) 
# PET is approximately 1 m/year 
ANNUAL_PET = 1.0                  # m/year
#convert the annual PET to a daily value (m/day)
BASELINE_PET_DAILY = ANNUAL_PET / 365.0

# rain
#mean rainfall intensity (m/day)
alpha = 0.011 #11mm 
#rain event frequency
#this value is passed to the stochastic rainfall generator 
lambda_inv = 1 / 0.23
#random seed for the stochastic rainfall generator
RAIN_SEED = 0

# time
# 20 years and half a day time step 
years, dt = 20, 0.5
#total simulation duration 
T = years * 365
#total number of numerical timesteps 
# 20 years x 365 days a year / 0.5 days/step = 14600 timesteps
N = int(T / dt)
#time vector corresponding to each model timestep [days]
time = np.arange(0, T, dt)

# carbon and nitrogen 
#dictionary containing the model parameters for carbon and nitrogen cycling
params = {
    "CNl": 58.0, #litter carbon to nitrogen ratio
    "CNh": 22.0, #humus carbon to nitrogen ratio
    "CNb": 11.5, # microbial biomass carbon to nitrogen ratio
    "kl": 0.000065, #litter decomposition rate 
    "kh": 0.0000025, #humus decomposition rate 
    "kd": 0.0085, #microbial biomass decomposition rate
    "rh_max": 0.25, #maximum humification coefficient, limits the fraction of decomposed litter than can be transsfered into the humus pool
    "rr": 0.60, # respiration fraction, fraction of decomposed carbon that is lost as CO2
    "kn": 0.6, # nitrification rate, fraction of ammonium that is converted to nitrate
    "up_pass": 1.0, # passive uptake scaling parameter 
    "up_frac_NH4": 0.4, #fraction associated with nh4 passive uptake 
    "imm_bias": 0.6, # bias controlling how immobilization is partitioned 
    "denit_min": 0.0, # minimum denitrificaiton rate 
    "denit_max": 0.005, # maximum denitrification rate
    "denit_thresh": 0.60, # soil moisture denitrification threshold
    "ADD": (1.5 / Zr) / 2, # carbon input rate to the litter pool (gC/m2/day)
    "CNadd": 58.0, # carbon to nitrogen ratio of added organic matter
    "ki_plus": 1.0, #NH4 immobilization coefficient, controls how quickly NH4 is immobilized into microbial biomass
    "ki_minus": 1.0, #NO3 immobilization coefficient, controls how quickly NO3 is immobilized into microbial biomass
    "a_plus": 0.05, #relative coefficient applied NH4 transport
    "a_minus": 1.0, #relative coefficient applied NO3 transport
    "DEM_plus": 0.2, #maximum/target plant demand for NH4
    "DEM_minus": 0.5, #maximum/target plant demand for NO3
    "F": 0.1, #scaling coefficient for active plant N uptake 
    "dd": 3, #exponent controlling how strongly active uptake responds to soil moisture
    "k_den": 0.1, #maximum denitrification rate coefficient, controls how quickly denitrification occurs
    "w": 2.0, #exponent controlling how strongly denitrification responds to soil moisture
    "Kmm": 10, #half saturation constant for denitrification, controls how quickly denitrification saturates with increasing nitrate concentration
    "fclay": 1, #scaling factor applied to humus decomposition. currently set to 1, so there is no reduction or enhancement 
}

INITIAL_CN_STATE = np.array([
    1200.0,  # Cl
    8500.0,  # Ch
    80.0,    # Cb
    20.0,    # Nl
    400.0,   # Nh
    7.0,     # Nb
    0.05,    # NH4
    1.0,     # NO3
], dtype=float)

# hydrology
# elevation of the land surface [m]
#this is the vertical reference datum
#elevations below the surface are negative 
z_datum = 0.0 
#elevation of the confining layer [m]
# the water table cannot drop below this elevation, it is the lower boundary of the soil column
z_cl = -5.5
#critical water table / capillary fringe elevation [m]
#used to determine when the low moisture / unsaturated zone forms 
#this separates the deep - water - table and shallow water table regimes 
z_cr = -1.0
#air entry pressure head [m]
#represents the pressure required for air to begin entering the largest soil pores 
psi_s = 0.20
#pressure head associated with field capacity
#calculated from the soil water retention relationship 
psi_fc = psi_s * (sfc ** (-b))
#soil specific yield 
#specific yield represents the fraction of groundwater storage released or filled when the water table changes elevation 
#used in the subsurface water table equation 
Sy_soil = n * (1.0 - sfc)
# characteristic/rooting depth used in the Laio water table ET equation 
#controls how rapidly groundwater supported ET decreases as the water table becomes deeper 
RD_upland = 0.4
# maximum depth of above ground ponded wetland water [m]
#once surface water depth exceeds this value, additional water can be treated as overflow/runoff 
#this is not calibrated/wetland specific yet 
WETLAND_STORAGE_MAX = 0.30 #meter
#initial above ground ponded water depth 
#the simulation currently starts with no standing water 
INITIAL_H_WET = 0.0 #meter
#denominator used to calculate A_upland 
# this comes from the Laio water table formulation and determinds the geometry/position 
# of the high moisture boundary 
den_A = (-psi_fc + psi_s - z_cr - 5.0 * RD_upland)
#dimensionless constant used in the equation for y_hm, the boundary between high and low moisture zones
#the conditional prevents division by zero 
A_upland = (-psi_fc + psi_s - z_cr) / den_A if abs(den_A) > 1e-12 else 1.0
#initial hydrologic conditions 
#initial relative soil saturation 
INITIAL_S = 0.2
#initial elevation of the high/low moisture zone boundary [m]
INITIAL_Y_HM = -0.5
#initial elevation of the top of the capillary fringe [m]
INITIAL_Y_CAP = -0.5 + psi_s
#initial water table elevation 
#-.5 means the simulation begins with the water table .5m below the land surface 
INITIAL_Y_WT = -0.5
