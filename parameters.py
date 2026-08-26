import numpy as np

# soil parameters
soil_types = {
    "sand":     {"n": 0.43, "sfc": 0.17, "sw": 0.05,  "sh": 0.02,  "Ks": 0.20,  "b": 4.05},
    "loam":     {"n": 0.43, "sfc": 0.27, "sw": 0.12,  "sh": 0.02,  "Ks": 0.03,  "b": 4.38},
    "clay":     {"n": 0.50, "sfc": 0.40, "sw": 0.27,  "sh": 0.02,  "Ks": 0.005, "b": 11.4},
    "silt":     {"n": 0.50, "sfc": 0.30, "sw": 0.15,  "sh": 0.02,  "Ks": 0.01,  "b": 7.75},
    "original": {"n": 0.40, "sfc": 0.30, "sw": 0.065, "sh": 0.02,  "Ks": 1.1,   "b": 4.38},
}

selected_soil = "original"
soil = soil_types[selected_soil]
n, sfc, sw, sh, Ks, b = soil["n"], soil["sfc"], soil["sw"], soil["sh"], soil["Ks"], soil["b"]

# switches
USE_SEASONAL_PET    = True
PLOT_PAPER_FIGS     = True
PLOT_WATER_BLOCK    = False
PLOT_CARBON_BLOCK   = False
PLOT_NITROGEN_BLOCK = False

# PET
PET_MIN     = 5e-5
PET_SPLIT_T = 0.55
PET_PHASE   = 0.0

# vegetation/depth
Zr = 0.80
s_star = 0.17
# PET target
ANNUAL_PET = 1.0                  # m/year
BASELINE_PET_DAILY = ANNUAL_PET / 365.0

# Fraction allocated to potential transpiration
PET_SPLIT_T = 0.55

# rain
alpha = 0.011 * 10
lambda_inv = 1 / 0.23
RAIN_SEED = 0

# time
years, dt = 20, 0.5
T = years * 365
N = int(T / dt)
time = np.arange(0, T, dt)

# carbon and nitrogen 
params = {
    "CNl": 58.0,
    "CNh": 22.0,
    "CNb": 11.5,
    "kl": 0.000065,
    "kh": 0.0000025,
    "kd": 0.0085,
    "rh_max": 0.25,
    "rr": 0.60,
    "kn": 0.6,
    "up_pass": 1.0,
    "up_frac_NH4": 0.4,
    "imm_bias": 0.6,
    "denit_min": 0.0,
    "denit_max": 0.005,
    "denit_thresh": 0.60,
    "ADD": (1.5 / Zr) / 2,
    "CNadd": 58.0,
    "ki_plus": 1.0,
    "ki_minus": 1.0,
    "a_plus": 0.05,
    "a_minus": 1.0,
    "DEM_plus": 0.2,
    "DEM_minus": 0.5,
    "F": 0.1,
    "dd": 3,
    "k_den": 0.1,
    "w": 2.0,
    "Kmm": 10,
    "fclay": 1,
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
z_datum = 0.0
z_cl = -5.5
z_cr = -1.0
psi_s = 0.20
psi_fc = psi_s * (sfc ** (-b))
Sy_soil = n * (1.0 - sfc)
RD_upland = 0.4
WETLAND_STORAGE_MAX = 0.30 #meter
INITIAL_H_WET = 0.0 #meter
den_A = (-psi_fc + psi_s - z_cr - 5.0 * RD_upland)
A_upland = (-psi_fc + psi_s - z_cr) / den_A if abs(den_A) > 1e-12 else 1.0

INITIAL_S = 0.2
INITIAL_Y_HM = -0.5
INITIAL_Y_CAP = -0.5 + psi_s
INITIAL_Y_WT = -0.5
