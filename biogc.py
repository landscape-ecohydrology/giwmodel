from hydrology import leakage, transpiration

# Moisture modifiers
def fd(sval, cfg):  # for decomposition
    if sval <= 0: return 0.0
    return (sval / cfg.sfc) if sval <= cfg.sfc else (cfg.sfc / max(sval, 1e-12))

def fn(sval, cfg):  # for nitrification (aerobic)
    if sval <= 0: return 0.0
    if sval <= cfg.sfc: return sval / cfg.sfc
    if sval >= 1.0: return 0.0
    return 1.0 - (sval - cfg.sfc) / max(1.0 - cfg.sfc, 1e-12)

# ODE system (8 variables)
def coupled_rhs(t, y, s_cur, p, Tmax_t, cfg, Lrate_override=None, Trate_override=None):
    # State (all in g m^-3)
    Cl, Ch, Cb, Nl, Nh, Nb, NH4, NO3 = y

    
    kl, kh = p["kl"], p["kh"]                         # m^3 d^-1 gC^-1
    rr     = p["rr"]                                  # -
    CNl, CNh, CNb, CNadd = p["CNl"], p["CNh"], p["CNb"], p["CNadd"]

    kd      = p.get("kd", p.get("km", 0.0))           # 1/d 
    rh_max  = p.get("rh_max", p.get("rh", 0.25))      # -
    fclay   = p.get("fclay", 1.0)                     # -
    kn      = p["kn"]                                  # m^3 d^-1 gC^-1

    ki_plus  = p.get("ki_plus", 1.0)                  # m^3 d^-1 gC^-1
    ki_minus = p.get("ki_minus", 1.0)

    a_plus   = p.get("a_plus", p.get("a_NH4", 0.15))  # partition/coef for NH4+
    a_minus  = p.get("a_minus", p.get("a_NO3", 0.90)) # for NO3-

    DEM_plus  = p.get("DEM_plus", 0.0)                # gN m^-3 d^-1
    DEM_minus = p.get("DEM_minus", 0.0)
    F         = p.get("F", 0.1)                       # uptake strength
    dd        = p.get("dd", 3)                        # moisture exponent

    k_den = p.get("k_den", 0.0)                       # 1/d
    w     = p.get("w", 2.0)                           # -
    Kmm   = p.get("Kmm", 10.0)                        # gN m^-3 

    # Moisture scalars
    fd_s = fd(s_cur, cfg)            # for decomposition
    fn_s = fn(s_cur, cfg)            # for nitrification

    # Dynamic litter CN and humification coefficient
    CN_lit = Cl / max(Nl, 1e-12)
    rh_eff = min(rh_max, CNh / max(CN_lit, 1e-12))

    
    # phi_den 
    phi_den = kh*Ch*(1.0/CNh - (1.0 - rr)/CNb) \
            + kl*Cl*(1.0/CN_lit - rh_eff/CNh - (1.0 - rh_eff - rr)/CNb)

    # φ (dimensionless limiter) and PHI = MIN - IMM (gN m^-3 d^-1)
    phi_num = -(ki_plus*NH4 + ki_minus*NO3)
    if phi_den >= 0.0:
        phi_small = 1.0
    else:
        phi_small = phi_num / max(phi_den, -1e-12)
        phi_small = max(0.0, min(1.0, phi_small))  # clamp to [0,1]

    PHI = phi_small * fd_s * Cb * phi_den  # net mineralization minus immobilization

    # Decomposition 
    DECl = phi_small * fd_s * kl * Cb * Cl
    DECh = fclay    * phi_small * fd_s * kh * Cb * Ch

    # Microbial death
    BD = kd * Cb

    # Carbon balances 
    dCl = p["ADD"] + BD - DECl
    dCh = rh_eff * DECl - DECh
    dCb = (1.0 - rh_eff - rr) * DECl + (1.0 - rr) * DECh - BD

    # MIN / IMM 
    IMM_max = (ki_plus*NH4 + ki_minus*NO3) * fd_s * Cb
    if PHI > 0.0:
        MIN = PHI
        IMM = 0.0
    else:
        MIN = 0.0
        IMM = min(-PHI, IMM_max)

    den_imm = max(ki_plus*NH4 + ki_minus*NO3, 1e-12)
    IMM_NH4 = IMM * (ki_plus*NH4  / den_imm)
    IMM_NO3 = IMM * (ki_minus*NO3 / den_imm)

    # Hydrologic scalings 
    
    water_vol = max(s_cur * cfg.n * cfg.Zr, 1e-12)
    if Lrate_override is None:
        Lrate = leakage(s_cur, cfg) / water_vol
    else:
        Lrate = Lrate_override
    if Trate_override is None:
        Trate = transpiration(s_cur, Tmax_t, cfg) / water_vol
    else:
        Trate = Trate_override

    # leaching 
    LE_NH4 = a_plus  * Lrate * NH4
    LE_NO3 = a_minus * Lrate * NO3

    # passive and active uptake
    UP_p_NH4 = a_plus  * Trate * NH4
    UP_p_NO3 = a_minus * Trate * NO3

    ku_plus  = a_plus  * F * (s_cur**dd) / water_vol
    ku_minus = a_minus * F * (s_cur**dd) / water_vol

    dem_p   = max(DEM_plus  - UP_p_NH4, 0.0)
    dem_m   = max(DEM_minus - UP_p_NO3, 0.0)
    UP_a_NH4 = min(ku_plus  * NH4, dem_p)
    UP_a_NO3 = min(ku_minus * NO3, dem_m)

    UP_NH4 = UP_p_NH4 + UP_a_NH4
    UP_NO3 = UP_p_NO3 + UP_a_NO3

    # nitrificatioin
    NIT = kn * fn_s * Cb * NH4

    # denitrification 
    if s_cur <= cfg.sfc:
        fs_den = 0.0
    else:
        fs_den = ((s_cur - cfg.sfc) / max(1.0 - cfg.sfc, 1e-12)) ** w
    fN_den = NO3 / (Kmm + NO3)   
    DENIT = k_den * fN_den * fs_den * NO3

    # nitrgen balances
    dNl  = (p["ADD"] / CNadd) + (BD / CNb) - (DECl / CN_lit)
    
    dNh  = (rh_eff * DECl - DECh) / CNh
    dNb  = (1.0 - rh_eff * (CN_lit / CNh)) * (DECl / CN_lit) \
           + (DECh / CNh) - (BD / CNb) - PHI
    dNH4 = MIN - IMM_NH4 - NIT - LE_NH4 - UP_NH4
    dNO3 = NIT - IMM_NO3 - LE_NO3 - UP_NO3 - DENIT

    return [dCl, dCh, dCb, dNl, dNh, dNb, dNH4, dNO3]
