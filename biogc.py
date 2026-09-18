#biogeochemical module 
#this module represents coupled carbon (C) and nitrogen (N) cycling in the modeled soil/wetland system 
#the model tracks 8 state variables: 
#Cl = litter carbon 
#Ch = humus carbon 
#Cb = microbial biomass carbon 
#Nl = litter nitrogen
#Nh = humus nitrogen
#Nb = microbial biomass nitrogen
#NH4 = ammonium nitrogen
#NO3 = nitrate nitrogen
#Hydrology affects these processes primarily through:
#1. soil moisture (s_cur)
#2. leakage / drainage
#3. transpiration
# These hydrologic controls influence decomposition,
# nitrification, denitrification, leaching, and plant uptake.

from hydrology import leakage, transpiration

# Moisture modifiers
def fd(sval, cfg): 
    """
    Moisture modifier for decomposition 
    Returns a dimensionless value between approximately 0 and 1 describing 
    how favorable the current soil moisture is for decomposition. 
    Decomposition increases with moisture up to field capacity (sfc), then 
    decreases as the soil becomes wetter than field capacity. 
    Therefore decomposition is greatest near s = sfc 
    """
    #no decomposition if there is no soil water 
    if sval <= 0: return 0.0
    #decomposition increases linearly as soil moisture rises: 
    #fd(s) = s/sfc
    #once moisture exceeds field capacity, decomposition declines: 
    #fd(s) = sfc/s 
    #this represents increasingly unfavorable conditions for aerobic decomposition as the soil becomes very wet 
    return (sval / cfg.sfc) if sval <= cfg.sfc else (cfg.sfc / max(sval, 1e-12))

def fn(sval, cfg):  
    """
    Moisture modifier for nitrification. 
    Nitrification is an aerobic process, so it is suppressed both when soils are extremely
    dry and when soils become saturated. 
    The response therefore increases from dry conditions to field capacityu and then decreases toward zero at saturation. 
    """
    #No nitrification if there is no soil moisture 
    if sval <= 0: return 0.0
    #nitrification increases linearly with moisture until field capacity 
    if sval <= cfg.sfc: return sval / cfg.sfc
    #at complete saturatin, aerobic nitrification stops 
    if sval >= 1.0: return 0.0
    return 1.0 - (sval - cfg.sfc) / max(1.0 - cfg.sfc, 1e-12)

# ODE system (8 variables)
def coupled_rhs(t, y, s_cur, p, Tmax_t, cfg, Lrate_override=None, Trate_override=None):
    """
    Calculate the rates of change of all eight carbon and nitrogen state variables. 
    This function is passed to scipy's ODE solver. 
    Inputs
    t : 
        Time within the current integration step 
    y : 
        current vales of the eight C/N state variables 
    s_cur : 
        Current soil saturation from the hydrology model 
    p : 
        dictionary containing biogeochemical parameters 
    Tmax_t : 
        potential transpiration for the current timestep 
    cfg : 
        model parameter/configuration module 
    Lrate_override : 
        Hydrology derived leakage rate supplied by hydrology.py 
        If supplied, this replaces the older standalone leakage calculation
    Trate_override : 
        Hydrology derived transpiration rate supplied by hydrology.py 

    Returns 
    Rates of change for eight state variables: 
    [dCl, dCh, dCb, dNl, dNh, dNb, dNH4, dNO3]

    """
    Cl, Ch, Cb, Nl, Nh, Nb, NH4, NO3 = y

    #decomposition rate coefficients for litter and humus
    kl, kh = p["kl"], p["kh"]                         # m^3 d^-1 gC^-1
    #fraction of decomposed carbon lost through respiration
    rr     = p["rr"]                                  # -
    #c/n ratios of litter, humus, microbial biomass, and externally added material 
    CNl, CNh, CNb, CNadd = p["CNl"], p["CNh"], p["CNb"], p["CNadd"]

    #microbial biomass death/turnover rate [1/day]
    #the nested p.get allows compatibility with older parameter names 
    kd      = p.get("kd", p.get("km", 0.0))           # 1/d 
    #maximum fraction of decomposed litter carbon that can be transferred into the humus pool 
    rh_max  = p.get("rh_max", p.get("rh", 0.25))      # -
    #clay related scaling of humus decomposition
    fclay   = p.get("fclay", 1.0)                     # -
    #nitrification rate coefficient 
    kn      = p["kn"]                                  # m^3 d^-1 gC^-1

    #coefficients controlling microbial immobilization of NH4 and NO3 
    ki_plus  = p.get("ki_plus", 1.0)                  # m^3 d^-1 gC^-1
    ki_minus = p.get("ki_minus", 1.0)

    #coefficients controlling how strongly NH4 and NO3 participate in hydrologic transport/plant uptake 
    a_plus   = p.get("a_plus", p.get("a_NH4", 0.15))  # partition/coef for NH4+
    a_minus  = p.get("a_minus", p.get("a_NO3", 0.90)) # for NO3-

    #maximum/target plant N demand for NH4 and NO3
    DEM_plus  = p.get("DEM_plus", 0.0)                # gN m^-3 d^-1
    DEM_minus = p.get("DEM_minus", 0.0)
    #strength of active plant nutrient uptake 
    F         = p.get("F", 0.1)                       # uptake strength
    #moisture exponent controlling active uptake response 
    dd        = p.get("dd", 3)                        # moisture exponent
    #denitrification rate coefficient 
    k_den = p.get("k_den", 0.0)                       # 1/d
    #moisture response exponent for denitrification 
    w     = p.get("w", 2.0)                           # -
    #half saturation parameter controlling nitrate limitation of denitrification 
    Kmm   = p.get("Kmm", 10.0)                        # gN m^-3 

    # Moisture scalars
    #hydrology enters the biogeochemical model here through the current soil saturation s_cur 
    fd_s = fd(s_cur, cfg)            # for decomposition
    fn_s = fn(s_cur, cfg)            # for nitrification

    # Dynamic litter CN and humification coefficient
    #instead of assuming that litter always has exactly the parameter CNl, calculate the current 
    #litter C/N ratio from the state variables: 
    # CN_lit = Cl / Nl 
    #this means litter quality can change through time 
    CN_lit = Cl / max(Nl, 1e-12)
    #rh_eff controls how much decomposed litter carbon is transferred into the humus pool 
    #rh_eff = min (rh_max, CNh/(CN_lit))
    #this prevents humification from exceeding rh_max and also allows litter C/N quality 
    #to constrain humification 
    rh_eff = min(rh_max, CNh / max(CN_lit, 1e-12))

    #potential net N mineralization / immobilization 
    # phi_den describes the N balance associated with microbial decomposition of humus and litter. 
    #its sign helps determine whether decomposition results in positive N release -> mineralization or 
    # microbial N demand -> immobilization 
    phi_den = kh*Ch*(1.0/CNh - (1.0 - rr)/CNb) \
            + kl*Cl*(1.0/CN_lit - rh_eff/CNh - (1.0 - rh_eff - rr)/CNb)

    # N availability limiter phi_small 
    # available mineral N is represented using NH4 and NO3 
    #phi_num = -(ki+ * NH4 + ki- * NO3)
    phi_num = -(ki_plus*NH4 + ki_minus*NO3)
    #if phi_den is positive, decomposition releases enough nitrogen rather than requiring additional mineral N
    #therefore decomposition is not N limited 
    if phi_den >= 0.0:
        phi_small = 1.0
    else:
        #if decomposition requires immobilization, microbial activity may be limited by the amount of available mineral nitrogen 
        phi_small = phi_num / max(phi_den, -1e-12)
        phi_small = max(0.0, min(1.0, phi_small))  # clamp to [0,1]

 # net mineralization minus immobilization
 # PHI > 0 -> net mineralization 
 # PHI < 0 -> net immobilization
    PHI = phi_small * fd_s * Cb * phi_den 

    # Decomposition 
    #DECl increases with litter carbon, microbial biomass, decomposition coefficient, favorable moisture 
    # and may be reduced by N limitation (phi_small)
    DECl = phi_small * fd_s * kl * Cb * Cl
    # humus decomposition 
    #same as litter, but using humus decomposition coefficient and clay scaling factor 
    DECh = fclay    * phi_small * fd_s * kh * Cb * Ch

    # Microbial death
    #more microbial biomass produces greater microbial turnover 
    BD = kd * Cb

    # Carbon balances 
    #litter carbon : gains external carbon addition + dead microbial biomass 
    # loss : litter decomposition 
    dCl = p["ADD"] + BD - DECl
    #humus carbon 
    # a fraction of rh_eff of decomposed litter is humified and enters the humus pool 
    #humus is lost through decomposition 
    dCh = rh_eff * DECl - DECh
    #microbial biomass carbon 
    #microbes gain carbon from litter and humus decomposition after accounting for respiration 
    #they lose carbon through microbial death 
    dCb = (1.0 - rh_eff - rr) * DECl + (1.0 - rr) * DECh - BD

    # MIN / IMM 
    #mineralization : organic N -> mineral N 
    #immobilization : mineral N -> microbial/organic N 
    #these are opposite directions of N transfer 
    #maximum amount of mineral N that can potentially be immobilized given the current NH4 and NO3 availability,
    #moisture, and microbial biomass 
    IMM_max = (ki_plus*NH4 + ki_minus*NO3) * fd_s * Cb
    #positive PHI means net N release 
    if PHI > 0.0:
        MIN = PHI
        IMM = 0.0
    else: #negative PHI means microbes require mineral N 
        MIN = 0.0
        #immobilization cannot exceed the available capacity 
        IMM = min(-PHI, IMM_max)
#microbial immobilization can draw from both mineral N pools 
#the relative contribution depends on ki_plus * NH4, ki_minus * NO3
    den_imm = max(ki_plus*NH4 + ki_minus*NO3, 1e-12)
    #amount of immobilization supplied by NH4 
    IMM_NH4 = IMM * (ki_plus*NH4  / den_imm)
    #amount supplied by NO3
    IMM_NO3 = IMM * (ki_minus*NO3 / den_imm)

    # Hydrologic scalings 
    #connection between hydrology.py and biogc.py
    #hydrologic water fluxes are converted into rates that can act on nutrient concentrations 
    #approximate volume/depth of water represented in the root-zone soil reservoir 
    # water_vol = s *n * Zr 
    # s = soil saturation 
    # n = porosity 
    # Zr = root zone depth 
    water_vol = max(s_cur * cfg.n * cfg.Zr, 1e-12)
    #leakage rate 
    # if hydrology.py supplies the timestep leakage rate, use that 
    # otherwise, fall back to the older standalone leakage() function 
    if Lrate_override is None:
        Lrate = leakage(s_cur, cfg) / water_vol
    else:
        Lrate = Lrate_override
    #same for transpiration
    if Trate_override is None:
        Trate = transpiration(s_cur, Tmax_t, cfg) / water_vol
    else:
        Trate = Trate_override

    # leaching 
    #leakage moving through the soil can carry dissolved mineral nitrogen out of the modeled soil reservoir 
    #leaching increases with water drainage rate x nutrient concentration 
    #NH4 and NO3 are scaled separately using a_plus and a_minus
    LE_NH4 = a_plus  * Lrate * NH4
    LE_NO3 = a_minus * Lrate * NO3

    # passive and active uptake
    #passive uptake occurs as plants take up water through transpiration 
    #therefore nutrient uptake is linked directly to the transpiration rate 
    UP_p_NH4 = a_plus  * Trate * NH4
    UP_p_NO3 = a_minus * Trate * NO3
    #plants may also actively acquire nutrients rather than obtaining them only transpiration 
    #active uptake depends on nutrient concentration, soil moisture, uptake strength parameter F, remaining plant N demand 
    ku_plus  = a_plus  * F * (s_cur**dd) / water_vol
    ku_minus = a_minus * F * (s_cur**dd) / water_vol
    #remaining NH4 demand after passive uptake 
    dem_p   = max(DEM_plus  - UP_p_NH4, 0.0)
    #remaining nO3 demand after passive uptake 
    dem_m   = max(DEM_minus - UP_p_NO3, 0.0)
    #active uptake cannot exceed the remaining plant demand 
    UP_a_NH4 = min(ku_plus  * NH4, dem_p)
    UP_a_NO3 = min(ku_minus * NO3, dem_m)
    #total plant uptake = passive + active 
    UP_NH4 = UP_p_NH4 + UP_a_NH4
    UP_NO3 = UP_p_NO3 + UP_a_NO3

    # nitrificatioin
    #converts NH4+ -> NO3-
    # it is an aerobic microbial process 
    #therefore it depends on available NH4, microbial biomass, nitrification coefficient kn, aerobic moisture suitability fn_s 
    NIT = kn * fn_s * Cb * NH4

    # denitrification 
    #denitrification removes nitrate under wet/oxygen poor conditions 
    # it is controlled by soil moisture, nitrate availability, denitrification rate coefficient in this model 
    #below field capacity, denitrification is turned off
    if s_cur <= cfg.sfc:
        fs_den = 0.0
    else: #above field capacity, denitrification increases nonlinearly as saturation approaches 1 
        #fs_den = ((s-sfc)/(1-sfc))^w
        fs_den = ((s_cur - cfg.sfc) / max(1.0 - cfg.sfc, 1e-12)) ** w
    #nitrate limitation: fN_den = (NO3/(Kmm + NO3))
    #when NO3 is scarce, fn_den approaches 0
    #when NO3 is abundant, fn_den approaches 1 
    fN_den = NO3 / (Kmm + NO3)   
    #actual denitrification rate: DENIT = maximum rate x nitrate limitation x moisture limitation x nitrate concentration 
    DENIT = k_den * fN_den * fs_den * NO3

    # nitrgen balances
    #litter nitrogen gains + N associated with external litter addition + N returned through microbial death 
    # loss - N associated with litter decomposition 
    dNl  = (p["ADD"] / CNadd) + (BD / CNb) - (DECl / CN_lit)
    #humus nitrogen 
    #humification transfer N into the humus pool 
    #humus decomposition removes N from it 
    dNh  = (rh_eff * DECl - DECh) / CNh
    #Microbial biomass nitrogen tracks N associeted with microbial growth and turnover, including PHI (mineralization/immobilization)
    dNb  = (1.0 - rh_eff * (CN_lit / CNh)) * (DECl / CN_lit) \
           + (DECh / CNh) - (BD / CNb) - PHI
    #ammonium balance gains + mineralization, losses : immobilization, nitrification, leaching, plant uptake 
    dNH4 = MIN - IMM_NH4 - NIT - LE_NH4 - UP_NH4
    #Nitrate balance gains + nitrification, losses: immobilization, leaching, plant uptake, denitrification 
    dNO3 = NIT - IMM_NO3 - LE_NO3 - UP_NO3 - DENIT
    #return derivatives to the ODE solver
    #these are not the new pool values themselves, they are the instantaeous rates of change: d(state variable / dt)
    #scipy's solve_ivp() uses these rates to integrate the C/N pools through the current model time step 
    return [dCl, dCh, dCb, dNl, dNh, dNb, dNH4, dNO3]
