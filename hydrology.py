#hydrology module 
#this file contains the hydrologic portion of the model. 
#the main processes represented here are: 
#1. soil moisture dependent evaporation/transpiration functions 
#2. drainage/leakage from wet soil 
#3. fluctuating water table dynamics based on Laio framework 
#4. low moisture zone water storage 
#5. recharge between the LMZ and saturated zone 
#6. ET supplied by the water table 
#7. saturation excess runoff 
#8. above ground wetland/ponded water storage 


import numpy as np

# soil moisture dependent evaporation
def evaporation(sval, Ew_t, cfg):
    """
    Calculate evaporation as a function of soil saturation. 
    Parameters
    sval : float 
        Current relative soil saturation
        
    Ew_t : float 
        Potential evaporation rate for the current time [m/day]
        
    cfg : module
        Model parameters imported from parameters.py
        
    Returns: 
    float
        Actual evaporation rate [m/day]
    """
    #if soil saturation is below the hygroscopic threshold, the soil is considered too dry for evaporation
    if sval <= cfg.sh:
        return 0.0
    #between the hygroscopic point and wilting point, evaporation increases linearly with soil moisture
    if sval <= cfg.sw:
        return (Ew_t * (sval - cfg.sh) / max(cfg.sw - cfg.sh, 1e-12))
    #above the wilting-point threshold, evaporation occurs at its full potential rate
    return Ew_t

# soil moisture dependent transpiration
def transpiration(sval, Tmax_t, cfg):
    """
    Calculate plant transpiration as a function of soil saturation. 
    
    Parameters
    sval : float 
        Current relative soil saturation
    
    Tmax_t : float 
        Potential transpiration rate [m/day]
        
    cfg : module 
        Model parameters 
        
    Returns: 
    float 
        Actual transpiration rate [m/day]
    """
    # below the wilting point, plants cannot extract enough water from the soil, so transpiration is zero
    if sval <= cfg.sw:
        return 0.0
    #between the wilting point and s_star, transpiration increases linearly with soil saturation 
    if sval <= cfg.s_star:
        return Tmax_t * (sval - cfg.sw) / max(cfg.s_star - cfg.sw, 1e-12)
    #above s_star, plants are not water limited and transpiration occurs at its potential rate 
    return Tmax_t

#soil leakage/drainage function 
def leakage(sval, cfg):
    """
    Calculate drainage/leakage as a function of soil saturation. 
    The function increases strongly as the soil becomes wetter 
    
    Parameters
    sval : float 
        Current relative soil saturation 
        
    cfg : module 
        Model parameters 
        
    Returns: 
    float 
        Leakage rate, constrained to be non-negative 
    """
    #soil dependent exponent controlling how sharply drainage increases with increasing saturation 
    beta = 2 * cfg.b + 4
    # numerator of the nonlinear drainage relationship 
    num = np.exp(beta * (sval - cfg.sfc)) - 1.0
    #denominator normalizes the relationship so that leakage approaches Ks near full saturation
    den = np.exp(beta * (1.0 - cfg.sfc)) - 1.0
    #calculate leakage using saturated hydraulic conductivity Ks 
    val = cfg.Ks * (num / max(den, 1e-12))
    #leakage cannot be negative
    return max(0.0, val)


def initialize_hydrology(N, cfg):
    """
    Allocate and initialize all hydrologic state and flux arrays.
    
    Each array has length N so the value of every hydrologic state or flux can be 
    saved at every model timestep 
    """
    #dictionary containing all hydrologic state variables and fluxes 
    hydro = {
        #soil saturation variables
        "s": np.zeros(N), # soil saturation passed to C/N model 
        "s_ex": np.zeros(N), # soil saturation before final clipping. "ex" indicates that this 
        #variable can temporarily contain water in excess of saturation before excess water is removed 
        "s_lim": np.zeros(N), # soil saturation limited to field capacity. used when calculating moisture limited ET 
        "ds": np.zeros(N), # changed in relative soil saturation during each timestep 
        "lmz_stor": np.zeros(N), #low moisture zone storage capacity / deficit [m]
        #water table geometry variables 
        "y_hm": np.full(N, cfg.INITIAL_Y_HM), #elevation of the boundary between the low moisture and high moisture zones [m] 
        "y_cap": np.full(N, cfg.INITIAL_Y_CAP), # elevation of the top of the capillary fringe [m]
        "y_wt": np.full(N, cfg.INITIAL_Y_WT), # water table elevation [m] z_datum = 0 is the land surface, so y_wt = -.5 means
        # WT is -.5 below ground 
        #hydrologic fluxes 
        "P_flux": np.zeros(N), #precipitation entering the model [m/step]
        "PET_flux": np.zeros(N), # potential evapotranspiration [m/step]
        "PET_lm_flux": np.zeros(N), # remaining PET allocated to the low moisture zone after water table ET has been calculated [m/step]
        "ET_lm_flux": np.zeros(N),# actual ET from the low moisture zone [m/step]
        "ET_wt_flux": np.zeros(N), #ET supplied directly by the water table [m/step]
        "R_flux": np.zeros(N), # recharge sent directly to the saturated zone [m/step]
        "R_lm_flux": np.zeros(N), # rainfall/recharge entering the low moisture zone [m/step]
        "L_lm_flux": np.zeros(N), #leakage from the low moisture zone into the saturated zone [m/step]
        "ExfilSat_flux": np.zeros(N), #upward exfiltration from the saturated zone toward the low moisture zone [m/step]
        "runoff_flux": np.zeros(N), # water leaving the modeled system as runoff / overflow [m/step]
        # Above-ground wetland storage
        "h_wet": np.zeros(N), # depth of water ponded above the soil surface [m], h_wet = 0 means no standing water
        "ET_surface_flux": np.zeros(N), # et removed directly from ponded surface water [m/step]
        "wetland_input_flux": np.zeros(N), # water supplied to the above ground wetland storage pool 
        #storage diagnostics 
        "storage": np.zeros(N), # simplified soil water storage calculation [m] (this uses fixed root depth Zr, the 
        # whole system water balance diagnostic later uses a more explicit LMZ + groundwater storage calculation)
        "total_storage": np.zeros(N), # soil storage + above ground wetland storage [m]
    }

    #initial conditions
    # set the starting depth of ponded wetland water  
    hydro["h_wet"][0] = cfg.INITIAL_H_WET
    #set initial soil saturation
    hydro["s"][0] = cfg.INITIAL_S
    # initially, s_ex is also equal to the specific starting soil saturation
    hydro["s_ex"][0] = cfg.INITIAL_S
    #s_lim cannot exceed field capacity 
    hydro["s_lim"][0] = min(cfg.INITIAL_S, cfg.sfc)
    #calculate initial simplified soil water storage [m]
    #saturation x porosity x soil/root zone depth 
    hydro["storage"][0] = cfg.INITIAL_S * cfg.n * cfg.Zr
    #total initial storage includes soil water plus any above ground wetland water. 
    hydro["total_storage"][0] = hydro["storage"][0] + hydro["h_wet"][0]
    #return all initialized hydrologic arrays 
    return hydro


def hydrology_step(t, P_step, PET_step, hydro, cfg):
    """
    Advance the fluctuating-water-table hydrology by one step.
    
    Parameters 
    t : int 
        Current timestep index 
        
    P_step : float 
        Precipitation during this timestep [m/step]
        
    PET_step : float 
        Potential ET available during this timestep [m/step]
        
    hydro : dict 
        Dictionary containing all hydrologic states and fluxes 
        
    cfg : module 
        Model parameters 
    
    Returns: 
    dict
        Hydrologic variables needed by the C/N model
        
    """
    #values from the previous timestep 

    #soil saturation from previous timestep
    s_prev = hydro["s_ex"][t - 1]
    #previous soil saturation capped at field capacity 
    s_lim_prev = hydro["s_lim"][t - 1]
    #previous capillary fringe elevation [m]
    y_cap_prev = hydro["y_cap"][t - 1]
    #previous water table elevation [m]
    y_wt_prev = hydro["y_wt"][t - 1]
    #previous above ground ponded water depth [m]
    h_wet_prev = hydro["h_wet"][t - 1]

    # If ponded water is already present, rainfall falls directly onto the
    # wetland surface and PET evaporates ponded water before drawing from
    # soil/groundwater.
    # P_step is total rainfall arriving from atmosphere
    surface_rain_step = P_step if h_wet_prev > 0.0 else 0.0 #otherwise there is no direct rainfall input to h_wet
    # if not ponded, rainfall goes to soil/subsurface hydrology, preventing rainfall from being counted in both pools 
    P_soil_step = 0.0 if h_wet_prev > 0.0 else P_step
    # calculate ET directly from ponded surface water 
    #surface ET cannot exceed: 
    #1. atmospheric PET demand
    #2. amount of surface water actually available 
    ET_surface_step = min(
        PET_step,
        h_wet_prev + surface_rain_step
    )
    #surface water remaining after rainfall and surface ET, before considering new groundwater derived surface water or overflow 
    # surface water remaining = previous surface water + rain - surface ET 
    surface_water_preoverflow = (
        h_wet_prev
        + surface_rain_step
        - ET_surface_step
    )
    #pet remaining after surface water ET is satisfied 
    # this remaining atmospheric demand can potentially be supplied by the soil or water table 
    PET_soil_step = max(
        PET_step - ET_surface_step,
        0.0
    )
    #start all hydrologic fluxes at zero. their actual values will be calculated below 
    #recharge entering the saturated zone 
    R_step = 0.0
    #rainfall/recharge entering the low moisture zone 
    R_lm_step = 0.0
    #leakage from LMZ to saturated zone 
    L_lm_step = 0.0
    #et supplied by the low moisture zone 
    ET_lm_step = 0.0
    #et supplied directly by groundwater
    ET_wt_step = 0.0
    #upward exfiltration from saturated zone 
    ExfilSat_step = 0.0
    #PET available to the low moisture zone 
    PET_lm_step = 0.0
    #rainfall that exceeds available LMZ storage 
    excess_R = 0.0
    #total runoff/overflow generated this timestep 
    h_runoff_step = 0.0

    # Case 1: deep water table; unsaturated/low-moisture zone forms
    #when the top of the capillary fringe is sufficiently below the critical depth z_cr, 
    # an unsaturated / low moisture zone forms above it
    if y_cap_prev < cfg.z_cr - 0.01:
        #threshold used in the Laio equation for determining the location of the high/low moisture boundary 
        #threshold = -5RD - pressure head associated with field capacity - air entry pressure head 
        threshold = -5.0 * cfg.RD_upland - cfg.psi_fc + cfg.psi_s
        #calculate y_hm, the elevation of the boundary between the low moisture zone and high moisture zone. 
        #its position depends on the depth of the capillary fringe 
        # if the capillary fringe is sufficiently deep, calculate y_hm = y_cap + psi_fc - psi_s
        if y_cap_prev <= threshold:
            hydro["y_hm"][t] = y_cap_prev + cfg.psi_fc - cfg.psi_s
        #when the capillary fringe is at an intermediate depth, a nonlinear laio relationship is used. 
        # this allows the low moisture zone geometry to change as the capillary fringe approaches the critical depth z_cr 
        else: # intermediate capillary fringe depth 
            #define the denominator used in the nonlinear relationship
            den_hm = (-cfg.z_cr - cfg.psi_fc + cfg.psi_s)
            #no division by zero
            if abs(den_hm) < 1e-12:
                den_hm = np.sign(den_hm) * 1e-12 if den_hm != 0 else 1e-12
            #intermediate depth Laio relationship 
            # y_hm = (1 - A^(3/4))(y_cap - z_cr) - [A^2(1-A^(-1/4))/D] (y_cap-z_cr)^2
            # where A = A_upland and D = -z_cr - psi_fc +psi_s 
            hydro["y_hm"][t] = (
                (1.0 - cfg.A_upland**0.75) * (y_cap_prev - cfg.z_cr)
                - cfg.A_upland**2.0
                * (1.0 - cfg.A_upland**(-0.25))
                / den_hm
                * (y_cap_prev - cfg.z_cr) ** 2.0
            )
        #Groundwater supported ET decreases exponentially as the capillary fringe becomes deeper 
        #shallower groundwater: exp(y_cap / RD) is larger -> more groundwater ET 
        #deeper groundwater: exp(y_cap / RD) becomes very small -> little WT ET 
        ET_wt_step = PET_soil_step * np.exp(
            y_cap_prev / cfg.RD_upland
        )
        #in the deep water table case, rainfall does not immediately recharge the saturated zone directly 
        R_step = 0.0
        #Low moisture zone depth 
        #calculate LMZ depth from y_hm 
        lmz_depth = max(abs(hydro["y_hm"][t]), 1e-12)
        #estimate how much additional water could be stored in the LMZ before reaching saturation 
        #porosity x depth x unfilled fraction 
        hydro["lmz_stor"][t] = max(
            0.0, cfg.n * lmz_depth * (1.0 - s_prev)
        )
        #whatever PET was not supplied by groundwater can potentially be supplied by the low moisture zone 
        PET_lm_step = max(
            PET_soil_step - ET_wt_step,
            0.0
        )
        #if soil moisture is above field capacity, the LMZ can satisfy all remaining PET demand 
        if s_prev > cfg.sfc:
            ET_lm_step = PET_lm_step
        #between wilting point and field capacity, ET becomes moisture limited 
        elif s_prev > cfg.sw:
            ET_lm_step = max(
                0.0,
                PET_lm_step
                * (s_lim_prev - cfg.sw)
                / max(cfg.sfc - cfg.sw, 1e-12),
            )
        #at or below the wilting point, no ET is supplied by the LMZ 
        else:
            ET_lm_step = 0.0

        #rainfall entering the LMZ 
        #current assumption: all rainfall reachign the soil is initially treated as recharge/input to the low moisture zone 
        #this is why rainfall and infiltration look nearly identical in my plots... 
        R_lm_step = P_soil_step
        #water above field capacity drains from the LMZ toward the saturated zone 
        #if s_prev <= sfc, this value becomes zero 
        L_lm_step = max(
            0.0,
            (s_prev - cfg.sfc) * cfg.n * lmz_depth,
        )
        #calculate upward transfer from the saturated zone associated with ET demand and the locations of y_hm/y_cap 
        ExfilSat_step = PET_soil_step * (
            np.exp(hydro["y_hm"][t] / cfg.RD_upland)
            - np.exp(y_cap_prev / cfg.RD_upland)
        )
        #calculate rainfall that is larger than the currently available LMZ storage capacity 
        excess_R = max(
            0.0,
            P_soil_step - hydro["lmz_stor"][t]
        )
        #if groundwater reaches the lower boundary of the model, no more groundwater can be extracted from below 
        if y_wt_prev <= cfg.z_cl:
            ET_wt_step = 0.0 #stop groundwater ET 
            R_step = P_soil_step #rainfall is instead treated as recharge to the groundwater system, allowing WT to recover upward 
            ExfilSat_step = 0.0 #no upward exfiltration is allowed 

    # Case 2: shallow water table; saturated/wetland-like condition
    # when the capillary fringe rises above the critical depth z_cr, the model no longer represents a separate low moisture zone 
    # reservoir. ground water/capillary influence is assumed to extend sufficiently close to the surface that the LMZ formulation
    # used in the deep WT case is no longer applied  
    else:
        #all remaining PET is supplied directly by the water table 
        ET_wt_step = PET_soil_step
        #rainfall directly recharges the saturated system 
        R_step = P_soil_step
        #there is no distinct high/low moisture boundary 
        hydro["y_hm"][t] = 0.0
        #because ET_wt equals all available PET in this regime, 
        # this term will normally equal zero 
        ExfilSat_step = PET_soil_step - ET_wt_step
        #no lmz exists, so there is no leakage from it 
        L_lm_step = 0.0
        #no separate rainfall excess is calculated here 
        excess_R = 0.0
        #no PET is assigned to an LMZ because the LMZ does not exist 
        PET_lm_step = 0.0

    # Water table and capillary fringe update
    #the water balance is 
    # inputs : R_step, L_lm_step 
    # outputs : ET_wt_step, ExfilSat_step
    # dividing by specific yield converts a water depth change into a change in groundwater elevation 
    hydro["y_cap"][t] = max(
        #do not allow the capillary fringe to fall below the confining layer 
        cfg.z_cl,
        y_cap_prev
        + (R_step + L_lm_step - ET_wt_step - ExfilSat_step)
        / max(cfg.Sy_soil, 1e-12),
    )
    #calculate water table elevation from the capillary fringe elevation 
    hydro["y_wt"][t] = hydro["y_cap"][t] - cfg.psi_s

    #check for water reaching above the land surface 
    #water generated because the groundwater/capillary system rises beyond the surface 
    surface_excess_step = 0.0
    #water generated because the LMZ becomes more than fully saturated 
    saturation_excess_step = 0.0

    # Water table rises above land surface
    if hydro["y_wt"][t] > cfg.z_datum:
        #convert the amount by which the groundwater state exceeds the surface into an equivalent water depth 
        #specific yield is used here because this excess was calculated from a subsurface water table rise 
        surface_excess_step += (
            hydro["y_wt"][t]
            - cfg.z_datum
            + excess_R
        ) * cfg.Sy_soil
        #once groundwater reaches the surface, do not allow the subsurface water table elevation itself to remain above z_datum 
        hydro["y_wt"][t] = cfg.z_datum


    # Capillary fringe rises above land surface
    if hydro["y_cap"][t] > cfg.z_datum:
        #convert excess capillary fringe elevation into a surface water equivalent 
        surface_excess_step += (
            hydro["y_cap"][t]
            - cfg.z_datum
            + excess_R
        ) * cfg.Sy_soil
        #cap the capillary fringe at the land surface 
        hydro["y_cap"][t] = cfg.z_datum

    # Low-moisture-zone water content update
    #if a low moisture zone exists...
    if hydro["y_hm"][t] != 0:
        #calculate change in relative saturation
        #inputs: R_lm_step, ExfilSat_step 
        #outputs: ET_lm_step, L_lm_step 
        #divide by LMZ depth x porosity to convert water depth into change in relative saturation 
        hydro["ds"][t] = (
            R_lm_step + ExfilSat_step - ET_lm_step - L_lm_step
        ) / abs(hydro["y_hm"][t]) / cfg.n
    else: #if there is no low moisture zone, there is no separate LMZ saturation update 
        hydro["ds"][t] = 0.0
    #update LMZ saturation. max(...,0) prevents negative saturation 
    hydro["s_ex"][t] = max(s_prev + hydro["ds"][t], 0.0)
    #limit s_lim to field capacity 
    #s_ex can exceed field capacity, but s_lim cannot 
    hydro["s_lim"][t] = min(hydro["s_ex"][t], cfg.sfc)

    #saturation excess water
    #if s_ex exceeds 1, the LMZ contains more water than its pore space can physically store 
    if hydro["s_ex"][t] > 1.0:
        #convert the excess saturation into a depth of water 
        #because 1 - s_ex < 0 and y_hm<0 their product gives a positive excess water depth 
        saturation_excess_step = (
            (1.0 - hydro["s_ex"][t])
            * hydro["y_hm"][t]
            * cfg.n
        )
        #soil saturation cannot physically exceed 1 
        hydro["s_ex"][t] = 1.0
    #LMZ saturation excess is routed directly to runoff, it is not currently added to h_wet. 
    #this was done to prevent ponded water from appearing while the modeled groundwater table was still well below the landsurface 
    h_runoff_step = saturation_excess_step

    # Above-ground wetland storage
    #the ponded water state is calculated after the original Laio subsurface update 
    # Water available at the surface includes:
    # 1. water already ponded from the previous timestep
    # 2. rainfall directly onto the ponded wetland
    # 3. water pushed above the soil surface

    wetland_available = max(
        0.0,
        surface_water_preoverflow
        + surface_excess_step
    )

    # Store water until the wetland reaches its maximum depth
    hydro["h_wet"][t] = min(
        wetland_available,
        cfg.WETLAND_STORAGE_MAX
    )

    # Anything above depression-storage capacity becomes runoff
    #+= is importrnat because saturation excess runoff may already have been calculated above 
    h_runoff_step += max(
        0.0,
        wetland_available
        - cfg.WETLAND_STORAGE_MAX
    )
#constrain the saturation value passed to the C/N model to the physical range 0-1 
    hydro["s"][t] = np.clip(hydro["s_ex"][t], 0.0, 1.0)

    # Store hydrologic fluxes/states
    hydro["P_flux"][t] = P_step #precipitation 
    hydro["PET_flux"][t] = PET_step #total PET demand 
    hydro["PET_lm_flux"][t] = PET_lm_step # PET allocated to the LMZ 
    hydro["ET_lm_flux"][t] = ET_lm_step #actual ET from the LMZ 
    hydro["ET_wt_flux"][t] = ET_wt_step #actual ET supplied by groundwater 
    hydro["R_flux"][t] = R_step #direct recharge to the saturated zone 
    hydro["R_lm_flux"][t] = R_lm_step #save rainfall/input entering the LMZ 
    hydro["L_lm_flux"][t] = L_lm_step #LMZ leakage into the saturated zone 
    hydro["ExfilSat_flux"][t] = ExfilSat_step #upward exfiltration from the saturated zone 
    hydro["runoff_flux"][t] = h_runoff_step #total runoff/overflow 

    hydro["ET_surface_flux"][t] = ET_surface_step #ET removed directly from surface ponded water 

    #total input to the surface water pool 
    #includes groundwater derived surface excess 
    #rainfall falling on already ponded water 
    hydro["wetland_input_flux"][t] = ( 
        surface_excess_step
        + surface_rain_step
    )
    #calculate soil water storage using a fixed root zone depth 
    # saturation x porosity x Zr
    hydro["storage"][t] = (
        hydro["s"][t]
        * cfg.n
        * cfg.Zr
    )
    #add ponded water depth to the simplified soil storage 
    hydro["total_storage"][t] = (
        hydro["storage"][t]
        + hydro["h_wet"][t]
    )
    #values passed to the C/N model 
    #estimate the volume/depth of water associated with the C/N soil domain 
    #the small lower bound prevents division by zero 
    water_vol = max(hydro["s"][t] * cfg.n * cfg.Zr, 1e-12)
    #return the hydrologic quantities required by biogc.py
    return { 
        #soil saturation controls several microbial and nitrogen processes 
        "s_for_biogeochem": hydro["s"][t],
        #soil water volume used to convert some hydrologic fluxes into effective rates 
        "water_vol": water_vol,
        #effective leakage rate used to calculate N leaching 
        #water depth fluxed divided by modeled water volume
        "Lrate_hydro": L_lm_step / water_vol,
        #effective plant water uptake/ET rate used in the passive N uptake equations 
        #currently this uses ET from the low moisture zone 
        "Trate_hydro": ET_lm_step / water_vol,
    }
