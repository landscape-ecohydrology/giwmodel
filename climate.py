#climate generation 
#this file creates the climate forcing used by the model: 
# stochastic rainfall 
# seasonal PET 
# the main model imports generate_climate() which returns: 
# rain, Ew_series, Tmax_series, PET_series 


import numpy as np


def generate_rain(N, lambda_inv, alpha, dt, seed=0):
    """
    Generate stochastic rainfall time series. 
    Paramters: 
    N: int 
        Total number of model timesteps 
        
    lambda_inv: float
        Parameter controlling the spacing between rainfall events 
        
    alpha : float 
        Mean rainfall event depth used by the exponential distribution [m]
        
    dt : float 
        Model timestep 
    
    seed : int 
        Random number seed used to make the rainfall sequence reproducible 
        
    Returns 
    rain : ndarray 
        Rainfall septh at each model timestep [m/step]
    """
    #set the random number seed, ensuring the stochastic rainfall sequence is identical every time the 
    # model is run with the same seed 
    np.random.seed(seed)
    #create an array containing one rainfall value for every timestep 
    rain = np.zeros(N)
    #generate the timing of the first rainfall event 
    #adding 1 ensures that the first event occurs at least one day after the start rather than 
    #potentially occurring at day 0 
    day = np.random.poisson(lambda_inv) + 1
    #continue generating rainfall events until the end of the simulation 
    # N * dt converts the number of timesteps into total simulation days 
    while day < N * dt:
        index = int(day / dt)
        #check to make sure the calculated array index is still inside the rainfall array
        if index < N:
            #generate rainfall depth for this storm 
            #the storm depth is drawn from an exponential distribution with mean/scale parameter alpha
            rain[index] += np.random.exponential(scale=alpha)
        #generate the waiting time until the next rainfall event
        #this is from Poisson distribution 
        day += np.random.poisson(lambda_inv)
    return rain


def build_seasonal_pet(N, dt, pet_min, pet_split_T, pet_phase, baseline_daily_pet):
    """Build seasonal PET and split it between potential evaporation and transpiration.
    Parameters
    N : int
        total number of model timesteps
    dt : float 
        model timestep [days]
    pet_min : float
        small minimum value added to the seasonal PET shape 
    pet_split_T : float 
        fraction of total PET assigned to potential transpiration 
    pet_phase : float 
        Phase shift of the seasonal sine curve 
    baseline_daily_pet : float 
        target average PET rate [m/day]
        the seasonal curve is scaled so that its annual integral equals baseline_daily_pet * 365
    
    returns: 
    E_pot : ndarray 
        potential evaporatioin rate 
    T_pot : ndarray 
        potential transpiration rate [m/day]
    PET : ndarray 
        total PET rate [m/day]
        """
    #calculate the number of model timesteps in one year 
    steps_per_year = int(round(365 / dt))
    #creating a day of year value for each timestep 
    doy = (np.arange(steps_per_year) * dt) % 365.0 + 1.0
    #create the seasonal shape of PET 
    # np.sin(...) produces a smooth seasonal cycle 
    #the sine function runs through approx. one positive half cycle over the year: 
    #low pet in winter, high pet in summer, low again in winter 
    #pet_min prevents the seasonal shape from being exactly zero 
    shape_year = np.maximum(
        0.0,
        pet_min + np.sin(np.pi * (doy / 365.0) + pet_phase),
    )
    #calculate the desired total PET for one year 
    #baseline_daily_pet has units m/day 
    #multiplying by 365 dats gives the target annual PET depth 
    baseline_total_year = baseline_daily_pet * 365.0
    #calculate a scaling factor so that the seasonal curve integrates to exactly the desired annual PET total 
    scale = baseline_total_year / (shape_year.sum() * dt)
    #repeate the one year seasonal PET pattern enough times to cover the full simulation
    #np.tile repeats the annual shape 
    shape_full = np.tile(shape_year, int(np.ceil(N / steps_per_year)))[:N]
    #convert seasonal shape into PET rate [m/day]
    PET = scale * shape_full
    #split total pet into potential transpiration
    #pet_split_T = .55 means 55% of pet is assigned to transpiration
    T_pot = PET * pet_split_T
    #the remaining fraction goes to evaporation
    E_pot = PET * (1.0 - pet_split_T)
    return E_pot, T_pot, PET

#combine rainfall and pet into one climate generator
def generate_climate(cfg):
    """
    generate all climate forcing required by the model. 
    the parameter values are pulled from parameters.py through cfg. 
    returns: 
    rain : ndarray 
        rainfall depth [m/step]
    Ew_series : ndarray
        potential evaporation rate [m/day]
    Tmax_series : ndarray 
        potential transpiration rate [m/day]
    PET_series : ndarray 
        total potential evapotranspiration rate [m/day]
    """

    rain = generate_rain(
        cfg.N, cfg.lambda_inv, cfg.alpha, cfg.dt, seed=cfg.RAIN_SEED
    )
    #generate PET 
    if cfg.USE_SEASONAL_PET:
        Ew_series, Tmax_series, PET_series = build_seasonal_pet(
            N=cfg.N,
            dt=cfg.dt,
            pet_min=cfg.PET_MIN,
            pet_split_T=cfg.PET_SPLIT_T,
            pet_phase=cfg.PET_PHASE,
            baseline_daily_pet=cfg.BASELINE_PET_DAILY,
        )
    #non seasonal PET option 
    else:
        Ew_series = np.full(cfg.N, cfg.Ew) #create a constant potential evaporation array
        Tmax_series = np.full(cfg.N, cfg.Tmax) # create a constant potential transpiration array
        PET_series = Ew_series + Tmax_series # total PET is the sum of potential evaporation and potential transpiration

    return rain, Ew_series, Tmax_series, PET_series
