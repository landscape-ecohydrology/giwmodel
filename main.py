'''
file: main.py
this file contains the main function for running the GIW model. 
it generates climate data, initializes hydrology and biogeocfhemistry states, and runs the model forward one timestep 
at a time. it couples the hydrology to the carbon/nitrogen model and stores the diagnostic fluxes for c/n fluxes.
it also runs mass balance checks, and creates plots and output tables. 
the detailed equations are not calculated here, but are instead in separate modules: parameters.py, climate.py, 
hydrology.py, biogc.py, and diagnostics.py.
'''



import numpy as np
from scipy.integrate import solve_ivp
#import model parameters as cfg (configuration)
import parameters as cfg
from climate import generate_climate
from hydrology import initialize_hydrology, hydrology_step
from biogc import coupled_rhs, fd, fn
from diagnostics import (
    plot_hydrology_zoom,
    print_water_balance,
    print_carbon_balance,
    print_nitrogen_balance,
    build_report_dataframe,
    plot_hydrology,
    plot_carbon,
    plot_nitrogen,
    plot_paper_figures,
)


def initialize_biogeochemistry(N, initial_state):
    """create arrays that will store carbon and nitrogen pools and fluxes for every timestep of the simulation.
    parameters: N: number of timesteps, initial_state: initial values for the 8 state variables (Cl, Ch, Cb, Nl, Nh, Nb, NH4, NO3)
    returns: bio: dictionary of arrays for each state variable and fluxes"""
    #create an empty dictionary to hold the arrays
    bio = {}
    #names of the eight C/N state variables
    #Cl = litter carbon, Ch = humus carbon, Cb = microbial biomass carbon
    #Nl = litter nitrogen, Nh = humus nitrogen, Nb = microbial biomass nitrogen
    #NH4 = ammonium nitrogen, NO3 = nitrate nitrogen
    names = ["Cl", "Ch", "Cb", "Nl", "Nh", "Nb", "NH4", "NO3"]
    #loop through each state variable
    #enumerate returns both the index (i) and the name of the state variable
    for i, name in enumerate(names):
        #create an array of zeros for each state variable, with length N
        bio[name] = np.zeros(N)
        #set the first timestep equal to the initial value for that state variable
        bio[name][0] = initial_state[i]
#create arrays for the biogeochemical fluxes
#these all begin at zero, and will be filled in during the model run
    for name in [
        "DEC_L_flux", "DEC_H_flux", "BD_flux", "CO2_flux", #carbon fluxes, litter, humus, microbial biomass, and CO2
        "N2_flux", "N2O_flux", "LE_NH4_flux", "LE_NO3_flux", #nitrogen fluxes, N2, N2O, produced by denitrification. ammonium and nitrate leaching
        "UP_NH4_flux", "UP_NO3_flux", "NIT_flux", "MIN_flux", #nitrogen fluxes, uptake, mineralization
        "IMM_NH4_flux", "IMM_NO3_flux", "DENIT_flux", #nitrogen fluxes, immobilization and denitrification
    ]:
        bio[name] = np.zeros(N) #allocate one zero filled array for each flux
        #return the dictionary of arrays for the biogeochemical state variables and fluxes
    return bio

#running the model 
def run_model():
    """ 
    Run the complete coupled hydrology-carbon-nitrogen simulation.
    Returns: 
    results: dict
        Contains the hydrology and biogeochemistry results
    rain: array of rainfall values for each timestep
    """

    #generate climate data
    #generate_climate returns: 
    #rain = precipitation depth for each timestep [m/step]
    #Ew_series = potential evaporation for each timestep [m/step]
    #Tmax_series = potential transpiration for each timestep [m/step]
    #PET_series = potential evapotranspiration for each timestep [m/step]
    # The climate calcuations are located in climate.py, and are based on the parameters defined in parameters.py
    rain, Ew_series, Tmax_series, PET_series = generate_climate(cfg)

#temp climate / timestep check
    print("\n===== CLIMATE / TIMESTEP CHECK =====")
    #print the timestep in days, number of steps, and steps per day
    print("dt:", cfg.dt, "days")
    print("Number of steps:", cfg.N)
    print("Steps per day:", 1.0 / cfg.dt)

    print("\nRain:")
    print("  Total:", rain.sum(), "m")
    print("  Annual:", rain.sum() / cfg.years, "m/yr")

    print("\nPET:")
    print("  PET integrated total:", PET_series.sum() * cfg.dt, "m")
    print(
        "  PET annual:",
        PET_series.sum() * cfg.dt / cfg.years,
        "m/yr"
    )
    print("  Mean PET step:", PET_series.mean(), "m/step")
    print("  Max PET step:", PET_series.max(), "m/step")

    #initialize hydrologic state variables 
    #create all hydrologic state and flux arrays
    #soil saturation, water table elevation, capillary fringe elevation, infiltration/recharge, ET, leakage, runoff, above ground wetland storage

    hydro = initialize_hydrology(cfg.N, cfg)

    #initalize carbon and nitrogen state variables
    #create all biogeochemical state and flux arrays
    bio = initialize_biogeochemistry(cfg.N, cfg.INITIAL_CN_STATE)

    #"state" is the current set of eight carbon and nitrogen state variables (Cl, Ch, Cb, Nl, Nh, Nb, NH4, NO3)
    #copy() is used to avoid modifying the original initial state array
    state = cfg.INITIAL_CN_STATE.copy()

#main time loop
#move foward through the simulation one timestep at a time 
#start at t=1 because the initial state is already set at t=0
    for t in range(1, cfg.N):
        #climate input for this timestep
        #rainfall is already stored as a depth per timestep
        P_step = rain[t]

        #PET_series is stored as a rate [m/day]
        #multiply by dt to convert it to the water depth available for this timestep [m/step]
        PET_step = PET_series[t] * cfg.dt

        #hydrology 
        #advance the hydrology model by one timestep.

        #hydrology_step updates 
        # soil saturation, water table elevation, capillary fringe elevation, infiltration/recharge, ET, leakage, runoff, above ground wetland storage
        #it also returns hydrologic values needed by the C/N model
        hydro_rates = hydrology_step(t, P_step, PET_step, hydro, cfg)
        #soil saturation used by the C/N model 
        s_cur = hydro_rates["s_for_biogeochem"]
        #current modeled water volume in the soil (m^3)
        #this is used to convert hydrologic fluxes to effective rates for nitrogen leaching and plant uptake
        water_vol = hydro_rates["water_vol"]
        #effective leakage rate passed from hydrology to the nitrogen model 
        Lrate = hydro_rates["Lrate_hydro"]
        #effective ET/transpiration related rate passed from hydrology to plant nitrogen uptake
        Trate = hydro_rates["Trate_hydro"]

        #carbon nitrogen ode solver
        #solve_ivp integrates the eight coupled c/n differential equations across the current timestep
        sol = solve_ivp(

            #lambda provides the arguments required by coupled_rhs()
            #_t = internal time variable used by solve_ivp
            #_y = current c/n state used by solve_ivp
            lambda _t, _y: coupled_rhs(
                _t, _y,
                #current soil moisture state from hydrology 
                s_cur, 
                #carbon/nitrogen parameters dictionary
                cfg.params, 
                #potential transpiration for this timestep
                Tmax_series[t],
                #full parameter configuration object
                cfg,
                #replace the default leakage and transpiration rates with the effective rates calculated by hydrology_step()
                Lrate_override=Lrate,
                Trate_override=Trate,
            ),
            #integrate from the beginning to the end of one timestep
            [0, cfg.dt],
            #starting c/n state for this timestep
            state,
            #explicit adaptive Runge-Kutta numerical solver
            method="RK45",
        )
        #update c/n state 
        #sol.y contains the solution throughout the integration
        #[:, -1] selects the final value of every state variable at the end of the timestep
        #which becomes the initial condition for the next timestep 
        state = sol.y[:, -1]
        #prevent NH4 and NO3 concentrations from becoming negative due to numerical errors
        state[6:8] = np.maximum(state[6:8], 0.0)
        #prenvent Cl, Ch, Cb, Nl, Nh, Nb from becoming negative due to numerical errors
        state[:6] = np.maximum(state[:6], 0.0)
        #save the current state values into their full simulation time-series arrays 
        for i, name in enumerate(["Cl", "Ch", "Cb", "Nl", "Nh", "Nb", "NH4", "NO3"]):
            bio[name][t] = state[i]

        # Recompute and store the same diagnostic C/N fluxes
        #coupled_rhs() calculates these processes internally while solving the ODEs
        #here they are recalculated once at the final state of the timestep so that we can save them for plotting, mass balances, and analysis
        #soil moisture response functions
        #moisture limitation factor used for decomposition
        fd_s = fd(s_cur, cfg)
        #moisture limitation factor used for nitrification
        fn_s = fn(s_cur, cfg)
        #give names to each value in the state array so the equations are easier to read
        Cl, Ch, Cb, Nl, Nh, Nb, NH4, NO3 = state
        #calculate the current litter c/n ratio
        #max(..., 1e-12) is used to prevent division by zero
        CN_lit = Cl / max(Nl, 1e-12)
        #determine how much decomposed litter can be transferred into the humus pool 
        #rh_eff cannot exceed rh_max, and is also limited by the current litter c/n ratio and the humus c/n ratio
        rh_eff = min(cfg.params["rh_max"], cfg.params["CNh"] / max(CN_lit, 1e-12))

        #phi_den describes the balance between nitrogen released from decomposition and nitrogen required for microbial growth
        phi_den = cfg.params["kh"] * Ch * (1.0/cfg.params["CNh"] - (1.0-cfg.params["rr"])/cfg.params["CNb"]) \
            + cfg.params["kl"] * Cl * (1.0/CN_lit - rh_eff/cfg.params["CNh"] - (1.0-rh_eff-cfg.params["rr"])/cfg.params["CNb"])
        #amount of mineral nitrogen potentially available for immobilization
        phi_num = -(cfg.params["ki_plus"] * NH4 + cfg.params["ki_minus"] * NO3)
        #if decomposition releases enough nitrogen, decomposition does not need to be N limited
        if phi_den >= 0.0:
            phi_small = 1.0
        else: #otherwise determine how strongly mineral nitrogen limits decomposition
            phi_small = phi_num / max(phi_den, -1e-12)
            #force the limiter to remain between 0 and 1 
            phi_small = max(0.0, min(1.0, phi_small))
        #CARBON DECOMPOSITION
        #litter decomposition rate
        DECl = phi_small * fd_s * cfg.params["kl"] * Cb * Cl
        #humus decomposition rate
        DECh = cfg.params["fclay"] * phi_small * fd_s * cfg.params["kh"] * Cb * Ch
        #microbial biomass mortality rate
        BD = cfg.params["kd"] * Cb
        #SAVE CARBON FLUXES
        #multiply rates by dt to convert amount/day to amount/timestep
        bio["DEC_L_flux"][t] = DECl * cfg.dt
        bio["DEC_H_flux"][t] = DECh * cfg.dt
        bio["BD_flux"][t] = BD * cfg.dt
        #fraction rr of decomposed carbon is respired as CO2 
        bio["CO2_flux"][t] = cfg.params["rr"] * (DECl + DECh) * cfg.dt

        #MINERALIZATION AND IMMOBILIZATION

        #net mineralization/immobilization rate 
        #positive phi = net mineralization, negative phi = net immobilization
        PHI = phi_small * fd_s * Cb * phi_den

        #maximum amount of mineral N that microbes could immobilize 
        IMM_max = (cfg.params["ki_plus"]*NH4 + cfg.params["ki_minus"]*NO3) * fd_s * Cb
        #if phi is positive, nitrogen is mineralized and there is no immobilization
        if PHI > 0.0:
            IMM = 0.0
        else: #if phi is negative, microbes immobilize mineral nitrogen, but not more than is available in the soil
            IMM = min(-PHI, IMM_max)
        #denominator used to divide immobilization between ammonium and nitrate pools
        den_imm = max(cfg.params["ki_plus"]*NH4 + cfg.params["ki_minus"]*NO3, 1e-12)
        #portion of immobilization taken from NH4
        IMM_NH4 = IMM * (cfg.params["ki_plus"] * NH4) / den_imm
        #portion of immobilization taken from NO3
        IMM_NO3 = IMM * (cfg.params["ki_minus"] * NO3) / den_imm
        #NITRIFICATION
        #convert NH4 to NO3
        #nitrification depends on nitrification coefficient kn, soil moisture response fn_s, microbial biomass Cb, and NH4 concentration
        NIT = cfg.params["kn"] * fn_s * Cb * NH4
        #store internal N fluxes
        #convert rates to timestep totals using dt 
        bio["NIT_flux"][t] = NIT * cfg.dt
        #only positive PHI represents mineralization 
        bio["MIN_flux"][t] = max(0.0, PHI) * cfg.dt
        bio["IMM_NH4_flux"][t] = IMM_NH4 * cfg.dt
        bio["IMM_NO3_flux"][t] = IMM_NO3 * cfg.dt

        #N LEACHING 
        #hydrologically driven NH4 loss and NO3 loss from the soil
        bio["LE_NH4_flux"][t] = cfg.params["a_plus"] * Lrate * NH4 * cfg.dt
        bio["LE_NO3_flux"][t] = cfg.params["a_minus"] * Lrate * NO3 * cfg.dt
        #passive NH4 uptake associated with water uptake
        UP_p_NH4 = cfg.params["a_plus"] * Trate * NH4
        #passive NO3 uptake associated with water uptake
        UP_p_NO3 = cfg.params["a_minus"] * Trate * NO3
        #ACTIVE PLANT UPTAKE
        #potential active NH4 uptake coefficient
        #uptake increases with soil moisture according to s^dd 
        ku_plus = cfg.params["a_plus"] * cfg.params["F"] * (s_cur ** cfg.params["dd"]) / water_vol
        #potential active NO3 uptake coefficient
        ku_minus = cfg.params["a_minus"] * cfg.params["F"] * (s_cur ** cfg.params["dd"]) / water_vol
        #remaining plant NH4 demand after passive uptake
        dem_p = max(cfg.params["DEM_plus"] - UP_p_NH4, 0.0)
        #remaining plant NO3 demand after passive uptake
        dem_m = max(cfg.params["DEM_minus"] - UP_p_NO3, 0.0)
        #active nh4 uptake cannot exceed remaining plant demand or the potential uptake rate
        UP_a_NH4 = min(ku_plus * NH4, dem_p)
        #active no3 uptake cannot exceed remaining plant demand or the potential uptake rate
        UP_a_NO3 = min(ku_minus * NO3, dem_m)
        #total NH4 uptake = passive + active uptake
        bio["UP_NH4_flux"][t] = (UP_p_NH4 + UP_a_NH4) * cfg.dt
        #total NO3 uptake = passive + active uptake
        bio["UP_NO3_flux"][t] = (UP_p_NO3 + UP_a_NO3) * cfg.dt

        #DENITRIFICATION    
        #no denitrification occurs below field capacity
        if s_cur <= cfg.sfc:
            fs_den = 0.0
        else: #above field capacity, denitrification increases as the soil approaches full saturation 
            fs_den = ((s_cur - cfg.sfc) / max(1.0 - cfg.sfc, 1e-12)) ** cfg.params["w"]
        #nitrate limitation factor for denitrification
        #at low nitrate concentrations, denitrification is limited by the availability of nitrate
        #at high nitrate concentration, this factor approaches 1 and denitrification is limited by soil moisture
        fN_den = NO3 / (cfg.params["Kmm"] + NO3)
        #calculate total N lost by denitrification during this timestep
        DENIT_step = cfg.params["k_den"] * fN_den * fs_den * NO3 * cfg.dt
        #store total denitrification fluxes
        bio["DENIT_flux"][t] = DENIT_step
        #assume that half of the denitrified nitrogen is lost as N2 and half as N2O
        #this is a fixed partition rather than a dynamically modeled ratio, could modify later 
        bio["N2O_flux"][t] = 0.5 * DENIT_step
        bio["N2_flux"][t] = 0.5 * DENIT_step
    #end of time loop 
    #packaging the hydrology and biogeochemistry together into one results object
    #making it easy to pass all model output to the diagnostics and plotting functions
    return {"hydro": hydro, "bio": bio}, rain

    #main output/diagnostics function 
def main():
    """
    run the model, calculate diagnostics, create output tables, and generate plots. 
    """
    #run the complete simulation
    #results contains the hydrology and biogeochemistry results, rain contains the rainfall time series
    results, rain = run_model()
    #check conservation of mass for water, carbon, and nitrogen
    print_water_balance(results, rain, cfg)
    print_carbon_balance(results, cfg)
    print_nitrogen_balance(results, cfg)

    df_report = build_report_dataframe(results, rain, cfg)
    # df_report.to_csv(f"soil_water_carbon_output_{cfg.selected_soil}.csv", index=False)

    #plot 14 day period starting on day 100
    plot_hydrology_zoom(
        results,
        rain,
        cfg,
        start_day=100,
        days=14
    )
    #plot soil moisture, rainfall, et, infiltration, leakage, runoff, and water table behavior 
    plot_hydrology(results, rain, cfg)
    #plot carbon and nitrogen pools and fluxes
    plot_carbon(results, cfg)
    plot_nitrogen(results, cfg)
    if cfg.PLOT_PAPER_FIGS:
        plot_paper_figures(results, cfg)

    return results, rain, df_report


if __name__ == "__main__":
    main()
