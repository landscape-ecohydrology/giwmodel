import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def vol_to_areal(x, Zr):
    return x * Zr / 1000.0


def print_water_balance(results, rain, cfg):
    h = results["hydro"]

    
    # External fluxes
    
    ET_lm = h["ET_lm_flux"]
    ET_wt = h["ET_wt_flux"]
    runoff = h["runoff_flux"]

    # Surface-water ET only exists after adding wetland storage.
    if "ET_surface_flux" in h:
        ET_surface = h["ET_surface_flux"]
    else:
        ET_surface = np.zeros_like(rain)

    total_ET = ET_lm + ET_wt + ET_surface

    
    # Storage components
    

    # 1. Vadose / low-moisture-zone water storage
    #
    # s_ex is volumetric saturation [-]
    # n is porosity [-]
    # |y_hm| is depth of the low-moisture zone [m]
    #
    # Water depth stored in LMZ:
    S_lmz = (
        h["s_ex"]
        * cfg.n
        * np.abs(h["y_hm"])
    )

    # 2. Groundwater storage
    #
    # y_cap is the groundwater state actually updated by the
    # Laio water-table balance.
    #
    # Use storage relative to the confining layer.
    S_gw = (
        cfg.Sy_soil
        * (h["y_cap"] - cfg.z_cl)
    )

    # 3. Above-ground ponded wetland storage
    if "h_wet" in h:
        S_wet = h["h_wet"]
    else:
        S_wet = np.zeros_like(rain)

    # Total modeled water storage
    S_total = S_lmz + S_gw + S_wet

    
    # Whole-simulation water balance
    

    total_P = rain.sum()
    total_ET_all = total_ET.sum()
    total_runoff = runoff.sum()

    delta_storage = S_total[-1] - S_total[0]

    residual = (
        total_P
        - total_ET_all
        - total_runoff
        - delta_storage
    )

    print("======= WHOLE-SYSTEM WATER BALANCE =======")
    print(f"Total rainfall:             {total_P:.6f} m")
    print(f"Total ET from LMZ:          {ET_lm.sum():.6f} m")
    print(f"Total ET from water table:  {ET_wt.sum():.6f} m")
    print(f"Total surface-water ET:     {ET_surface.sum():.6f} m")
    print(f"Total ET:                   {total_ET_all:.6f} m")
    print(f"Total runoff/overflow:      {total_runoff:.6f} m")
    print(f"Δ LMZ storage:              {S_lmz[-1] - S_lmz[0]:.6f} m")
    print(f"Δ groundwater storage:      {S_gw[-1] - S_gw[0]:.6f} m")
    print(f"Δ wetland storage:          {S_wet[-1] - S_wet[0]:.6f} m")
    print(f"Δ total storage:            {delta_storage:.6f} m")
    print(f"Water balance residual:     {residual:.6f} m")
    print(
        f"Percent error:              "
        f"{100 * residual / max(total_P, 1e-12):.4f}%"
    )
    print("==========================================")

    
    # Yearly water balance
    

    print("\n======= YEARLY WATER BALANCE =======")

    steps_per_year = int(365 / cfg.dt)

    annual_results = []

    for y in range(cfg.years):

        start = y * steps_per_year
        end = min((y + 1) * steps_per_year, len(rain))

        P_y = rain[start:end].sum()

        ET_lm_y = ET_lm[start:end].sum()
        ET_wt_y = ET_wt[start:end].sum()
        ET_surface_y = ET_surface[start:end].sum()

        ET_y = (
            ET_lm_y
            + ET_wt_y
            + ET_surface_y
        )

        runoff_y = runoff[start:end].sum()

        # Storage change from start to end of the year
        if end < len(S_total):
            dS_y = S_total[end] - S_total[start]
        else:
            dS_y = S_total[end - 1] - S_total[start]

        residual_y = (
            P_y
            - ET_y
            - runoff_y
            - dS_y
        )

        percent_error_y = (
            100.0
            * residual_y
            / max(P_y, 1e-12)
        )

        annual_results.append(
            (
                P_y,
                ET_lm_y,
                ET_wt_y,
                ET_surface_y,
                ET_y,
                runoff_y,
                dS_y,
                residual_y,
                percent_error_y,
            )
        )

        print(f"Year {y+1}")
        print(f"  Rainfall:                {P_y:.4f} m")
        print(f"  ET from LMZ:             {ET_lm_y:.4f} m")
        print(f"  ET from water table:     {ET_wt_y:.4f} m")
        print(f"  Surface-water ET:        {ET_surface_y:.4f} m")
        print(f"  Total ET:                {ET_y:.4f} m")
        print(f"  Runoff/overflow:         {runoff_y:.4f} m")
        print(f"  Δ Total Storage:         {dS_y:.4f} m")
        print(f"  Balance Residual:        {residual_y:.6f} m")
        print(f"  Percent Error:           {percent_error_y:.3f}%")
        print()

    return annual_results


def print_carbon_balance(results, cfg):
    b = results["bio"]
    steps_per_year = int(365 / cfg.dt)
    ADD_areal_per_day = vol_to_areal(cfg.params["ADD"], cfg.Zr)
    annual = []

    print("======= YEARLY CARBON BALANCE =======")
    for y in range(cfg.years):
        start, end = y * steps_per_year, (y + 1) * steps_per_year
        days = (end - start) * cfg.dt
        dec_l_y = vol_to_areal(b["DEC_L_flux"][start:end].sum(), cfg.Zr)
        dec_h_y = vol_to_areal(b["DEC_H_flux"][start:end].sum(), cfg.Zr)
        bd_y = vol_to_areal(b["BD_flux"][start:end].sum(), cfg.Zr)
        co2_y = vol_to_areal(b["CO2_flux"][start:end].sum(), cfg.Zr)
        dCl = vol_to_areal(b["Cl"][end - 1] - b["Cl"][start], cfg.Zr)
        dCh = vol_to_areal(b["Ch"][end - 1] - b["Ch"][start], cfg.Zr)
        dCb = vol_to_areal(b["Cb"][end - 1] - b["Cb"][start], cfg.Zr)
        dC_total = dCl + dCh + dCb
        ADD_y = ADD_areal_per_day * days
        error = ADD_y - co2_y - dC_total
        annual.append((ADD_y, dec_l_y, dec_h_y, bd_y, co2_y, dCl, dCh, dCb, dC_total, error))
        print(f"Year {y+1}: ADD={ADD_y:.4f}, CO2={co2_y:.4f}, ΔC={dC_total:.4f}, error={error:.6f} kg/m²")

    totals = np.sum(np.asarray(annual), axis=0)
    print("======= TOTAL CARBON BALANCE =======")
    print(f"Total Litter Added:        {totals[0]:.4f} kg/m²")
    print(f"Total CO2 Emission:        {totals[4]:.4f} kg/m²")
    print(f"Δ Total Soil Carbon:       {totals[8]:.4f} kg/m²")
    print(f"Total Carbon Balance Error:{totals[9]:.6f} kg/m²")
    print("====================================")
    return annual


def print_nitrogen_balance(results, cfg):
    b = results["bio"]
    steps_per_year = int(365 / cfg.dt)
    ADD_N_areal_per_day = vol_to_areal(cfg.params["ADD"] / cfg.params["CNadd"], cfg.Zr)
    annual = []

    print("======= YEARLY NITROGEN BALANCE =======")
    for y in range(cfg.years):
        start, end = y * steps_per_year, (y + 1) * steps_per_year
        days = (end - start) * cfg.dt
        ADDN_y = ADD_N_areal_per_day * days
        LE_y = vol_to_areal(b["LE_NH4_flux"][start:end].sum() + b["LE_NO3_flux"][start:end].sum(), cfg.Zr)
        UPT_y = vol_to_areal(b["UP_NH4_flux"][start:end].sum() + b["UP_NO3_flux"][start:end].sum(), cfg.Zr)
        DENIT_y = vol_to_areal(b["DENIT_flux"][start:end].sum(), cfg.Zr)
        dN = (b["Nl"][end-1]-b["Nl"][start] + b["Nh"][end-1]-b["Nh"][start] + b["Nb"][end-1]-b["Nb"][start] + b["NH4"][end-1]-b["NH4"][start] + b["NO3"][end-1]-b["NO3"][start])
        dN_total = vol_to_areal(dN, cfg.Zr)
        error = ADDN_y - (LE_y + UPT_y + DENIT_y) - dN_total
        annual.append((ADDN_y, LE_y, UPT_y, DENIT_y, dN_total, error))
        print(f"Year {y+1}: input={ADDN_y:.6f}, leach={LE_y:.6f}, uptake={UPT_y:.6f}, denit={DENIT_y:.6f}, ΔN={dN_total:.6f}, error={error:.6f} kgN/m²")

    totals = np.sum(np.asarray(annual), axis=0)
    print("======= TOTAL NITROGEN BALANCE =======")
    print(f"Total N Input:             {totals[0]:.6f} kgN/m²")
    print(f"Total N Leaching:          {totals[1]:.6f} kgN/m²")
    print(f"Total N Uptake:            {totals[2]:.6f} kgN/m²")
    print(f"Total Denitrification:     {totals[3]:.6f} kgN/m²")
    print(f"Total Nitrogen MB Error:   {totals[5]:.6f} kgN/m²")
    print("======================================")
    return annual


def build_report_dataframe(results, rain, cfg):
    h, b = results["hydro"], results["bio"]
    return pd.DataFrame({
        "Time (days)": cfg.time,
        "Rain (m)": rain,
        "Soil Saturation": h["s"],
        "Water Table Elevation (m)": h["y_wt"],
        "Capillary Fringe Top (m)": h["y_cap"],
        "High/Low Moisture Boundary (m)": h["y_hm"],
        "Infiltration (m/step)": h["R_lm_flux"],
        "Runoff (m/step)": h["runoff_flux"],
        "Evaporation (m/step)": h["ET_lm_flux"],
        "Transpiration (m/step)": h["ET_wt_flux"],
        "Leakage (m/step)": h["L_lm_flux"],
        "Litter (kg/m^2)": vol_to_areal(b["Cl"], cfg.Zr),
        "Humus (kg/m^2)": vol_to_areal(b["Ch"], cfg.Zr),
        "Microbial Biomass (kg/m^2)": vol_to_areal(b["Cb"], cfg.Zr),
        "Litter Decomp (kg/m^2/step)": vol_to_areal(b["DEC_L_flux"], cfg.Zr),
        "Humus Decomp (kg/m^2/step)": vol_to_areal(b["DEC_H_flux"], cfg.Zr),
        "Biomass Death (kg/m^2/step)": vol_to_areal(b["BD_flux"], cfg.Zr),
        "CO2 Emission (kg/m^2/step)": vol_to_areal(b["CO2_flux"], cfg.Zr),
    })


def plot_hydrology(results, rain, cfg):
    h = results["hydro"]
    fig, axs = plt.subplots(4, 2, figsize=(16, 12))
    axs = axs.flatten()
    axs[0].plot(cfg.time, h["s"])
    axs[0].axhline(cfg.sfc, linestyle="--", label="Field Capacity")
    axs[0].axhline(1.0, linestyle="--", label="Saturation Max")
    axs[0].set_title("Soil Saturation"); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(cfg.time, rain / cfg.dt); axs[1].set_title("Rainfall (m/day)"); axs[1].grid(True)
    axs[2].plot(
        cfg.time,
        h["ET_lm_flux"] / cfg.dt,
        label="ET from low-moisture zone"
    )

    axs[2].plot(
        cfg.time,
        h["ET_wt_flux"] / cfg.dt,
        label="ET from water table"
    )

    if "ET_surface_flux" in h:
        axs[2].plot(
            cfg.time,
            h["ET_surface_flux"] / cfg.dt,
            label="ET from ponded water"
        )

    axs[2].set_title(
        "ET Components (m/day)"
    )

    axs[2].legend()
    axs[2].grid(True)
    axs[3].plot(cfg.time, h["R_lm_flux"] / cfg.dt); axs[3].set_title("Infiltration (m/day)"); axs[3].grid(True)
    axs[4].plot(cfg.time, h["L_lm_flux"] / cfg.dt); axs[4].set_title("Leakage (m/day)"); axs[4].grid(True)
    axs[5].plot(cfg.time, h["runoff_flux"] / cfg.dt); axs[5].set_title("Runoff (m/day)"); axs[5].grid(True)
    axs[6].plot(cfg.time, h["y_wt"], label="Water table")
    axs[6].plot(cfg.time, h["y_cap"], label="Capillary fringe top")
    axs[6].axhline(cfg.z_datum, linestyle="--", label="Surface")
    axs[6].axhline(cfg.z_cl, linestyle=":", label="Confining layer")
    axs[6].set_title("Fluctuating Water Table"); axs[6].legend(); axs[6].grid(True)
    axs[7].plot(cfg.time, h["ET_wt_flux"] / cfg.dt, label="ET from water table")
    axs[7].plot(cfg.time, h["ET_lm_flux"] / cfg.dt, label="ET from low-moisture zone")
    axs[7].set_title("Hydrology ET Components"); axs[7].legend(); axs[7].grid(True)
    plt.tight_layout(); plt.show()


def plot_carbon(results, cfg):
    b = results["bio"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    axs[0].plot(cfg.time, vol_to_areal(b["DEC_L_flux"] / cfg.dt, cfg.Zr)); axs[0].set_title("Litter Decomposition (kg/m²/day)"); axs[0].grid(True)
    axs[1].plot(cfg.time, vol_to_areal(b["DEC_H_flux"] / cfg.dt, cfg.Zr)); axs[1].set_title("Humus Decomposition (kg/m²/day)"); axs[1].grid(True)
    axs[2].plot(cfg.time, vol_to_areal(b["BD_flux"] / cfg.dt, cfg.Zr)); axs[2].set_title("Microbial Death (kg/m²/day)"); axs[2].grid(True)
    axs[3].plot(cfg.time, vol_to_areal(b["Cl"], cfg.Zr), label="Litter")
    axs[3].plot(cfg.time, vol_to_areal(b["Ch"], cfg.Zr), label="Humus")
    axs[3].plot(cfg.time, vol_to_areal(b["Cb"], cfg.Zr), label="Microbial Biomass")
    axs[3].set_title("Carbon Pools (kg/m²)"); axs[3].legend(); axs[3].grid(True)
    plt.tight_layout(); plt.show()

def plot_hydrology_zoom(results, rain, cfg, start_day=100, days=14):

    h = results["hydro"]

    end_day = start_day + days

    mask = (
        (cfg.time >= start_day)
        & (cfg.time <= end_day)
    )

    time = cfg.time[mask]

    fig, axs = plt.subplots(
        4,
        1,
        figsize=(12, 12),
        sharex=True
    )

    
    # Rainfall
    

    axs[0].plot(
        time,
        rain[mask] / cfg.dt
    )

    axs[0].set_ylabel("Rainfall\n(m/day)")
    axs[0].set_title("Rainfall")
    axs[0].grid(True)

    
    # Water table
    

    axs[1].plot(
        time,
        h["y_wt"][mask],
        label="Water table"
    )

    axs[1].plot(
        time,
        h["y_cap"][mask],
        label="Capillary fringe"
    )

    axs[1].axhline(
        cfg.z_datum,
        linestyle="--",
        label="Land surface"
    )

    axs[1].set_ylabel("Elevation (m)")
    axs[1].set_title("Water Table")
    axs[1].legend()
    axs[1].grid(True)

    
    # Above-ground wetland storage
    

    axs[2].plot(
        time,
        h["h_wet"][mask]
    )

    axs[2].axhline(
        cfg.WETLAND_STORAGE_MAX,
        linestyle="--",
        label="Storage capacity"
    )

    axs[2].set_ylabel("Ponded depth (m)")
    axs[2].set_title("Above-Ground Wetland Storage")
    axs[2].legend()
    axs[2].grid(True)

    
    # ET
    

    axs[3].plot(
        time,
        h["ET_lm_flux"][mask] / cfg.dt,
        label="ET from LMZ"
    )

    axs[3].plot(
        time,
        h["ET_wt_flux"][mask] / cfg.dt,
        label="ET from water table"
    )

    axs[3].plot(
        time,
        h["ET_surface_flux"][mask] / cfg.dt,
        label="ET from ponded water"
    )

    axs[3].set_ylabel("ET (m/day)")
    axs[3].set_xlabel("Time (days)")
    axs[3].set_title("ET Components")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_nitrogen(results, cfg):
    b = results["bio"]
    rate = lambda x: vol_to_areal(x / cfg.dt, cfg.Zr)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()
    axs[0].plot(cfg.time, vol_to_areal(b["NH4"], cfg.Zr), label="NH4+")
    axs[0].plot(cfg.time, vol_to_areal(b["NO3"], cfg.Zr), label="NO3-")
    axs[0].set_title("Mineral N Pools (kg/m²)"); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(cfg.time, vol_to_areal(b["Nl"], cfg.Zr), label="Nl")
    axs[1].plot(cfg.time, vol_to_areal(b["Nh"], cfg.Zr), label="Nh")
    axs[1].plot(cfg.time, vol_to_areal(b["Nb"], cfg.Zr), label="Nb")
    axs[1].set_title("Organic N Pools (kg/m²)"); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(cfg.time, rate(b["NIT_flux"]), label="Nitrification")
    axs[2].plot(cfg.time, rate(b["MIN_flux"]), label="Mineralization")
    axs[2].plot(cfg.time, rate(b["IMM_NH4_flux"]), label="Immobilization NH4")
    axs[2].plot(cfg.time, rate(b["IMM_NO3_flux"]), label="Immobilization NO3")
    axs[2].set_title("Internal N Transformations"); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(cfg.time, rate(b["LE_NH4_flux"]), label="Leach NH4")
    axs[3].plot(cfg.time, rate(b["LE_NO3_flux"]), label="Leach NO3")
    axs[3].plot(cfg.time, rate(b["UP_NH4_flux"]), label="Uptake NH4")
    axs[3].plot(cfg.time, rate(b["UP_NO3_flux"]), label="Uptake NO3")
    axs[3].plot(cfg.time, rate(b["DENIT_flux"]), label="Denitrification")
    axs[3].set_title("External N Losses"); axs[3].legend(); axs[3].grid(True)
    plt.tight_layout(); plt.show()


def plot_paper_figures(results, cfg):
    b, h = results["bio"], results["hydro"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10)); axs = axs.flatten()
    for ax, data, title in zip(axs, [h["s"], b["Cl"], b["Ch"], b["Cb"], b["NH4"], b["NO3"]], ["s", "C_l (gC m$^{-3}$)", "C_h (gC m$^{-3}$)", "C_b (gC m$^{-3}$)", "NH$_4^+$ (gN m$^{-3}$)", "NO$_3^-$ (gN m$^{-3}$)"]):
        ax.plot(cfg.time, data); ax.set_title(title); ax.grid(True)
    plt.tight_layout(); plt.show()

    gC_rate = b["DEC_L_flux"] / cfg.dt
    netMIN = (b["MIN_flux"] - (b["IMM_NH4_flux"] + b["IMM_NO3_flux"])) / cfg.dt
    no3_up = b["UP_NO3_flux"] / cfg.dt
    no3_le = b["LE_NO3_flux"] / cfg.dt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8)); axs = axs.flatten()
    for ax, data, title in zip(axs, [gC_rate, netMIN, no3_up, no3_le], ["Litter decomposition (gC m$^{-3}$ d$^{-1}$)", "Net mineralization (gN m$^{-3}$ d$^{-1}$)", "NO$_3^-$ uptake (gN m$^{-3}$ d$^{-1}$)", "NO$_3^-$ leaching (gN m$^{-3}$ d$^{-1}$)"]):
        ax.plot(cfg.time, data); ax.set_title(title); ax.grid(True)
    plt.tight_layout(); plt.show()
