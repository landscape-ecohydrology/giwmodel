import numpy as np


def evaporation(sval, Ew_t, cfg):
    if sval <= cfg.sh:
        return 0.0
    if sval <= cfg.sw:
        return Ew_t * (sval - cfg.sh) / max(cfg.sw - cfg.sh, 1e-12)
    return Ew_t


def transpiration(sval, Tmax_t, cfg):
    if sval <= cfg.sw:
        return 0.0
    if sval <= cfg.s_star:
        return Tmax_t * (sval - cfg.sw) / max(cfg.s_star - cfg.sw, 1e-12)
    return Tmax_t


def leakage(sval, cfg):
    beta = 2 * cfg.b + 4
    num = np.exp(beta * (sval - cfg.sfc)) - 1.0
    den = np.exp(beta * (1.0 - cfg.sfc)) - 1.0
    val = cfg.Ks * (num / max(den, 1e-12))
    return max(0.0, val)


def initialize_hydrology(N, cfg):
    """Allocate and initialize all hydrologic state and flux arrays."""
    hydro = {
        "s": np.zeros(N),
        "s_ex": np.zeros(N),
        "s_lim": np.zeros(N),
        "ds": np.zeros(N),
        "lmz_stor": np.zeros(N),
        "y_hm": np.full(N, cfg.INITIAL_Y_HM),
        "y_cap": np.full(N, cfg.INITIAL_Y_CAP),
        "y_wt": np.full(N, cfg.INITIAL_Y_WT),
        "P_flux": np.zeros(N),
        "PET_flux": np.zeros(N),
        "PET_lm_flux": np.zeros(N),
        "ET_lm_flux": np.zeros(N),
        "ET_wt_flux": np.zeros(N),
        "R_flux": np.zeros(N),
        "R_lm_flux": np.zeros(N),
        "L_lm_flux": np.zeros(N),
        "ExfilSat_flux": np.zeros(N),
        "runoff_flux": np.zeros(N),
        # Above-ground wetland storage
        "h_wet": np.zeros(N),
        "ET_surface_flux": np.zeros(N),
        "wetland_input_flux": np.zeros(N),

        "storage": np.zeros(N),
        "total_storage": np.zeros(N),
    }

    hydro["h_wet"][0] = cfg.INITIAL_H_WET
    hydro["s"][0] = cfg.INITIAL_S
    hydro["s_ex"][0] = cfg.INITIAL_S
    hydro["s_lim"][0] = min(cfg.INITIAL_S, cfg.sfc)
    hydro["storage"][0] = cfg.INITIAL_S * cfg.n * cfg.Zr
    hydro["total_storage"][0] = hydro["storage"][0] + hydro["h_wet"][0]
    return hydro


def hydrology_step(t, P_step, PET_step, hydro, cfg):
    """Advance the Laio fluctuating-water-table hydrology by one step."""
    s_prev = hydro["s_ex"][t - 1]
    s_lim_prev = hydro["s_lim"][t - 1]
    y_cap_prev = hydro["y_cap"][t - 1]
    y_wt_prev = hydro["y_wt"][t - 1]
    h_wet_prev = hydro["h_wet"][t - 1]

    # If ponded water is already present, rainfall falls directly onto the
    # wetland surface and PET evaporates ponded water before drawing from
    # soil/groundwater.

    surface_rain_step = P_step if h_wet_prev > 0.0 else 0.0

    P_soil_step = 0.0 if h_wet_prev > 0.0 else P_step

    ET_surface_step = min(
        PET_step,
        h_wet_prev + surface_rain_step
    )

    surface_water_preoverflow = (
        h_wet_prev
        + surface_rain_step
        - ET_surface_step
    )

    PET_soil_step = max(
        PET_step - ET_surface_step,
        0.0
    )

    R_step = 0.0
    R_lm_step = 0.0
    L_lm_step = 0.0
    ET_lm_step = 0.0
    ET_wt_step = 0.0
    ExfilSat_step = 0.0
    PET_lm_step = 0.0
    excess_R = 0.0
    h_runoff_step = 0.0

    # Case 1: deep water table; unsaturated/low-moisture zone forms
    if y_cap_prev < cfg.z_cr - 0.01:
        threshold = -5.0 * cfg.RD_upland - cfg.psi_fc + cfg.psi_s

        if y_cap_prev <= threshold:
            hydro["y_hm"][t] = y_cap_prev + cfg.psi_fc - cfg.psi_s
        else:
            den_hm = (-cfg.z_cr - cfg.psi_fc + cfg.psi_s)
            if abs(den_hm) < 1e-12:
                den_hm = np.sign(den_hm) * 1e-12 if den_hm != 0 else 1e-12

            hydro["y_hm"][t] = (
                (1.0 - cfg.A_upland**0.75) * (y_cap_prev - cfg.z_cr)
                - cfg.A_upland**2.0
                * (1.0 - cfg.A_upland**(-0.25))
                / den_hm
                * (y_cap_prev - cfg.z_cr) ** 2.0
            )

        ET_wt_step = PET_soil_step * np.exp(
            y_cap_prev / cfg.RD_upland
        )
        R_step = 0.0

        lmz_depth = max(abs(hydro["y_hm"][t]), 1e-12)
        hydro["lmz_stor"][t] = max(
            0.0, cfg.n * lmz_depth * (1.0 - s_prev)
        )

        PET_lm_step = max(
            PET_soil_step - ET_wt_step,
            0.0
        )

        if s_prev > cfg.sfc:
            ET_lm_step = PET_lm_step
        elif s_prev > cfg.sw:
            ET_lm_step = max(
                0.0,
                PET_lm_step
                * (s_lim_prev - cfg.sw)
                / max(cfg.sfc - cfg.sw, 1e-12),
            )
        else:
            ET_lm_step = 0.0

        R_lm_step = P_soil_step
        L_lm_step = max(
            0.0,
            (s_prev - cfg.sfc) * cfg.n * lmz_depth,
        )

        ExfilSat_step = PET_soil_step * (
            np.exp(hydro["y_hm"][t] / cfg.RD_upland)
            - np.exp(y_cap_prev / cfg.RD_upland)
        )

        excess_R = max(
            0.0,
            P_soil_step - hydro["lmz_stor"][t]
        )

        if y_wt_prev <= cfg.z_cl:
            ET_wt_step = 0.0
            R_step = P_soil_step
            ExfilSat_step = 0.0

    # Case 2: shallow water table; saturated/wetland-like condition
    else:
        ET_wt_step = PET_soil_step
        R_step = P_soil_step
        hydro["y_hm"][t] = 0.0
        ExfilSat_step = PET_soil_step - ET_wt_step
        L_lm_step = 0.0
        excess_R = 0.0
        PET_lm_step = 0.0

    # Water table and capillary fringe update
    hydro["y_cap"][t] = max(
        cfg.z_cl,
        y_cap_prev
        + (R_step + L_lm_step - ET_wt_step - ExfilSat_step)
        / max(cfg.Sy_soil, 1e-12),
    )
    hydro["y_wt"][t] = hydro["y_cap"][t] - cfg.psi_s

    # Water that would previously have become immediate runoff
    # is now routed first into above-ground wetland storage.

    surface_excess_step = 0.0
    saturation_excess_step = 0.0

    # Water table rises above land surface
    if hydro["y_wt"][t] > cfg.z_datum:

        surface_excess_step += (
            hydro["y_wt"][t]
            - cfg.z_datum
            + excess_R
        ) * cfg.Sy_soil

        hydro["y_wt"][t] = cfg.z_datum


    # Capillary fringe rises above land surface
    if hydro["y_cap"][t] > cfg.z_datum:

        surface_excess_step += (
            hydro["y_cap"][t]
            - cfg.z_datum
            + excess_R
        ) * cfg.Sy_soil

        hydro["y_cap"][t] = cfg.z_datum

    # Low-moisture-zone water content update
    if hydro["y_hm"][t] != 0:
        hydro["ds"][t] = (
            R_lm_step + ExfilSat_step - ET_lm_step - L_lm_step
        ) / abs(hydro["y_hm"][t]) / cfg.n
    else:
        hydro["ds"][t] = 0.0

    hydro["s_ex"][t] = max(s_prev + hydro["ds"][t], 0.0)
    hydro["s_lim"][t] = min(hydro["s_ex"][t], cfg.sfc)

    if hydro["s_ex"][t] > 1.0:

        saturation_excess_step = (
            (1.0 - hydro["s_ex"][t])
            * hydro["y_hm"][t]
            * cfg.n
        )

        hydro["s_ex"][t] = 1.0

    h_runoff_step = saturation_excess_step

    # Above-ground wetland storage
  
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
    h_runoff_step += max(
        0.0,
        wetland_available
        - cfg.WETLAND_STORAGE_MAX
    )

    hydro["s"][t] = np.clip(hydro["s_ex"][t], 0.0, 1.0)

    # Store hydrologic fluxes/states
    hydro["P_flux"][t] = P_step
    hydro["PET_flux"][t] = PET_step
    hydro["PET_lm_flux"][t] = PET_lm_step
    hydro["ET_lm_flux"][t] = ET_lm_step
    hydro["ET_wt_flux"][t] = ET_wt_step
    hydro["R_flux"][t] = R_step
    hydro["R_lm_flux"][t] = R_lm_step
    hydro["L_lm_flux"][t] = L_lm_step
    hydro["ExfilSat_flux"][t] = ExfilSat_step
    hydro["runoff_flux"][t] = h_runoff_step

    hydro["ET_surface_flux"][t] = ET_surface_step

    hydro["wetland_input_flux"][t] = (
        surface_excess_step
        + surface_rain_step
    )

    hydro["storage"][t] = (
        hydro["s"][t]
        * cfg.n
        * cfg.Zr
    )

    hydro["total_storage"][t] = (
        hydro["storage"][t]
        + hydro["h_wet"][t]
    )

    water_vol = max(hydro["s"][t] * cfg.n * cfg.Zr, 1e-12)
    return {
        "s_for_biogeochem": hydro["s"][t],
        "water_vol": water_vol,
        "Lrate_hydro": L_lm_step / water_vol,
        "Trate_hydro": ET_lm_step / water_vol,
    }
