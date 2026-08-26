import numpy as np


def generate_rain(N, lambda_inv, alpha, dt, seed=0):
    """Generate stochastic rainfall using the same formulation as the original script."""
    np.random.seed(seed)
    rain = np.zeros(N)
    day = np.random.poisson(lambda_inv) + 1
    while day < N * dt:
        index = int(day / dt)
        if index < N:
            rain[index] += np.random.exponential(scale=alpha)
        day += np.random.poisson(lambda_inv)
    return rain


def build_seasonal_pet(N, dt, pet_min, pet_split_T, pet_phase, baseline_daily_pet):
    """Build seasonal PET and split it between potential evaporation and transpiration."""
    steps_per_year = int(round(365 / dt))
    doy = (np.arange(steps_per_year) * dt) % 365.0 + 1.0
    shape_year = np.maximum(
        0.0,
        pet_min + np.sin(np.pi * (doy / 365.0) + pet_phase),
    )

    baseline_total_year = baseline_daily_pet * 365.0
    scale = baseline_total_year / (shape_year.sum() * dt)

    shape_full = np.tile(shape_year, int(np.ceil(N / steps_per_year)))[:N]
    PET = scale * shape_full
    T_pot = PET * pet_split_T
    E_pot = PET * (1.0 - pet_split_T)
    return E_pot, T_pot, PET


def generate_climate(cfg):
    rain = generate_rain(
        cfg.N, cfg.lambda_inv, cfg.alpha, cfg.dt, seed=cfg.RAIN_SEED
    )

    if cfg.USE_SEASONAL_PET:
        Ew_series, Tmax_series, PET_series = build_seasonal_pet(
            N=cfg.N,
            dt=cfg.dt,
            pet_min=cfg.PET_MIN,
            pet_split_T=cfg.PET_SPLIT_T,
            pet_phase=cfg.PET_PHASE,
            baseline_daily_pet=cfg.BASELINE_PET_DAILY,
        )
    else:
        Ew_series = np.full(cfg.N, cfg.Ew)
        Tmax_series = np.full(cfg.N, cfg.Tmax)
        PET_series = Ew_series + Tmax_series

    return rain, Ew_series, Tmax_series, PET_series
