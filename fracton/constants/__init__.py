"""
Fracton Constants Package

Single source of truth for all mathematical, physical, and Standard Model
constants used in Dawn Field Theory.

Usage:
    from fracton.constants import PHI, XI, GAMMA, LN2
    from fracton.constants import G, C, HBAR, M_P
    from fracton.constants import ALPHA_EM_PAC, SIN2_THETA_W_PAC
    from fracton.constants import mathematical, physical, standard_model
"""

# Mathematical constants (derived from PAC axioms)
from .mathematical import (
    PHI, PHI_INV, PHI_SQUARED, PHI_CUBED,
    LN_PHI, LN2, LN2_SQUARED,
    GAMMA, SQRT5, TAU,
    GOLDEN_ANGLE, GOLDEN_ANGLE_FRACTION,
    XI, XI_ANALYTIC, XI_DISCRETE, XI_PAC, XI_FLOOR,
    XI_SPREAD, EULER_GAP, EULER_GAP_FACTOR,
    DUTY_CYCLE, UNIVERSAL_RATIO,
    FIBONACCI_GRAVITY_DEPTH,
)

# Physical constants (measured — CODATA/PDG)
from .physical import (
    C, G, HBAR, H, K_B,
    SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT, PLANCK_CONSTANT, BOLTZMANN,
    ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY, VACUUM_PERMEABILITY,
    M_E, M_P, M_N, M_MU, M_TAU,
    ELECTRON_MASS, PROTON_MASS, NEUTRON_MASS, MUON_MASS, TAU_MASS,
    MUON_ELECTRON_RATIO, TAU_ELECTRON_RATIO, PROTON_ELECTRON_RATIO,
    PLANCK_MASS, PLANCK_LENGTH, PLANCK_TIME, PLANCK_ENERGY,
    PLANCK_TEMPERATURE, PLANCK_DENSITY,
    W_BOSON_MASS_GEV, Z_BOSON_MASS_GEV, HIGGS_MASS_GEV,
    HUBBLE_CONSTANT, H0, OMEGA_MATTER, OMEGA_BARYON, OMEGA_CDM,
    OMEGA_LAMBDA, OMEGA_LAMBDA_UNCERTAINTY, VACUUM_ENERGY_DENSITY_LOG10,
    CMB_TEMPERATURE, AGE_OF_UNIVERSE_GYR,
    PLANCK_PROTON_RATIO, EM_GRAVITY_RATIO,
    G_UNCERTAINTY,
)

# Standard Model parameters (PAC-derived)
from .standard_model import (
    ALPHA_EM_PAC, ALPHA_EM_MEASURED, ALPHA_EM_INV_PAC, ALPHA_EM_INV_MEASURED,
    SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED,
    MW_MZ_RATIO_PAC, MW_MZ_RATIO_MEASURED,
    ALPHA_S_BARE, ALPHA_S_MEASURED, ALPHA_S_C2, ALPHA_S_C3,
    G_PAC, G_CORRECTION_K,
    OMEGA_LAMBDA_PAC,
    CC_TILING_XI, CC_TILING_XI_PAC,
    KOIDE_Q_PAC, KOIDE_Q_MEASURED,
    MUON_ELECTRON_PAC, PROTON_ELECTRON_PAC,
    THETA12_PMNS_PAC, THETA13_PMNS_PAC, THETA12_CKM_PAC,
    GAUGE_ADJOINT_DIMS, GAUGE_TOTAL, SM_IS_FIBONACCI_CLOSED,
    FORCE_SCORECARD, B_HIERARCHY,
    F,
    alpha_em_pac, alpha_s_bare, alpha_s_c2, alpha_s_c3,
    g_predicted, omega_lambda_pac, cc_tiling_log10, koide_q_measured,
)

__all__ = [
    # Mathematical
    "PHI", "PHI_INV", "PHI_SQUARED", "PHI_CUBED",
    "LN_PHI", "LN2", "LN2_SQUARED",
    "GAMMA", "SQRT5", "TAU",
    "GOLDEN_ANGLE", "GOLDEN_ANGLE_FRACTION",
    "XI", "XI_ANALYTIC", "XI_DISCRETE", "XI_PAC", "XI_FLOOR",
    "XI_SPREAD", "EULER_GAP", "EULER_GAP_FACTOR",
    "DUTY_CYCLE", "UNIVERSAL_RATIO", "FIBONACCI_GRAVITY_DEPTH",
    # Physical
    "C", "G", "HBAR", "H", "K_B", "M_E", "M_P", "M_N", "M_MU", "M_TAU",
    "PLANCK_MASS", "PLANCK_LENGTH", "PLANCK_TIME", "PLANCK_ENERGY",
    "OMEGA_LAMBDA", "OMEGA_CDM", "H0",
    # Standard Model (PAC-derived)
    "ALPHA_EM_PAC", "ALPHA_EM_MEASURED",
    "SIN2_THETA_W_PAC", "SIN2_THETA_W_MEASURED",
    "ALPHA_S_BARE", "ALPHA_S_MEASURED", "ALPHA_S_C2", "ALPHA_S_C3",
    "G_PAC", "G_CORRECTION_K",
    "OMEGA_LAMBDA_PAC",
    "KOIDE_Q_PAC", "KOIDE_Q_MEASURED",
    "FORCE_SCORECARD", "B_HIERARCHY",
    "F",
    # Sub-modules
    "mathematical", "physical", "standard_model",
]
