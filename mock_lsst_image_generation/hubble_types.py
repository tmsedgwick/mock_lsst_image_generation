"""Hubble types for the mock galaxies, drawn with probabilities taken from published measurements.

Each galaxy's type is drawn at random from P(type | stellar mass, redshift, star-forming or passive), so every type
occurs wherever the measured mass functions allow it. The steps, and the source behind each:

1. Broad morphology, from Huertas-Company et al. 2016 (HC16), MNRAS 462, 4495 (arXiv:1606.04952), Table 2: Schechter
   fits to the stellar mass functions of star-forming and quiescent CANDELS galaxies, separately for spheroids (SPH),
   disc+spheroid systems (DS), discs (DISK) and irregulars (IRR), in seven bins over 0.2 < z < 3. For a galaxy of
   mass M in state s, P(class | M, z, s) = phi_class,s(M, z) / sum over classes of phi_class,s(M, z).
   Some fits in the table are unconstrained (log M* > 12 with phi* printed as 0.00), and three rows of the
   2.5 < z < 3 bin repeat the 0.2 < z < 0.5 row exactly (a copying error in the paper); for those the class takes the
   mass-function shape of all galaxies in the same state and bin, scaled by the class's share of the galaxy counts
   the table also gives. log phi is interpolated linearly in z between bin centres (end bins beyond them). Outside
   the masses a bin measured (below its completeness limit, and above where HC16's survey has fewer than
   MIN_PER_DEX galaxies per dex) probabilities are held at their value at the edge instead of extrapolating fits.
2. E from SPH; S0 or Sa from DS and Sb or Sc from DISK in the ratios of the visual T-types of Nair & Abraham 2010,
   ApJS 186, 427, Table 3 (14,034 SDSS galaxies), using the class boundaries of Kelvin et al. 2014 (S0-Sa; Sab-Scd),
   with S0/a and Sbc split evenly between their neighbours; IRR becomes Irr.
3. Bars (S0, Sa, Sb, Sc): the mass dependence of Erwin 2018, MNRAS 474, 5372 (S4G logistic fit, peak 0.70 at
   log M = 9.7), halving between z = 0.4 and 1.0 as measured by Melvin et al. 2014, MNRAS 438, 2882, and held at
   half beyond.
4. Ellipticals become En with n = 10 x their observed ellipticity (Hubble's definition), capped at E7.

Kelvin et al. 2014 (MNRAS 444, 1647) fractions at z ~ 0.04 are an independent check (tests/test_hubble_types.py).

The arm, bar and irregularity parameters drawn for rendering (draw_structure) are visual choices, not yet from the
literature: pitch angle and arm contrast grow from Sa to Sc, as observed.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

BROAD = ["SPH", "DS", "DISK", "IRR"]
Z_BINS = ["0.2-0.5", "0.5-0.8", "0.8-1.1", "1.1-1.5", "1.5-2", "2-2.5", "2.5-3"]
Z_CENTRES = np.array([np.mean([float(v) for v in b.split("-")]) for b in Z_BINS])

# HC16 Table 2, verbatim: (class, z bin, star-forming (N, log M_complete, log M*, phi* [1e-3 Mpc^-3 dex^-1], alpha),
# quiescent (same)). 99.99 = not fitted.
HC16_TABLE2 = [
    ("all", "0.2-0.5", (4465, 8.43, 10.68, 0.92, -1.46), (768, 8.60, 11.01, 0.56, -1.09)),
    ("all", "0.5-0.8", (7032, 8.94, 10.92, 0.83, -1.46), (1244, 9.04, 10.90, 1.47, -0.68)),
    ("all", "0.8-1.1", (6743, 9.29, 10.93, 0.81, -1.41), (859, 9.48, 10.84, 1.17, -0.40)),
    ("all", "1.1-1.5", (6534, 9.61, 10.98, 0.51, -1.41), (565, 9.79, 10.73, 0.64, 0.00)),
    ("all", "1.5-2", (6261, 10.02, 10.82, 0.93, -1.00), (541, 10.16, 10.78, 0.44, 0.00)),
    ("all", "2-2.5", (3961, 10.21, 11.29, 0.13, -1.61), (211, 10.51, 10.85, 0.18, -0.56)),
    ("all", "2.5-3", (2057, 10.36, 11.02, 0.19, -1.08), (90, 11.05, 99.99, 99.99, 99.99)),
    ("SPH", "0.2-0.5", (418, 8.43, 15.58, 0.00, -1.71), (202, 8.60, 11.04, 0.28, -0.96)),
    ("SPH", "0.5-0.8", (798, 8.94, 14.37, 0.00, -1.77), (586, 9.04, 10.88, 0.85, -0.61)),
    ("SPH", "0.8-1.1", (872, 9.29, 10.86, 0.10, -1.46), (470, 9.48, 10.81, 0.62, -0.41)),
    ("SPH", "1.1-1.5", (928, 9.61, 10.72, 0.16, -1.20), (331, 9.79, 10.52, 0.39, 0.29)),
    ("SPH", "1.5-2", (889, 10.02, 10.58, 0.27, -0.38), (316, 10.16, 10.35, 0.14, 1.68)),
    ("SPH", "2-2.5", (568, 10.21, 11.00, 0.06, -1.08), (120, 10.51, 10.72, 0.13, -0.14)),
    ("SPH", "2.5-3", (248, 10.36, 15.58, 0.00, -1.71), (39, 11.05, 99.99, 99.99, 99.99)),
    ("DS", "0.2-0.5", (233, 8.43, 10.53, 0.48, -0.85), (161, 8.60, 10.42, 0.94, -0.17)),
    ("DS", "0.5-0.8", (408, 8.94, 10.66, 0.46, -0.78), (319, 9.04, 10.63, 0.89, -0.01)),
    ("DS", "0.8-1.1", (360, 9.29, 10.94, 0.24, -0.82), (208, 9.48, 10.67, 0.43, 0.29)),
    ("DS", "1.1-1.5", (281, 9.61, 11.02, 0.11, -0.82), (113, 9.79, 10.57, 0.10, 1.29)),
    ("DS", "1.5-2", (196, 10.02, 10.91, 0.08, -0.38), (74, 10.16, 10.65, 0.03, 1.64)),
    ("DS", "2-2.5", (81, 10.21, 11.21, 0.01, -1.06), (10, 10.51, 10.56, 0.00, 1.94)),
    ("DS", "2.5-3", (28, 10.36, 10.53, 0.48, -0.85), (7, 11.05, 99.99, 99.99, 99.99)),
    ("DISK", "0.2-0.5", (1263, 8.43, 10.33, 1.21, -1.17), (202, 8.60, 16.02, 0.00, -1.55)),
    ("DISK", "0.5-0.8", (2162, 8.94, 10.66, 0.80, -1.24), (179, 9.04, 15.81, 0.00, -1.54)),
    ("DISK", "0.8-1.1", (1752, 9.29, 10.80, 0.54, -1.18), (84, 9.48, 11.52, 0.02, -1.17)),
    ("DISK", "1.1-1.5", (1126, 9.61, 11.01, 0.14, -1.30), (44, 9.79, 11.15, 0.02, -0.60)),
    ("DISK", "1.5-2", (748, 10.02, 10.84, 0.13, -0.96), (39, 10.16, 10.79, 0.04, -0.11)),
    ("DISK", "2-2.5", (299, 10.21, 15.81, 0.00, -1.76), (14, 10.51, 11.25, 0.01, 0.07)),
    ("DISK", "2.5-3", (103, 10.36, 10.33, 1.21, -1.17), (2, 11.05, 99.99, 99.99, 99.99)),
    ("IRR", "0.2-0.5", (2472, 8.43, 10.15, 0.40, -1.71), (192, 8.60, 14.48, 0.00, -2.07)),
    ("IRR", "0.5-0.8", (3460, 8.94, 10.56, 0.31, -1.66), (130, 9.04, 16.17, 0.00, -1.92)),
    ("IRR", "0.8-1.1", (3532, 9.29, 10.72, 0.33, -1.57), (81, 9.48, 18.87, 0.00, -1.57)),
    ("IRR", "1.1-1.5", (3887, 9.61, 10.86, 0.23, -1.57), (48, 9.79, 20.15, 0.00, -1.45)),
    ("IRR", "1.5-2", (4132, 10.02, 10.88, 0.37, -1.28), (65, 10.16, 19.21, 0.00, -1.59)),
    ("IRR", "2-2.5", (2776, 10.21, 10.93, 0.24, -1.28), (38, 10.51, 17.85, 0.00, -1.93)),
    ("IRR", "2.5-3", (1530, 10.36, 11.53, 0.03, -1.67), (23, 11.05, 99.99, 99.99, 99.99)),
]
HC16 = {(cls, z): dict(sf=sf, q=q) for cls, z, sf, q in HC16_TABLE2}
HC16_AREA_ARCMIN2, HC16_H0, HC16_OM = 880.0, 70.0, 0.3  # survey area (abstract) and cosmology (section 2)
MIN_PER_DEX = 10.0  # fits are trusted up to the mass where HC16's survey still has this many galaxies per dex

# Nair & Abraham 2010 Table 3: galaxies per T-type.
NA10_COUNTS = dict(S0=966, S0a=1193, Sa=1322, Sab=893, Sb=1890, Sbc=1207, Sc=1562, Scd=541)
S0_SHARE_OF_DS = (NA10_COUNTS["S0"] + NA10_COUNTS["S0a"] / 2) / sum(NA10_COUNTS[t] for t in ["S0", "S0a", "Sa"])
SB_SHARE_OF_DISK = ((NA10_COUNTS["Sab"] + NA10_COUNTS["Sb"] + NA10_COUNTS["Sbc"] / 2)
                    / sum(NA10_COUNTS[t] for t in ["Sab", "Sb", "Sbc", "Sc", "Scd"]))

ERWIN18_BAR_FIT = (-82.2, 17.1, -0.88)  # logit f_bar = a + b1 x + b2 x^2, x = log M
MELVIN14_BAR_DECLINE = (0.4, 1.0, 0.5)  # the bar fraction falls by this factor between these redshifts

# Structure drawn per type for rendering: (mean, scatter) of arm pitch angle (deg) and arm contrast, and the arms'
# sharpness (power of the arm profile). Visual choices, see the module docstring.
class SpiralStructure(NamedTuple):
    pitch: tuple[float, float]
    strength: tuple[float, float]
    sharpness: float


SPIRAL_STRUCTURE = dict(a=SpiralStructure(pitch=(10, 2), strength=(0.45, 0.1), sharpness=1.5),
                        b=SpiralStructure(pitch=(16, 3), strength=(0.55, 0.1), sharpness=2.0),
                        c=SpiralStructure(pitch=(24, 4), strength=(0.65, 0.08), sharpness=2.5))
BAR_RADIUS_H = 1.6  # arms start at the bar's end, in disc scale lengths; unbarred arms start at ARM_START_H
ARM_START_H = 0.5
IRREGULARITY = (0.5, 0.1)  # amplitude of an Irr's random low-order modes
# Low-mass discs are Magellanic (Sm) rather than grand-design: between these log masses the spiral arms fade out and
# irregular modes fade in (visual choice). Bars are kept, as measured (Erwin 2018); discs mostly faded are labelled
# Sm / SBm.
DWARF_SPIRAL_MASS = (8.5, 9.5)


def schechter(logM, log_mstar, phi_star, alpha):
    """Number density per dex of a Schechter function."""
    x = 10.0 ** (logM - log_mstar)
    return np.log(10) * phi_star * x ** (alpha + 1) * np.exp(-x)


def usable(cls, z_bin, state):
    """Whether HC16's fit for this class, bin and state is constrained (see the module docstring)."""
    fit = HC16[cls, z_bin][state]
    repeats_first_bin = z_bin == "2.5-3" and fit[2:] == HC16[cls, "0.2-0.5"][state][2:]
    return fit[2] < 12 and fit[3] > 0 and not repeats_first_bin


def fitted_bin(z_bin, state):
    """This bin, or the nearest lower one with a fit for all galaxies (quiescent 2.5 < z < 3 has none)."""
    while not usable("all", z_bin, state):
        z_bin = Z_BINS[Z_BINS.index(z_bin) - 1]
    return z_bin


def class_density(logM, cls, z_bin, state):
    """HC16 number density per dex of one broad class at log masses logM, in one redshift bin and state."""
    if usable(cls, z_bin, state):
        return schechter(logM, *HC16[cls, z_bin][state][2:])
    share = HC16[cls, z_bin][state][0] / HC16["all", z_bin][state][0]
    return share * schechter(logM, *HC16["all", fitted_bin(z_bin, state)][state][2:])


def comoving_volume(z_lo, z_hi, area_arcmin2):
    """Comoving volume (Mpc^3) between two redshifts over an area, in HC16's flat LCDM cosmology."""
    z = np.linspace(0, z_hi, 2001)
    distance = 2.99792458e5 / HC16_H0 * np.concatenate(
        [[0], np.cumsum(np.diff(z) / np.sqrt(HC16_OM * (1 + z[1:]) ** 3 + 1 - HC16_OM))])
    sky_fraction = area_arcmin2 / (4 * np.pi * (180 * 60 / np.pi) ** 2)
    return sky_fraction * 4 / 3 * np.pi * (np.interp(z_hi, z, distance) ** 3 - np.interp(z_lo, z, distance) ** 3)


def mass_range(z_bin, state):
    """(lowest, highest) log mass HC16 measured in this bin and state: the completeness limit, and where the mass
    function of all galaxies drops below MIN_PER_DEX galaxies per dex in their survey."""
    z_lo, z_hi = (float(v) for v in z_bin.split("-"))
    volume = comoving_volume(z_lo, z_hi, HC16_AREA_ARCMIN2)
    logM = np.arange(9.0, 12.5, 0.01)
    per_dex = schechter(logM, *HC16["all", fitted_bin(z_bin, state)][state][2:]) * 1e-3 * volume
    return HC16["all", z_bin][state][1], float(logM[np.argmax(per_dex < MIN_PER_DEX)])


MASS_RANGES = {(b, state): mass_range(b, state) for b in Z_BINS for state in ["sf", "q"]}


def broad_probabilities(logM, z, passive):
    """(N, 4) probabilities of SPH, DS, DISK, IRR for galaxies of log mass logM at redshift z."""
    logM, z, passive = np.atleast_1d(logM), np.atleast_1d(z), np.atleast_1d(passive)
    log_phi = np.empty((len(logM), len(Z_BINS), len(BROAD)))
    for j, z_bin in enumerate(Z_BINS):
        m_sf, m_q = np.clip(logM, *MASS_RANGES[z_bin, "sf"]), np.clip(logM, *MASS_RANGES[z_bin, "q"])
        for k, cls in enumerate(BROAD):
            density = np.where(passive, class_density(m_q, cls, z_bin, "q"), class_density(m_sf, cls, z_bin, "sf"))
            log_phi[:, j, k] = np.log(density + 1e-300)
    zc = np.clip(z, Z_CENTRES[0], Z_CENTRES[-1])
    upper = np.clip(np.searchsorted(Z_CENTRES, zc), 1, len(Z_BINS) - 1)
    t = ((zc - Z_CENTRES[upper - 1]) / (Z_CENTRES[upper] - Z_CENTRES[upper - 1]))[:, None]
    rows = np.arange(len(logM))
    log_p = (1 - t) * log_phi[rows, upper - 1] + t * log_phi[rows, upper]
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def bar_probability(logM, z):
    """Erwin 2018 bar fraction at mass logM, scaled by the Melvin et al. 2014 decline with redshift."""
    a, b1, b2 = ERWIN18_BAR_FIT
    z0, z1, factor = MELVIN14_BAR_DECLINE
    return factor ** np.clip((z - z0) / (z1 - z0), 0, 1) / (1 + np.exp(-(a + b1 * logM + b2 * logM ** 2)))


def draw_hubble_types(logM, z, passive, ellipticity, rng):
    """Hubble type (E0-E7, S0, SB0, Sa, Sb, Sc, SBa, SBb, SBc, Irr) for each galaxy."""
    p = broad_probabilities(logM, z, passive)
    broad = np.array(BROAD)[np.minimum((rng.random((len(p), 1)) > np.cumsum(p, axis=1)).sum(axis=1), 3)]
    u = rng.random(len(p))
    en = np.clip(np.round(10 * np.nan_to_num(ellipticity)), 0, 7).astype(int)
    types = np.select([broad == "SPH", broad == "DS", broad == "DISK"],
                      [np.char.add("E", en.astype(str)), np.where(u < S0_SHARE_OF_DS, "S0", "Sa"),
                       np.where(u < SB_SHARE_OF_DISK, "Sb", "Sc")], "Irr").astype(object)
    barred = np.isin(broad, ["DS", "DISK"]) & (rng.random(len(p)) < bar_probability(logM, z))
    types[barred] = ["SB" + t[1:] for t in types[barred]]
    return types


def spiral_weight(logM):
    """How much of a spiral's grand-design structure a disc of this mass keeps: 0 below DWARF_SPIRAL_MASS[0], 1 above
    DWARF_SPIRAL_MASS[1]."""
    lo, hi = DWARF_SPIRAL_MASS
    return np.clip((np.asarray(logM, float) - lo) / (hi - lo), 0.0, 1.0)


def draw_structure(types, rng, logM=None):
    """Rendering parameters for each galaxy's type: arm count, pitch (deg), strength, sharpness and phase (deg), the
    radius where arms start (disc scale lengths), bar flag, irregularity amplitude and a seed for the Irr modes. With
    logM, low-mass spirals trade their arms for irregular modes (see DWARF_SPIRAL_MASS)."""
    n = len(types)
    out = pd.DataFrame(dict(n_arms=0, arm_pitch_deg=20.0, arm_strength=0.0, arm_sharpness=1.0,
                            arm_phase_deg=rng.uniform(0, 360, n), arm_start_h=ARM_START_H,
                            barred=[t.startswith("SB") for t in types], irregularity=0.0,
                            structure_seed=rng.integers(0, 2**31, n)))
    out.loc[out["barred"], "arm_start_h"] = BAR_RADIUS_H
    for stage, s in SPIRAL_STRUCTURE.items():
        rows = np.flatnonzero([t in (f"S{stage}", f"SB{stage}") for t in types])
        out.loc[rows, "n_arms"] = 2
        out.loc[rows, "arm_pitch_deg"] = np.clip(rng.normal(*s.pitch, len(rows)), 5, 40)
        out.loc[rows, "arm_strength"] = np.clip(rng.normal(*s.strength, len(rows)), 0.2, 0.8)
        out.loc[rows, "arm_sharpness"] = float(s.sharpness)
    irregular = np.flatnonzero(types == "Irr")
    out.loc[irregular, "irregularity"] = np.clip(rng.normal(*IRREGULARITY, len(irregular)), 0.1, 0.9)
    spirals = np.flatnonzero(out["n_arms"].to_numpy() > 0)
    patchiness = np.clip(rng.normal(*IRREGULARITY, len(spirals)), 0.1, 0.9)  # drawn even when unused: stable stream
    if logM is not None:
        w = spiral_weight(np.asarray(logM, float)[spirals])
        out.loc[spirals, "arm_strength"] *= w
        out.loc[spirals, "irregularity"] = (1 - w) * patchiness
        out.loc[spirals[w == 0], "n_arms"] = 0
    return out


def add_hubble_types(table, rng):
    """Catalogue table with hubble_type and its rendering structure columns added to every galaxy row (star rows get
    none). Spirals below the middle of DWARF_SPIRAL_MASS are labelled Sm / SBm."""
    galaxies = (table["type"] != "star").to_numpy()
    gal = table.loc[galaxies]
    types = draw_hubble_types(gal["logM"].to_numpy(float), gal["z"].to_numpy(float),
                              (gal["type"] == "passive").to_numpy(), gal["ellipticity_total"].to_numpy(float), rng)
    logM = gal["logM"].to_numpy(float)
    structure = draw_structure(types, rng, logM).set_index(gal.index)
    magellanic = np.isin(types, ["Sa", "Sb", "Sc", "SBa", "SBb", "SBc"]) & (spiral_weight(logM) < 0.5)
    types[magellanic] = [t[:-1] + "m" for t in types[magellanic]]
    out = table.copy()
    out.loc[galaxies, "hubble_type"] = types
    for column in structure:
        out.loc[galaxies, column] = structure[column]
    return out
