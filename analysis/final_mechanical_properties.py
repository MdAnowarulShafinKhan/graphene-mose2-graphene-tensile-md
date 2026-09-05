"""
Combined stress-strain analysis.

Outputs:
- Ultimate tensile stress (UTS) and strain at UTS
- Post-peak fracture strain/stress detection
- Young's modulus from an automatically selected initial linear region, limited to <= 3% strain
- 0.2% offset yield stress/strain, with documented fallbacks
- Modulus of resilience (area under the actual stress-strain curve to yield)
- Modulus of toughness (area under the actual stress-strain curve to fracture)
- Text, CSV, and DAT summaries
- Full stress-strain, toughness-area, and resilience-area plots

Expected input by default:
- strain.dat in the same directory as this script
- strain in numeric column 1 and 2D stress in N/m in numeric column 2
"""

from pathlib import Path
import csv
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "strain.dat"

# For 2D materials: numerical conversion N/m -> GPa is stress_N_per_m / thickness_nm.
thickness_nm = 1.3196

# Numeric-column positions after extracting numbers from each line.
# Example: if each row is "timestep strain stress", use 1 and 2.
strain_index = 0
stress_index = 1

# ----- Young's modulus settings -----
# Search only within the first 3% strain, as commonly stated in the methodology.
ym_search_max_strain = 0.030
ym_candidate_start_strain = 0.002
ym_candidate_step = 0.001
ym_r2_threshold = 0.995
minimum_fit_points = 50
force_elastic_fit_through_origin = False

# ----- Yield / resilience settings -----
offset_strain = 0.002          # 0.2% offset method
yield_deviation_fraction = 0.02  # fallback: 2% departure from the fitted elastic line

# ----- Fracture / toughness settings -----
near_zero_fraction_of_UTS = 0.02
hold_points = 5

# ----- Output files -----
full_plot_file = BASE_DIR / "stress_strain.png"
toughness_plot_file = BASE_DIR / "modulus_of_toughness_area.png"
resilience_plot_file = BASE_DIR / "modulus_of_resilience_area.png"
summary_txt_file = BASE_DIR / "stress_strain_summary.txt"
summary_csv_file = BASE_DIR / "stress_strain_summary.csv"
mechanical_properties_file = BASE_DIR / "mechanical_properties.dat"


# ============================================================
# Validation
# ============================================================
if thickness_nm <= 0:
    raise ValueError("thickness_nm must be greater than zero.")
if strain_index < 0 or stress_index < 0:
    raise ValueError("strain_index and stress_index must be non-negative integers.")
if ym_search_max_strain <= 0:
    raise ValueError("ym_search_max_strain must be positive.")
if ym_candidate_step <= 0:
    raise ValueError("ym_candidate_step must be positive.")
if ym_candidate_start_strain <= 0 or ym_candidate_start_strain > ym_search_max_strain:
    raise ValueError("ym_candidate_start_strain must lie in (0, ym_search_max_strain].")
if not (0 < ym_r2_threshold <= 1):
    raise ValueError("ym_r2_threshold must lie in (0, 1].")
if minimum_fit_points < 3:
    raise ValueError("minimum_fit_points must be at least 3.")
if offset_strain <= 0:
    raise ValueError("offset_strain must be positive.")
if not (0 < near_zero_fraction_of_UTS < 1):
    raise ValueError("near_zero_fraction_of_UTS must lie in (0, 1).")
if hold_points < 1:
    raise ValueError("hold_points must be at least 1.")
if not file_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {file_path}\n"
        "Place strain.dat in the same folder as this script, or change file_path."
    )


# ============================================================
# Robust data reader
# ============================================================
raw = file_path.read_bytes().replace(b"\x00", b"")
text = raw.decode("utf-8", errors="ignore")
text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")

float_re = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")

rows = []
for line in text.splitlines():
    s = line.strip()
    if not s or s.startswith(("#", "@")):
        continue

    nums = float_re.findall(s)
    if len(nums) <= max(strain_index, stress_index):
        continue

    a = nums[strain_index].replace("D", "E").replace("d", "e")
    b = nums[stress_index].replace("D", "E").replace("d", "e")

    try:
        eps = float(a)
        sig = float(b)
    except ValueError:
        continue

    if np.isfinite(eps) and np.isfinite(sig):
        rows.append((eps, sig))

if len(rows) < 3:
    sample = [ln for ln in text.splitlines() if ln.strip()][:10]
    print("Could not parse at least 3 numeric strain/stress rows.")
    print("First decoded non-empty lines (preview):")
    for ln in sample:
        print(repr(ln[:160]))
    raise SystemExit(
        "Check strain_index/stress_index. If timestep is the first numeric column, "
        "try strain_index=1 and stress_index=2."
    )

arr = np.asarray(rows, dtype=float)
strain = arr[:, 0]
stress_npm = arr[:, 1]

# Sort by strain. Stable sorting preserves original order among duplicate strains.
order = np.argsort(strain, kind="mergesort")
strain = strain[order]
stress_npm = stress_npm[order]
stress_gpa = stress_npm / thickness_nm

if np.any(np.diff(strain) < 0):
    raise RuntimeError("Strain sorting failed unexpectedly.")
if np.unique(strain).size < 3:
    raise RuntimeError("At least 3 distinct strain values are required.")

# Warn because repeated strain values can make interpolation ambiguous.
duplicate_strain_count = int(len(strain) - np.unique(strain).size)
if duplicate_strain_count:
    warnings.warn(
        f"Found {duplicate_strain_count} repeated strain value(s). "
        "Regression remains valid, but interpolation uses the sorted data directly."
    )


# ============================================================
# Helpers
# ============================================================
def line_fit(x, y, force_origin=False):
    """Return slope, intercept, R^2 for a first-order least-squares fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if force_origin:
        denom = float(np.dot(x, x))
        if denom <= 0:
            raise ValueError("Cannot fit through origin because sum(x^2) is zero.")
        slope = float(np.dot(x, y) / denom)
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)

    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return slope, intercept, r2


def interpolate_sorted(xq, x, y):
    """Interpolate y(xq) after collapsing duplicate x values by their mean y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size == x.size:
        unique_y = y
    else:
        sums = np.zeros(unique_x.size, dtype=float)
        counts = np.zeros(unique_x.size, dtype=float)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        unique_y = sums / counts
    return float(np.interp(xq, unique_x, unique_y))


def curve_segment(x, y, x_start, x_end, y_end_override=None):
    """Build a curve segment with interpolated boundary points for integration."""
    if x_end <= x_start:
        raise ValueError("Integration end strain must exceed start strain.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x_start < x[0] or x_end > x[-1]:
        raise ValueError(
            f"Requested integration range [{x_start}, {x_end}] lies outside "
            f"available strain range [{x[0]}, {x[-1]}]."
        )

    mask = (x > x_start) & (x < x_end)
    xs = x[mask]
    ys = y[mask]

    y_start = interpolate_sorted(x_start, x, y)
    if y_end_override is None:
        y_end = interpolate_sorted(x_end, x, y)
    else:
        y_end = float(y_end_override)

    xs = np.concatenate(([x_start], xs, [x_end]))
    ys = np.concatenate(([y_start], ys, [y_end]))
    return xs, ys


# ============================================================
# Ultimate tensile stress (UTS)
# ============================================================
peak_idx = int(np.argmax(stress_gpa))
uts_gpa = float(stress_gpa[peak_idx])
uts_npm = float(stress_npm[peak_idx])
strain_at_uts = float(strain[peak_idx])

if uts_gpa <= 0:
    raise RuntimeError("Maximum stress is non-positive; check stress column/sign/conversion.")


# ============================================================
# Fracture detection for modulus of toughness
# ============================================================
fracture_idx = None
fracture_strain = None
fracture_stress_gpa = None
fracture_detection_method = None

# Prefer a clear positive-to-nonpositive zero crossing after UTS.
for i in range(peak_idx + 1, len(stress_gpa)):
    s1 = float(stress_gpa[i - 1])
    s2 = float(stress_gpa[i])
    if s1 > 0.0 and s2 <= 0.0:
        e1 = float(strain[i - 1])
        e2 = float(strain[i])
        if s2 != s1:
            fracture_strain = float(e1 + (0.0 - s1) * (e2 - e1) / (s2 - s1))
        else:
            fracture_strain = e2
        fracture_idx = i
        fracture_stress_gpa = 0.0
        fracture_detection_method = "post-UTS zero-crossing"
        break

# Otherwise look for a sustained near-zero region after UTS.
if fracture_strain is None:
    threshold = near_zero_fraction_of_UTS * uts_gpa
    last_start = len(stress_gpa) - hold_points + 1
    for i in range(peak_idx + 1, max(peak_idx + 1, last_start)):
        window = stress_gpa[i:i + hold_points]
        if len(window) == hold_points and np.all(np.abs(window) <= threshold):
            fracture_strain = float(strain[i])
            fracture_idx = i
            fracture_stress_gpa = float(stress_gpa[i])
            fracture_detection_method = (
                f"first sustained post-UTS region within +/-{near_zero_fraction_of_UTS:.1%} of UTS"
            )
            break

# If failure is not visible in the file, integrate to the final point but report that explicitly.
if fracture_strain is None:
    fracture_idx = len(strain) - 1
    fracture_strain = float(strain[fracture_idx])
    fracture_stress_gpa = float(stress_gpa[fracture_idx])
    fracture_detection_method = "end of file; no post-UTS zero/near-zero fracture region detected"

if fracture_strain <= 0:
    raise RuntimeError("Detected fracture strain is not positive; check the input curve.")

# Toughness: area under the actual stress-strain curve from zero strain to fracture.
# For an interpolated zero-crossing fracture, force the final stress to exactly zero.
if strain[0] > 0:
    toughness_start_strain = float(strain[0])
    toughness_start_note = "integration starts at first available positive strain because strain=0 is absent"
else:
    toughness_start_strain = 0.0
    toughness_start_note = "integration starts at strain=0"

y_end_override = 0.0 if fracture_detection_method == "post-UTS zero-crossing" else None
strain_tough, stress_tough = curve_segment(
    strain,
    stress_gpa,
    toughness_start_strain,
    fracture_strain,
    y_end_override=y_end_override,
)
toughness_gpa = float(np.trapezoid(stress_tough, strain_tough))
toughness_J_m3 = toughness_gpa * 1e9


# ============================================================
# Young's modulus: automatically select the initial linear region <= 3%
# ============================================================
# Never let the initial elastic fit extend beyond UTS. Usually UTS is well above 3%,
# so the effective cap remains 3%; this safeguard matters for very brittle curves.
ym_effective_max_strain = min(ym_search_max_strain, strain_at_uts)
if ym_effective_max_strain <= 0:
    raise RuntimeError("UTS occurs at non-positive strain; cannot define an initial tensile fit.")

ym_scan_mask = (strain >= 0.0) & (strain <= ym_effective_max_strain)
ym_scan_strain = strain[ym_scan_mask]
ym_scan_stress = stress_gpa[ym_scan_mask]

if ym_scan_strain.size < 3 or np.unique(ym_scan_strain).size < 2:
    raise RuntimeError(
        f"Not enough data in 0 <= strain <= {ym_effective_max_strain:.3f} "
        "to estimate Young's modulus."
    )

# Include the effective endpoint explicitly despite floating-point stepping.
if ym_effective_max_strain < ym_candidate_start_strain:
    candidate_ends = np.array([ym_effective_max_strain], dtype=float)
else:
    candidate_ends = np.arange(
        ym_candidate_start_strain,
        ym_effective_max_strain + 0.5 * ym_candidate_step,
        ym_candidate_step,
    )
    candidate_ends = candidate_ends[
        candidate_ends <= ym_effective_max_strain + 1e-12
    ]
    if candidate_ends.size == 0 or not np.isclose(
        candidate_ends[-1], ym_effective_max_strain
    ):
        candidate_ends = np.append(candidate_ends, ym_effective_max_strain)
candidate_ends = np.unique(np.round(candidate_ends, 12))

fits = []
for smax_candidate in candidate_ends:
    mask = (strain >= 0.0) & (strain <= smax_candidate)
    x = strain[mask]
    y = stress_gpa[mask]

    if len(x) < minimum_fit_points or np.unique(x).size < 2:
        continue

    slope, intercept, r2_candidate = line_fit(
        x, y, force_origin=force_elastic_fit_through_origin
    )
    if np.isfinite(r2_candidate):
        fits.append(
            {
                "smax": float(smax_candidate),
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2_candidate),
                "npts": int(len(x)),
            }
        )

# Sensible sparse-data fallback: use all available points up to 3% rather than fail solely
# because the configurable minimum point count is too high. The method is clearly reported.
sparse_fit_fallback = False
if not fits:
    x = ym_scan_strain
    y = ym_scan_stress
    if len(x) < 3 or np.unique(x).size < 2:
        raise RuntimeError("Insufficient distinct low-strain data for Young's modulus fitting.")
    slope, intercept, r2_candidate = line_fit(
        x, y, force_origin=force_elastic_fit_through_origin
    )
    if not np.isfinite(r2_candidate) or slope <= 0:
        raise RuntimeError(
            "Could not obtain a positive, finite Young's-modulus fit from the <=3% data."
        )
    fits = [
        {
            "smax": float(np.max(x)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2_candidate),
            "npts": int(len(x)),
        }
    ]
    sparse_fit_fallback = True

positive_fits = [fit for fit in fits if fit["slope"] > 0]
if not positive_fits:
    raise RuntimeError(
        "All candidate Young's-modulus fits have non-positive slopes. "
        "Check strain/stress columns and signs."
    )

good_fits = [fit for fit in positive_fits if fit["r2"] >= ym_r2_threshold]

if sparse_fit_fallback:
    selected_fit = positive_fits[0]
    ym_selection_method = (
        f"sparse-data fallback: all available nonnegative strain points up to {ym_effective_max_strain:.3f}"
    )
elif good_fits:
    selected_fit = max(good_fits, key=lambda fit: fit["smax"])
    ym_selection_method = (
        f"longest positive-slope initial fit with R^2 >= {ym_r2_threshold:.3f}, "
        f"restricted to strain <= {ym_effective_max_strain:.3f}"
    )
else:
    selected_fit = max(positive_fits, key=lambda fit: fit["r2"])
    ym_selection_method = (
        f"fallback: highest-R^2 positive-slope fit because no candidate reached "
        f"R^2 >= {ym_r2_threshold:.3f}"
    )

young_E = float(selected_fit["slope"])
elastic_intercept = float(selected_fit["intercept"])
ym_fit_max_strain = float(selected_fit["smax"])
ym_fit_r2 = float(selected_fit["r2"])
ym_fit_npts = int(selected_fit["npts"])

if ym_fit_r2 < ym_r2_threshold:
    warnings.warn(
        f"Selected Young's-modulus fit has R^2={ym_fit_r2:.6f}, below "
        f"the target {ym_r2_threshold:.3f}. Inspect the curve and fit plot."
    )


# ============================================================
# Yield detection and modulus of resilience
# ============================================================
elastic_line = young_E * strain + elastic_intercept
offset_line = young_E * (strain - offset_strain) + elastic_intercept
diff = stress_gpa - offset_line

# Apply the 0.2% offset construction after the offset strain, but never accept an
# intersection on the post-UTS softening branch as a yield point.
yield_search_start_strain = offset_strain
start_idx_for_yield = int(np.searchsorted(strain, yield_search_start_strain, side="left"))

yield_strain = None
yield_stress = None
yield_idx = None
yield_detection_method = None

for i in range(max(start_idx_for_yield + 1, 1), peak_idx + 1):
    d1 = float(diff[i - 1])
    d2 = float(diff[i])
    if d1 >= 0.0 and d2 <= 0.0:
        e1 = float(strain[i - 1])
        e2 = float(strain[i])
        if d2 != d1:
            yield_strain = float(e1 + (0.0 - d1) * (e2 - e1) / (d2 - d1))
        else:
            yield_strain = e2
        yield_stress = interpolate_sorted(yield_strain, strain, stress_gpa)
        yield_idx = i
        yield_detection_method = "0.2% offset yield"
        break

# Fallback: first meaningful departure from the selected elastic fit, but only after
# the accepted elastic fitting interval rather than immediately after 0.2% strain.
if yield_strain is None:
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.maximum(np.abs(elastic_line), 1e-12)
        relative_deviation = np.abs(stress_gpa - elastic_line) / denominator

    fallback_start_idx = int(
        np.searchsorted(strain, max(ym_fit_max_strain, offset_strain), side="right")
    )
    for i in range(max(fallback_start_idx, 1), peak_idx + 1):
        if relative_deviation[i] >= yield_deviation_fraction:
            yield_strain = float(strain[i])
            yield_stress = float(stress_gpa[i])
            yield_idx = i
            yield_detection_method = (
                f"fallback: first post-fit deviation >= {yield_deviation_fraction:.1%} "
                "from the selected elastic line"
            )
            break

# Final fallback: if no pre-UTS offset-yield or clear nonlinearity is found, treat the
# UTS point as the elastic limit. This is appropriate for a nearly linear brittle curve
# and avoids inventing a post-peak "yield" intersection.
if yield_strain is None:
    yield_strain = strain_at_uts
    yield_stress = uts_gpa
    yield_idx = peak_idx
    yield_detection_method = "fallback: no pre-UTS yield detected; UTS used as elastic limit"

if yield_strain <= 0:
    raise RuntimeError("Detected yield strain is not positive; check the stress-strain data.")

if strain[0] > 0:
    resilience_start_strain = float(strain[0])
    resilience_start_note = "integration starts at first available positive strain because strain=0 is absent"
else:
    resilience_start_strain = 0.0
    resilience_start_note = "integration starts at strain=0"

strain_res, stress_res = curve_segment(
    strain,
    stress_gpa,
    resilience_start_strain,
    yield_strain,
    y_end_override=yield_stress,
)
resilience_gpa = float(np.trapezoid(stress_res, strain_res))
resilience_J_m3 = resilience_gpa * 1e9

# Optional diagnostic for an ideal linear elastic triangle. The integrated value above
# remains the reported modulus of resilience because it uses the actual curve.
resilience_linear_approx_gpa = (
    float(yield_stress ** 2 / (2.0 * young_E)) if young_E > 0 else np.nan
)


# ============================================================
# Write text summary
# ============================================================
summary_lines = [
    f"Input file: {file_path}",
    f"Parsed rows: {len(strain)}",
    f"Distinct strain values: {np.unique(strain).size}",
    f"Repeated strain rows: {duplicate_strain_count}",
    "",
    f"Thickness used for conversion (nm): {thickness_nm:.6f}",
    "Stress conversion: stress_GPa = stress_N_per_m / thickness_nm",
    "",
    f"Ultimate tensile stress, UTS (N/m): {uts_npm:.10f}",
    f"Ultimate tensile stress, UTS (GPa): {uts_gpa:.10f}",
    f"Strain at UTS: {strain_at_uts:.10f}",
    "",
    f"Detected fracture strain: {fracture_strain:.10f}",
    f"Stress at detected fracture point (GPa): {fracture_stress_gpa:.10f}",
    f"Fracture detection method: {fracture_detection_method}",
    f"Toughness integration note: {toughness_start_note}",
    f"Full modulus of toughness (GPa): {toughness_gpa:.10f}",
    f"Full modulus of toughness (J/m^3): {toughness_J_m3:.10e}",
    "",
    f"Young's modulus (GPa): {young_E:.10f}",
    f"Elastic fit intercept (GPa): {elastic_intercept:.10f}",
    f"Young's modulus selected strain range: 0 to {ym_fit_max_strain:.6f}",
    f"Young's modulus fit R^2: {ym_fit_r2:.10f}",
    f"Young's modulus fit points: {ym_fit_npts}",
    f"Young's modulus selection method: {ym_selection_method}",
    f"Young's modulus configured search cap: {ym_search_max_strain:.6f}",
    f"Young's modulus effective search cap: {ym_effective_max_strain:.6f}",
    f"Force elastic fit through origin: {force_elastic_fit_through_origin}",
    "",
    f"Detected yield strain: {yield_strain:.10f}",
    f"Yield stress (GPa): {yield_stress:.10f}",
    f"Yield detection method: {yield_detection_method}",
    f"Resilience integration note: {resilience_start_note}",
    f"Modulus of resilience (GPa): {resilience_gpa:.10f}",
    f"Modulus of resilience (J/m^3): {resilience_J_m3:.10e}",
    f"Linear-elastic resilience approximation sigma_y^2/(2E) (GPa): {resilience_linear_approx_gpa:.10f}",
    "",
    f"Full stress-strain plot: {full_plot_file.name}",
    f"Toughness plot: {toughness_plot_file.name}",
    f"Resilience plot: {resilience_plot_file.name}",
    f"Mechanical properties file: {mechanical_properties_file.name}",
]
summary_txt_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


# ============================================================
# Write CSV summary
# ============================================================
csv_rows = [
    ["metric", "value", "units", "notes"],
    ["parsed_rows", len(strain), "count", ""],
    ["distinct_strain_values", np.unique(strain).size, "count", ""],
    ["repeated_strain_rows", duplicate_strain_count, "count", ""],
    ["thickness_for_conversion", f"{thickness_nm:.10f}", "nm", "used to convert N/m to GPa"],
    ["ultimate_stress", f"{uts_npm:.10f}", "N/m", ""],
    ["ultimate_stress", f"{uts_gpa:.10f}", "GPa", ""],
    ["strain_at_UTS", f"{strain_at_uts:.10f}", "strain", ""],
    ["fracture_strain", f"{fracture_strain:.10f}", "strain", fracture_detection_method],
    ["fracture_stress", f"{fracture_stress_gpa:.10f}", "GPa", fracture_detection_method],
    ["modulus_of_toughness", f"{toughness_gpa:.10f}", "GPa", "integral of actual curve to detected fracture"],
    ["modulus_of_toughness", f"{toughness_J_m3:.10e}", "J/m^3", "integral of actual curve to detected fracture"],
    ["youngs_modulus", f"{young_E:.10f}", "GPa", ym_selection_method],
    ["elastic_fit_intercept", f"{elastic_intercept:.10f}", "GPa", ""],
    ["youngs_modulus_fit_end_strain", f"{ym_fit_max_strain:.10f}", "strain", "selected linear-fit endpoint"],
    ["youngs_modulus_fit_R2", f"{ym_fit_r2:.10f}", "-", ""],
    ["youngs_modulus_fit_points", ym_fit_npts, "count", ""],
    ["yield_strain", f"{yield_strain:.10f}", "strain", yield_detection_method],
    ["yield_stress", f"{yield_stress:.10f}", "GPa", yield_detection_method],
    ["modulus_of_resilience", f"{resilience_gpa:.10f}", "GPa", "integral of actual curve to detected yield"],
    ["modulus_of_resilience", f"{resilience_J_m3:.10e}", "J/m^3", "integral of actual curve to detected yield"],
    ["resilience_linear_approximation", f"{resilience_linear_approx_gpa:.10f}", "GPa", "sigma_y^2/(2E), diagnostic only"],
]

with summary_csv_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)


# ============================================================
# Write compact mechanical_properties.dat
# ============================================================
with mechanical_properties_file.open("w", encoding="utf-8") as f:
    f.write("# Mechanical properties extracted from stress-strain data\n")
    f.write(f"# Input file: {file_path.name}\n")
    f.write("# Stress conversion: GPa = (N/m) / thickness_nm\n")
    f.write(f"# thickness_nm = {thickness_nm:.10f}\n")
    f.write("# Columns: Property  Value  Unit\n")
    f.write(f"Ultimate_Stress  {uts_gpa:.10f}  GPa\n")
    f.write(f"Strain_at_UTS  {strain_at_uts:.10f}  -\n")
    f.write(f"Fracture_Strain  {fracture_strain:.10f}  -\n")
    f.write(f"Youngs_Modulus  {young_E:.10f}  GPa\n")
    f.write(f"Yield_Strain  {yield_strain:.10f}  -\n")
    f.write(f"Yield_Stress  {yield_stress:.10f}  GPa\n")
    f.write(f"Modulus_of_Resilience  {resilience_gpa:.10f}  GPa\n")
    f.write(f"Modulus_of_Resilience  {resilience_J_m3:.10e}  J/m^3\n")
    f.write(f"Modulus_of_Toughness  {toughness_gpa:.10f}  GPa\n")
    f.write(f"Modulus_of_Toughness  {toughness_J_m3:.10e}  J/m^3\n")
    f.write(
        "# Young's modulus fit details: "
        f"strain_range=0_to_{ym_fit_max_strain:.6f}, "
        f"R2={ym_fit_r2:.10f}, npts={ym_fit_npts}, method={ym_selection_method}\n"
    )
    f.write(f"# Yield detection method: {yield_detection_method}\n")
    f.write(f"# Fracture detection method: {fracture_detection_method}\n")


# ============================================================
# Plot 1: full stress-strain curve with principal mechanical points
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(strain, stress_gpa, lw=1.8, label="Stress-strain curve")

xfit = np.linspace(0.0, ym_fit_max_strain, 200)
yfit = young_E * xfit + elastic_intercept
plt.plot(
    xfit,
    yfit,
    "--",
    lw=1.4,
    label=f"Young's modulus fit (R^2={ym_fit_r2:.4f})",
)

plt.scatter([strain_at_uts], [uts_gpa], s=50, zorder=4, label="UTS")
plt.scatter([yield_strain], [yield_stress], s=50, zorder=4, label="Yield")
plt.scatter(
    [fracture_strain],
    [fracture_stress_gpa],
    s=50,
    zorder=4,
    label="Detected fracture",
)

plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress-strain curve and extracted mechanical properties")
plt.legend()
plt.tight_layout()
plt.savefig(full_plot_file, dpi=300)
plt.close()


# ============================================================
# Plot 2: modulus of toughness area
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(strain, stress_gpa, lw=1.8, label="Stress-strain curve")
plt.fill_between(
    strain_tough,
    stress_tough,
    0.0,
    alpha=0.30,
    label="Area = modulus of toughness",
)
plt.scatter([strain_at_uts], [uts_gpa], s=50, zorder=4, label="UTS")
plt.scatter(
    [fracture_strain],
    [fracture_stress_gpa],
    s=50,
    zorder=4,
    label="Detected fracture",
)
plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress-strain curve with modulus of toughness area")
plt.legend()
plt.tight_layout()
plt.savefig(toughness_plot_file, dpi=300)
plt.close()


# ============================================================
# Plot 3: modulus of resilience area and yield construction
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(strain, stress_gpa, lw=1.8, label="Stress-strain curve")
plt.fill_between(
    strain_res,
    stress_res,
    0.0,
    alpha=0.30,
    label="Area = modulus of resilience",
)

xmax_plot = max(yield_strain * 1.25, ym_fit_max_strain * 1.5, 0.02)
mask_small = strain <= xmax_plot
plt.plot(
    strain[mask_small],
    elastic_line[mask_small],
    "--",
    lw=1.4,
    label="Selected elastic fit",
)
plt.plot(
    strain[mask_small],
    offset_line[mask_small],
    "--",
    lw=1.4,
    label="0.2% offset line",
)
plt.scatter([yield_strain], [yield_stress], s=50, zorder=4, label="Yield point")

plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress-strain curve with modulus of resilience area")
plt.legend()
plt.tight_layout()
plt.savefig(resilience_plot_file, dpi=300)
plt.close()


# ============================================================
# Console output
# ============================================================
print("Reading:", file_path)
print("Parsed rows:", len(strain))
print()
print(f"UTS = {uts_npm:.6f} N/m = {uts_gpa:.6f} GPa")
print(f"Strain at UTS = {strain_at_uts:.6f}")
print()
print(f"Detected fracture strain = {fracture_strain:.6f}")
print(f"Detected fracture stress = {fracture_stress_gpa:.6f} GPa")
print(f"Fracture detection method = {fracture_detection_method}")
print(f"Full modulus of toughness = {toughness_gpa:.6f} GPa")
print(f"Full modulus of toughness (SI) = {toughness_J_m3:.6e} J/m^3")
print()
print(f"Young's modulus = {young_E:.6f} GPa")
print(f"Elastic fit intercept = {elastic_intercept:.6f} GPa")
print(
    f"Young's modulus fit = strain 0 to {ym_fit_max_strain:.3f}, "
    f"R^2 = {ym_fit_r2:.6f}, n = {ym_fit_npts}"
)
print(f"Young's modulus selection method = {ym_selection_method}")
print()
print(f"Detected yield strain = {yield_strain:.6f}")
print(f"Yield stress = {yield_stress:.6f} GPa")
print(f"Yield detection method = {yield_detection_method}")
print(f"Modulus of resilience = {resilience_gpa:.6f} GPa")
print(f"Modulus of resilience (SI) = {resilience_J_m3:.6e} J/m^3")
print(
    "Linear-elastic resilience approximation sigma_y^2/(2E) = "
    f"{resilience_linear_approx_gpa:.6f} GPa"
)
print()
print("Saved:", full_plot_file)
print("Saved:", toughness_plot_file)
print("Saved:", resilience_plot_file)
print("Saved:", summary_txt_file)
print("Saved:", summary_csv_file)
print("Saved:", mechanical_properties_file)
