import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# =========================
# User inputs
# =========================
thickness_nm = 1.3196

# strain.dat must be in the SAME folder as this script
file_path = Path(__file__).with_name("strain.dat")

# If your file has an extra numeric column first (e.g., timestep),
# set strain_index=1 and stress_index=2.
strain_index = 0
stress_index = 1

# ----- fracture detection settings (for FULL modulus of toughness) -----
near_zero_fraction_of_UTS = 0.02   # 2% of UTS
hold_points = 5                    # require this many consecutive near-zero points

# ----- yield/resilience settings -----
offset_strain = 0.002              # 0.2% offset
elastic_fit_max_strain = 0.010     # fit the initial linear region up to this strain
force_elastic_fit_through_origin = False
yield_deviation_fraction = 0.02    # fallback: 2%

# Output files
toughness_plot_file = Path(__file__).with_name("modulus_of_toughness_area.png")
resilience_plot_file = Path(__file__).with_name("modulus_of_resilience_area.png")
summary_txt_file = Path(__file__).with_name("stress_strain_summary.txt")
summary_csv_file = Path(__file__).with_name("stress_strain_summary.csv")

# =========================
# Robust read
# =========================
raw = file_path.read_bytes()
raw = raw.replace(b"\x00", b"")
text = raw.decode("utf-8", errors="ignore")
text = text.replace("−", "-").replace("–", "-").replace("—", "-")

float_re = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")

rows = []
for line in text.splitlines():
    s = line.strip()
    if not s:
        continue
    if s.startswith(("#", "@")):
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

    rows.append((eps, sig))

if len(rows) < 2:
    sample = [ln for ln in text.splitlines() if ln.strip()][:10]
    print("Could not parse >=2 numeric rows.")
    print("First decoded non-empty lines (preview):")
    for ln in sample:
        print(repr(ln[:160]))
    raise SystemExit(
        "Fix: your strain/stress may not be the 1st/2nd numeric columns. "
        "Try strain_index=1, stress_index=2 (if timestep is first)."
    )

arr = np.array(rows, dtype=float)
strain = arr[:, 0]
stress_npm = arr[:, 1]

order = np.argsort(strain)
strain = strain[order]
stress_npm = stress_npm[order]

stress_gpa = stress_npm / thickness_nm

# =========================
# UTS
# =========================
peak_idx = int(np.argmax(stress_gpa))
uts_gpa = float(stress_gpa[peak_idx])
uts_npm = float(stress_npm[peak_idx])
strain_at_uts = float(strain[peak_idx])

# =========================
# Fracture detection for toughness
# =========================
fracture_idx = None
fracture_strain = None
fracture_detection_method = None

for i in range(peak_idx + 1, len(stress_gpa)):
    s1 = stress_gpa[i - 1]
    s2 = stress_gpa[i]
    if s1 > 0.0 and s2 <= 0.0:
        e1 = strain[i - 1]
        e2 = strain[i]
        fracture_strain = float(e1 + (0.0 - s1) * (e2 - e1) / (s2 - s1))
        fracture_idx = i
        fracture_detection_method = "zero-crossing after peak"
        break

if fracture_strain is None:
    threshold = near_zero_fraction_of_UTS * uts_gpa
    for i in range(peak_idx + 1, len(stress_gpa) - hold_points + 1):
        window = stress_gpa[i:i + hold_points]
        if np.all(window <= threshold):
            fracture_strain = float(strain[i])
            fracture_idx = i
            fracture_detection_method = (
                f"first sustained region below {near_zero_fraction_of_UTS:.1%} of UTS"
            )
            break

if fracture_strain is None:
    fracture_idx = len(strain) - 1
    fracture_strain = float(strain[fracture_idx])
    fracture_detection_method = "end of file (no zero/near-zero post-peak region found)"

if fracture_detection_method == "zero-crossing after peak":
    strain_tough = np.concatenate([strain[:fracture_idx], [fracture_strain]])
    stress_tough = np.concatenate([stress_gpa[:fracture_idx], [0.0]])
else:
    strain_tough = strain[:fracture_idx + 1]
    stress_tough = stress_gpa[:fracture_idx + 1]

toughness_gpa = float(np.trapezoid(stress_tough, strain_tough))
toughness_J_m3 = toughness_gpa * 1e9

# =========================
# Yield detection for resilience
# =========================
fit_mask = (strain >= 0.0) & (strain <= elastic_fit_max_strain)
if np.count_nonzero(fit_mask) < 3:
    raise SystemExit(
        f"Not enough data points in the elastic-fit window 0 <= strain <= {elastic_fit_max_strain}."
    )

x_fit = strain[fit_mask]
y_fit = stress_gpa[fit_mask]

if force_elastic_fit_through_origin:
    elastic_slope = float(np.dot(x_fit, y_fit) / np.dot(x_fit, x_fit))
    elastic_intercept = 0.0
else:
    elastic_slope, elastic_intercept = np.polyfit(x_fit, y_fit, 1)
    elastic_slope = float(elastic_slope)
    elastic_intercept = float(elastic_intercept)

offset_line = elastic_slope * (strain - offset_strain) + elastic_intercept
diff = stress_gpa - offset_line

yield_strain = None
yield_stress = None
yield_idx = None
yield_detection_method = None

start_idx_for_yield = int(np.searchsorted(strain, offset_strain))

for i in range(max(start_idx_for_yield + 1, 1), len(diff)):
    d1 = diff[i - 1]
    d2 = diff[i]
    if d1 >= 0.0 and d2 <= 0.0:
        e1 = strain[i - 1]
        e2 = strain[i]
        yield_strain = float(e1 + (0.0 - d1) * (e2 - e1) / (d2 - d1))
        yield_stress = float(np.interp(yield_strain, strain, stress_gpa))
        yield_idx = i
        yield_detection_method = "0.2% offset yield"
        break

if yield_strain is None:
    elastic_line = elastic_slope * strain + elastic_intercept
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_deviation = np.abs(stress_gpa - elastic_line) / np.maximum(np.abs(elastic_line), 1e-12)

    for i in range(max(start_idx_for_yield, 1), len(relative_deviation)):
        if relative_deviation[i] >= yield_deviation_fraction:
            yield_strain = float(strain[i])
            yield_stress = float(stress_gpa[i])
            yield_idx = i
            yield_detection_method = f"first deviation >= {yield_deviation_fraction:.1%} from fitted elastic line"
            break

if yield_strain is None:
    yield_strain = float(elastic_fit_max_strain)
    yield_stress = float(np.interp(yield_strain, strain, stress_gpa))
    yield_idx = int(np.searchsorted(strain, yield_strain))
    yield_detection_method = "end of elastic-fit window fallback"

strain_res = np.concatenate([strain[:yield_idx], [yield_strain]])
stress_res = np.concatenate([stress_gpa[:yield_idx], [yield_stress]])

resilience_gpa = float(np.trapezoid(stress_res, strain_res))
resilience_J_m3 = resilience_gpa * 1e9

# =========================
# Write summary files
# =========================
summary_lines = [
    f"Input file: {file_path}",
    f"Parsed rows: {len(strain)}",
    "",
    f"Thickness used for conversion (nm): {thickness_nm:.6f}",
    f"Stress conversion: N/m -> GPa by dividing by thickness_nm",
    "",
    f"Strain at UTS: {strain_at_uts:.6f}",
    f"UTS (N/m): {uts_npm:.6f}",
    f"UTS (GPa): {uts_gpa:.6f}",
    "",
    f"Detected fracture strain: {fracture_strain:.6f}",
    f"Fracture detection method: {fracture_detection_method}",
    f"Full modulus of toughness (GPa): {toughness_gpa:.6f}",
    f"Full modulus of toughness (J/m^3): {toughness_J_m3:.6e}",
    "",
    f"Elastic fit slope / effective Young's modulus (GPa): {elastic_slope:.6f}",
    f"Elastic fit intercept (GPa): {elastic_intercept:.6f}",
    f"Detected yield strain: {yield_strain:.6f}",
    f"Yield stress (GPa): {yield_stress:.6f}",
    f"Yield detection method: {yield_detection_method}",
    f"Modulus of resilience (GPa): {resilience_gpa:.6f}",
    f"Modulus of resilience (J/m^3): {resilience_J_m3:.6e}",
    "",
    f"Toughness plot: {toughness_plot_file.name}",
    f"Resilience plot: {resilience_plot_file.name}",
]
summary_txt_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

csv_rows = [
    ["metric", "value", "units", "notes"],
    ["parsed_rows", len(strain), "count", ""],
    ["thickness_for_conversion", f"{thickness_nm:.6f}", "nm", "used to convert N/m to GPa"],
    ["strain_at_uts", f"{strain_at_uts:.6f}", "strain", ""],
    ["uts", f"{uts_npm:.6f}", "N/m", ""],
    ["uts", f"{uts_gpa:.6f}", "GPa", ""],
    ["fracture_strain", f"{fracture_strain:.6f}", "strain", fracture_detection_method],
    ["modulus_of_toughness", f"{toughness_gpa:.6f}", "GPa", "area under stress-strain curve up to fracture"],
    ["modulus_of_toughness", f"{toughness_J_m3:.6e}", "J/m^3", "area under stress-strain curve up to fracture"],
    ["elastic_slope_effective_youngs_modulus", f"{elastic_slope:.6f}", "GPa", f"fit up to strain={elastic_fit_max_strain:.6f}"],
    ["elastic_intercept", f"{elastic_intercept:.6f}", "GPa", ""],
    ["yield_strain", f"{yield_strain:.6f}", "strain", yield_detection_method],
    ["yield_stress", f"{yield_stress:.6f}", "GPa", yield_detection_method],
    ["modulus_of_resilience", f"{resilience_gpa:.6f}", "GPa", "area under stress-strain curve up to yield"],
    ["modulus_of_resilience", f"{resilience_J_m3:.6e}", "J/m^3", "area under stress-strain curve up to yield"],
]
with summary_csv_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

# =========================
# Plot 1: toughness
# =========================
plt.figure(figsize=(8, 6))
plt.plot(strain, stress_gpa, lw=1.8, label="Stress-strain curve")
plt.fill_between(strain_tough, stress_tough, 0.0, alpha=0.30, label="Area = modulus of toughness")
plt.scatter([strain_at_uts], [uts_gpa], s=50, zorder=3, label="UTS")
plt.scatter([fracture_strain], [0.0], s=50, zorder=3, label="Fracture point")
plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress-strain curve with modulus of toughness area")
plt.legend()
plt.tight_layout()
plt.savefig(toughness_plot_file, dpi=300)
plt.close()

# =========================
# Plot 2: resilience
# =========================
plt.figure(figsize=(8, 6))
plt.plot(strain, stress_gpa, lw=1.8, label="Stress-strain curve")
plt.fill_between(strain_res, stress_res, 0.0, alpha=0.30, label="Area = modulus of resilience")

xmax_plot = max(yield_strain * 1.25, elastic_fit_max_strain * 1.5, 0.02)
elastic_line_plot = elastic_slope * strain + elastic_intercept
offset_line_plot = elastic_slope * (strain - offset_strain) + elastic_intercept
mask_small = strain <= xmax_plot

plt.plot(strain[mask_small], elastic_line_plot[mask_small], "--", lw=1.4, label="Elastic fit")
plt.plot(strain[mask_small], offset_line_plot[mask_small], "--", lw=1.4, label="0.2% offset line")
plt.scatter([yield_strain], [yield_stress], s=50, zorder=3, label="Yield point")

plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress-strain curve with modulus of resilience area")
plt.legend()
plt.tight_layout()
plt.savefig(resilience_plot_file, dpi=300)
plt.close()

# =========================
# Console output
# =========================
print("Reading:", file_path)
print("Parsed rows:", len(strain))
print()
print(f"Strain at UTS = {strain_at_uts:.6f}")
print(f"UTS = {uts_npm:.6f} N/m = {uts_gpa:.6f} GPa")
print()
print(f"Detected fracture strain = {fracture_strain:.6f}")
print(f"Fracture detection method = {fracture_detection_method}")
print(f"Full modulus of toughness = {toughness_gpa:.6f} GPa")
print(f"Full modulus of toughness (SI) = {toughness_J_m3:.6e} J/m^3")
print()
print(f"Elastic fit slope (effective Young's modulus) = {elastic_slope:.6f} GPa")
print(f"Elastic fit intercept = {elastic_intercept:.6f} GPa")
print(f"Detected yield strain = {yield_strain:.6f}")
print(f"Yield stress = {yield_stress:.6f} GPa")
print(f"Yield detection method = {yield_detection_method}")
print(f"Modulus of resilience = {resilience_gpa:.6f} GPa")
print(f"Modulus of resilience (SI) = {resilience_J_m3:.6e} J/m^3")
print()
print("Saved:", toughness_plot_file)
print("Saved:", resilience_plot_file)
print("Saved:", summary_txt_file)
print("Saved:", summary_csv_file)
