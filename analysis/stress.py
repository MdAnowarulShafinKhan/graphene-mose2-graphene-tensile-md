import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Inputs
# ----------------------------
dat_path = "strain.dat"                 # your .dat file
nm_to_gpa_factor = 1.3196              # divide stress (N/m) by this to get stress (GPa)
out_props_path = "mechanical_properties.dat"  # output .dat file for UTS, fracture strain, Young's modulus

# ----------------------------
# Load data (1st col = strain, 2nd col = stress in N/m). Ignore col 3 & 4.
# ----------------------------
df = pd.read_csv(dat_path, sep=r"\s+", header=None, usecols=[0, 1], names=["strain", "stress_Nm"])

# Convert to GPa
df["stress_GPa"] = df["stress_Nm"] / nm_to_gpa_factor

# ----------------------------
# Ultimate tensile stress (UTS) and fracture strain (peak strain)
# Fracture strain is defined as strain at maximum stress.
# ----------------------------
i_uts = df["stress_GPa"].idxmax()
uts_gpa = float(df.loc[i_uts, "stress_GPa"])
fracture_strain = float(df.loc[i_uts, "strain"])   # per your definition
fracture_stress = uts_gpa

# ----------------------------
# Young's modulus from initial linear region
# Approach: scan candidate end-strains, fit line, compute R^2,
# choose the longest window with R^2 >= 0.995; else best R^2.
# ----------------------------
scan = df[(df["strain"] >= 0) & (df["strain"] <= 0.05)].copy()

candidate_ends = np.arange(0.002, 0.030, 0.001)  # adjust if your linear region extends further
fits = []

for smax in candidate_ends:
    sub = scan[scan["strain"] <= smax]
    if len(sub) < 50:
        continue

    x = sub["strain"].to_numpy()
    y = sub["stress_GPa"].to_numpy()

    # Least-squares line fit: y = E*x + b
    E, b = np.polyfit(x, y, 1)
    yhat = E * x + b

    # R^2
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fits.append((smax, E, b, r2, len(sub)))

if not fits:
    raise RuntimeError("Not enough points in the initial strain range to estimate Young's modulus. "
                       "Consider increasing the scan range or lowering the minimum point requirement.")

good = [t for t in fits if (t[3] >= 0.995 and t[1] > 0)]
if good:
    smax, young_E, intercept_b, r2, npts = max(good, key=lambda t: t[0])  # longest good window
else:
    smax, young_E, intercept_b, r2, npts = max(fits, key=lambda t: t[3])  # best R^2

# ----------------------------
# Print results
# ----------------------------
print(f"Ultimate stress (UTS): {uts_gpa:.6f} GPa")
print(f"Fracture strain (peak strain): {fracture_strain:.6f}")
print(f"Young's modulus: {young_E:.6f} GPa (fit over strain 0 → {smax:.3f}, R^2 = {r2:.6f}, n={npts})")

# ----------------------------
# Save results to a .dat file  (ADDED)
# ----------------------------
with open(out_props_path, "w") as f:
    f.write("# Mechanical properties extracted from stress–strain data\n")
    f.write(f"# Input file: {dat_path}\n")
    f.write("# Columns: Property  Value  Unit\n")
    f.write(f"Ultimate_Stress  {uts_gpa:.10f}  GPa\n")
    f.write(f"Fracture_Strain  {fracture_strain:.10f}  -\n")
    f.write(f"Youngs_Modulus   {young_E:.10f}  GPa\n")
    f.write(f"# Young's modulus fit details: strain_range=0_to_{smax:.3f}, R2={r2:.8f}, npts={npts}\n")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(df["strain"], df["stress_GPa"], linewidth=1.2)
plt.xlabel("Strain")
plt.ylabel("Stress (GPa)")
plt.title("Stress–strain curve (converted to GPa)")

# Mark peak point
plt.scatter([fracture_strain], [fracture_stress], zorder=5)

# Plot linear-fit line over selected region
xfit = np.linspace(0, smax, 200)
yfit = young_E * xfit + intercept_b
plt.plot(xfit, yfit, linewidth=1.2)

plt.tight_layout()
plt.savefig("stress_strain.png", dpi=300)
plt.show()
