import numpy as np
import re
from pathlib import Path

# -------- user inputs --------
thickness_nm = 1.3196

# strain.dat must be in the SAME folder as this script
file_path = Path(__file__).with_name("strain.dat")

# If your file has an extra numeric column first (e.g., timestep),
# set strain_index=1 and stress_index=2.
strain_index = 0
stress_index = 1

# -------- robust read: bytes -> clean -> text --------
raw = file_path.read_bytes()

# remove NUL bytes (common if file is UTF-16 or has binary junk)
raw = raw.replace(b"\x00", b"")

# decode safely (drops problematic bytes like 0x9D)
text = raw.decode("utf-8", errors="ignore")

# replace unicode minus dashes with normal "-"
text = text.replace("−", "-").replace("–", "-").replace("—", "-")

# float finder (supports E/e and also Fortran D/d exponents)
float_re = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")

rows = []
for line in text.splitlines():
    s = line.strip()
    if not s:
        continue
    if s.startswith(("#", "@")):  # common comment/header markers
        continue

    nums = float_re.findall(s)
    if len(nums) <= max(strain_index, stress_index):
        continue

    # convert Fortran D exponent to E
    a = nums[strain_index].replace("D", "E").replace("d", "e")
    b = nums[stress_index].replace("D", "E").replace("d", "e")

    try:
        eps = float(a)
        sig = float(b)
    except ValueError:
        continue

    rows.append((eps, sig))

if len(rows) < 2:
    # Print a few decoded lines for diagnosis
    sample = [ln for ln in text.splitlines() if ln.strip()][:10]
    print("Could not parse >=2 numeric rows.")
    print("First decoded non-empty lines (preview):")
    for ln in sample:
        print(repr(ln[:160]))
    raise SystemExit(
        "Fix: your strain/stress may not be the 1st/2nd numeric columns. "
        "Try strain_index=1, stress_index=2 (if timestep is first)."
    )

arr = np.array(rows, dtype=float)  # guaranteed Nx2 now
strain = arr[:, 0]
stress_npm = arr[:, 1]

# sort by strain (safe)
order = np.argsort(strain)
strain = strain[order]
stress_npm = stress_npm[order]

# convert N/m -> GPa (your requested conversion)
stress_gpa = stress_npm / thickness_nm

# fracture point = peak stress (UTS)
peak_idx = int(np.argmax(stress_npm))

# toughness = area under stress-strain curve up to fracture point
toughness_gpa = float(np.trapz(stress_gpa[:peak_idx + 1], strain[:peak_idx + 1]))
toughness_J_m3 = toughness_gpa * 1e9  # 1 GPa = 1e9 J/m^3

print("Reading:", file_path)
print("Parsed rows:", len(strain))
print(f"Fracture strain (at peak) = {strain[peak_idx]:.6f}")
print(f"UTS = {stress_npm[peak_idx]:.6f} N/m = {stress_gpa[peak_idx]:.6f} GPa")
print(f"Toughness (paper-style numeric) = {toughness_gpa:.6f}")
print(f"Toughness (SI) = {toughness_J_m3:.6e} J/m^3")