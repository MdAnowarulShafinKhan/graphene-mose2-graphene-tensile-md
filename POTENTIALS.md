# Interatomic potential files

The third-party potential parameter files are **not redistributed in this repository**.
Before running the representative LAMMPS inputs, place the required potential files in the same directory as the corresponding input script (or edit the file paths in the input script).

## Required files

### Graphene
- `CH.airebo`
- AIREBO potential: Stuart, Tutein & Harrison, *J. Chem. Phys.* 112, 6472-6486 (2000).
- DOI: https://doi.org/10.1063/1.481208

### MoSe2 - Stillinger-Weber cases
- `h-mose2.sw`
- Parameterization source used in the project: Jiang & Zhou, *Handbook of Stillinger-Weber Potential Parameters for Two-Dimensional Atomic Crystals* (2017).
- DOI: https://doi.org/10.5772/intechopen.71929

### MoSe2 - Tersoff cases
- `MoSe.tersoff`
- Parameterization source used in the project: Zhang et al., *npj Computational Materials* 7, 113 (2021).
- DOI: https://doi.org/10.1038/s41524-021-00573-x

See the manuscript/project documentation for the exact mapping and interlayer Lennard-Jones parameters.
