# graphene-mose2-graphene-tensile-md
# Layer-Resolved Tensile Failure of Graphene/MoSe₂/Graphene Trilayers

Molecular dynamics investigation of the mechanical response and fracture
behavior of graphene/MoSe₂/graphene (GMG) van der Waals heterostructures.

## Overview

This project studies how graphene and MoSe₂ layers share mechanical load
and fail under uniaxial tensile deformation.

The simulations investigate:

- Armchair and zigzag tensile loading
- Temperatures from 100 to 600 K
- Young's modulus
- Ultimate tensile stress
- Fracture strain
- Modulus of resilience
- Modulus of toughness
- Layer-resolved fracture
- Temperature-dependent fracture sequence
- Sensitivity to different MoSe₂ interatomic potentials

## Simulation Model

- System: Graphene/MoSe₂/Graphene trilayer
- Total atoms: 43,264
- Approximate dimensions: 18.43 × 21.28 nm
- Maximum lattice mismatch: ~0.11%
- Boundary conditions: periodic in-plane, free out-of-plane

## Molecular Dynamics Method

**MD code:** LAMMPS  
**Visualization:** OVITO  
**Structure construction:** Atomsk  

### Interatomic interactions

- Graphene: AIREBO
- MoSe₂: Stillinger-Weber (SW) and Tersoff
- Graphene–MoSe₂ interface: 12-6 Lennard-Jones interaction

Two independent MoSe₂ potentials were used to test whether the observed
mechanical and fracture behavior depends strongly on the chosen force field.

### Tensile simulation

- Energy minimization
- 50 ps NVE equilibration
- 100 ps NPT equilibration
- Time step: 0.5 fs
- Engineering strain rate: 1 × 10⁸ s⁻¹
- Temperature range: 100–600 K
- Loading directions: armchair and zigzag

Stress was calculated using the virial stress formulation and used to
construct stress–strain curves.

## Model Validation

The simulation methodology was checked at multiple levels:

1. Mechanical properties of monolayer graphene were compared with literature.
2. Mechanical properties of monolayer MoSe₂ were compared with literature.
3. A graphene/WS₂/graphene trilayer was reconstructed and compared with a
   published benchmark study.
4. Rule-of-mixtures predictions were compared with the MD results.

The benchmark trilayer calculations showed deviations below approximately
4.1% for the investigated mechanical properties.

## Main Findings

### Graphene strongly reinforces MoSe₂

At 300 K, graphene encapsulation increases the Young's modulus of MoSe₂ by
approximately 3.7–3.8× and the ultimate tensile stress by approximately
3.1–3.4×.

### Mechanical response depends on loading direction

**Armchair loading**
- Higher stiffness
- Higher recoverable elastic energy

**Zigzag loading**
- Higher ultimate tensile strength
- Higher fracture strain
- Higher toughness

### Failure occurs layer by layer

The three layers do not fail simultaneously.

After one layer fractures, the remaining layers can continue carrying load,
producing step-like post-peak behavior in the stress–strain curves.

### Temperature changes which layer fails first

Under armchair loading:

**Graphene fails first from 100–600 K.**

Under zigzag loading:

- 100 K → graphene fails first
- 200–600 K → MoSe₂ fails first

This fracture-sequence crossover was reproduced using both SW and Tersoff
potentials.

The results therefore indicate that the tensile reliability of the trilayer
is controlled by **first-layer instability**, rather than simply by the
strength of the individual materials.

## Skills Demonstrated

This project demonstrates experience with:

- Molecular dynamics
- LAMMPS
- Uniaxial deformation using `fix deform`
- Stress–strain analysis
- Atomistic fracture analysis
- Virial stress calculation
- Mechanical property extraction
- Interatomic potential selection
- Force-field sensitivity analysis
- Model validation
- van der Waals heterostructures
- OVITO visualization
- Atomsk structure construction
- Scientific data analysis

## Limitations

The simulations represent idealized pristine atomistic systems.

Important limitations include:

- MD-accessible strain rates are much higher than experimental rates.
- The simulation cells are finite.
- Defects and polycrystalline structures are not considered.
- The adopted interlayer interaction is non-reactive.

Future work can extend the model to defects, grain boundaries, ripples,
different strain rates, and experimentally realistic structures.

## My Contribution

**Md. Anowarul Shafin Khan**

Contributions include:

- Conceptualization
- Methodology
- Investigation
- Formal analysis
- Validation
- Resources
- Writing – original draft

## Project Status

**Manuscript under review.**

This repository is being prepared as a research and reproducibility portfolio.
Selected simulation inputs, analysis workflows, representative results, and
visualizations will be added progressively.

## Authors

- Md. Anowarul Shafin Khan
- Md. Jobayer Aziz
- Md. Rezwanul Karim

## Data Availability

Research data are available on reasonable request.

## Citation

The manuscript is currently under review. Citation information will be updated
after publication.
