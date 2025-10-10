# Stresstesting road networks using topological metrics — Code & Data

This repository contains the code, processed datasets, and scripts used to reproduce the experiments and figures in the paper **"Topology shapes road network recovery: Global evidence from 224 cities"** by Srijith Balakrishnan and Patrick Stokkink. The paper and supplemental material are included in the repository.

The study uses a large-scale simulation-based stress-testing framework to evaluate how urban road networks recover after disruptions. For each of 224 cities/towns (Figure 1), road networks extracted from OpenStreetMap are repeatedly damaged by random edge removals at varying intensities, and multiple recovery strategies (e.g., based on betweenness, closeness, or degree centrality) are applied to restore connectivity. Network performance—measured through efficiency, accessibility, equity, and connectivity—is tracked throughout recovery to quantify resilience and identify structural trade-offs across cities.
<img src="results/figures/road_networks_map.png" alt="224 cities" width="700" />
*Figure 1. Locations of the 224 cities included in the study.*

---

## Repository structure

```
/data/road_networks      # Openstreet road network datasets
/src/                    # Simulation scripts
/results/                # Output figures, aggregated model results
requirements.txt         # Python dependencies
README.md                # This file
LICENSE                  # MIT License
```
---

## Citation

If you use this repository, please cite:

> Balakrishnan, S. & Stokkink, P. *Topology shapes road network recovery: Global evidence from 224 cities* (2025).

---

## License

The code is released under the **MIT License**.  
Road network data derived from **OpenStreetMap** are covered by the **ODbL license**.

---

## Contact

- Srijith Balakrishnan — s.balakrishnan@tudelft.nl  
- Patrick Stokkink — p.s.a.stokkink@tudelft.nl  

---

_Last updated: 2025-10-10_
