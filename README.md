# Stress Testing Road Networks Using Topological Metrics: Code and Data

This repository contains the code, processed datasets, and scripts used to reproduce the experiments and figures in the paper **"Topology Shapes Road Network Recovery: Global Evidence from 224 Cities"** by *Srijith Balakrishnan* and *Patrick Stokkink*.

The study employs a large-scale, simulation-based stress-testing framework to evaluate how urban road networks recover after disruptions.  
For each of 224 cities/towns (Figure 1), road networks extracted from **OpenStreetMap** are repeatedly damaged by random edge removals at varying intensities.  
Multiple recovery strategies are then applied to restore connectivity, and multidimensional network performance is tracked throughout recovery to quantify resilience and identify structural trade-offs across cities.

<img src="results/figures/road_networks_map.png" alt="224 cities" width="700" />

*Figure 1. Locations of the 224 cities included in the study.*

---

## 1. Methodology

Road networks from **OpenStreetMap** are converted into graph models using Python’s **`igraph`** package, where intersections are represented as nodes and road segments as edges.  
Network disruptions are simulated by progressively removing random edges to emulate real-world disturbances such as accidents or hazards.

Recovery is modeled using five centrality-based strategies:
- **Degree**
- **Edge betweenness**
- **Nearest neighbour edge**
- **Eigenvector**
- **Closeness**  
*(plus a random baseline for comparison)*

These strategies enable a systematic evaluation of different prioritization approaches for restoring network connectivity and resilience, while tracking recovery across four competing dimensions:
**efficiency**, **accessibility**, **equity**, and **connectivity**.

<img src="results/figures/graph_methodology.png" alt="Methodology" width="550" />

*Figure 2. Methodology and workflow.*

---

## 2. Sample Simulation Outputs

<img src="results/figures/Sample_map.png" alt="Sample Map" width="350" />
<img src="results/figures/Sample_strategies_50pct.png" alt="Recovery Strategies" width="390" />

*Figure 3. Simulated recovery curves based on operational efficiency metric under different recovery strategies when the initial disruption is 50% of links (Network: Thenkara, Kerala, India).*

---

## 3. Repository Structure

```text
/data/road_networks       # OpenStreetMap road network datasets, including simulation results
/src/                     # Simulation scripts
/results/                 # Output figures and aggregated datasets for modeling
stress_testing.ipynb      # Jupyter notebook to run simulations and statistical models
requirements.txt          # Python dependencies
README.md                 # This file (auto-generated)
LICENSE                   # MIT License
```

---

## 4. Citation

If you use this repository, please cite:

> Srijith Balakrishnan, Patrick Stokkink.  
> *Topology Shapes Road Network Recovery: Global Evidence from 224 Cities.*  
> 20 October 2025, PREPRINT (Version 1) available at Research Square.  
> [https://doi.org/10.21203/rs.3.rs-7851098/v1](https://doi.org/10.21203/rs.3.rs-7851098/v1)

---

## 5. License

The code is released under the **MIT License**.  
Road network data derived from **OpenStreetMap** are covered by the **ODbL license**.

---

## 6. Contact

- **Srijith Balakrishnan** — [s.balakrishnan@tudelft.nl](mailto:s.balakrishnan@tudelft.nl)  
- **Patrick Stokkink** — [p.s.a.stokkink@tudelft.nl](mailto:p.s.a.stokkink@tudelft.nl)

---

_Last updated: 2025-10-22_
