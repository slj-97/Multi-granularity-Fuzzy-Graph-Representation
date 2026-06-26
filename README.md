# Multi-granularity-Fuzzy-Graph-Representation

Experiment code for **Multi-Granularity Fuzzy Graph Representation (MG-FGR)**.

MG-FGR constructs a fuzzy similarity graph, computes its max-min transitive closure, extracts multi-granularity lambda-cut partitions, and fuses the corresponding single-granularity indicator vectors into a traceable representation. The experiments reproduce the real-world disease-subtyping study and the manuscript figures/tables.

## Repository Contents

- `mgfgr.py`: core MG-FGR implementation, including affinity construction, max-min closure, lambda selection, multi-granularity fusion, and clustering metrics.
- `run_section5_2.py`: main experiment script for the Dermatology case study, baseline comparison, single-granularity ablation, interpretability analysis, robustness analysis, and sensitivity to the neighbor count `k`.
- `redraw_mgfgr_figures.py`: script for regenerating the illustrative MG-FGR figures used in the manuscript.
- `data/`: cached public UCI datasets used by the loaders. The main manuscript experiment uses `dermatology.data`.
- `figures/`: generated figures.
- `tables/`: generated LaTeX tables.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code was developed with Python 3.10+ and uses NumPy, SciPy, scikit-learn, pandas, and Matplotlib.

## Reproducing the Main Experiments

Run:

```bash
python run_section5_2.py
```

The script will:

1. load the Dermatology dataset,
2. construct the locally scaled mutual-kNN fuzzy graph,
3. compute the max-min transitive closure,
4. select lambda-cut granularity levels,
5. build the MG-FGR representation,
6. compare against classical clustering baselines,
7. run single-granularity and weighting ablations,
8. generate interpretability, robustness, and `k`-sensitivity figures.

Outputs are written to:

- `figures/`
- `tables/`

## Reproducing the Illustrative Figures

Run:

```bash
python redraw_mgfgr_figures.py
```

This regenerates the small hand-computable illustrative figures for the MG-FGR construction.

## Notes on Data

The main case study uses the Dermatology dataset from the UCI Machine Learning Repository. The script first looks for a cached local copy in `data/`; if it is not present, it attempts to download the dataset and cache it locally.

## Citation

If you use this code, please cite the associated manuscript:

```text
Multi-granularity Fuzzy Graph Representation.
```

The final bibliographic information can be added after publication.
