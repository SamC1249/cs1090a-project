# CS1090A Project: Dark Matter Halo Prediction Using Observable Galaxy Properties

**Team:** Brian Durst, Patrice Chonel, Susanna Wai Ha Fok, Sam Chen

## Overview

The goal of this project is to find which observable galaxy properties (and combinations of them) are most predictive of dark matter halo mass, and whether those combinations outperform stellar mass alone — a well-known predictor of halo mass. This can potentially provide insights into the relationship between dark matter and galaxy formation.

## Data

We use the [IllustrisTNG](https://www.tng-project.org/data/) hydrodynamical simulations (TNG100-1), which provide known dark-matter halo masses along with galaxy properties corresponding to measurable quantities: stellar mass, color, size, star-formation rate, metallicity, etc. The study is restricted to central galaxies (not satellites) to keep the problem tractable.

## Methods

- Baseline: linear regression on stellar mass (a well-known predictor of halo mass)
- Explore other models and combinations of predictor variables, compared against the baseline

**Stretch goals:** validate against real observational data (e.g. the SPARC database), and potentially experiment with a physics-informed neural network (PINN).

## Repo structure

- `docs/` — project proposals and write-ups
- `data/` — datasets (raw/processed); large files should not be committed
- `notebooks/` — analysis and modeling notebooks
- `src/` — data download and modeling scripts
- `results/` — model comparison tables and figures

## Setup

The processed dataset (`data/tng100-1_snap99_centrals.csv`, 17,464 central galaxies, ~4 MB) is committed to the repo, so **no API key is needed to run the models** — just set up the environment:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python requests h5py pandas numpy scikit-learn matplotlib
```

Only if you want to regenerate the dataset from scratch (e.g. different mass cut or extra columns): get a TNG API key (free registration at https://www.tng-project.org/users/register/), put it in a `.env` file at the repo root as `TNG_API_KEY=...` (the `.env` file is gitignored — never commit it), and run `src/download_data.py`.

## Pipeline

```bash
.venv/bin/python src/download_data.py   # (optional, needs API key) rebuild data/*.csv
.venv/bin/python src/train_models.py    # baseline + ridge/random forest/gradient boosting
.venv/bin/python src/make_figures.py    # plots in results/figures/
```

**Evaluation protocol:** one fixed 60/20/20 train/validation/test split (seed 109). Models are fit on train, hyperparameters and feature sets are chosen on validation, and the test set is used exactly once at the end for the baseline and the best model.
