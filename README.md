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
