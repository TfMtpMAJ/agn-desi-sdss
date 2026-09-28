# Accretion Properties and Population Structure of Active Galactic Nuclei: A Comparative Study of DESI and SDSS

**Ruici (Roselyn) Xi** · Hefei No.1 High School, Hefei, China · December 2025

## Abstract

We present a systematic comparative analysis of active galactic nuclei (AGN) observed by the Sloan Digital Sky Survey (SDSS) and the Dark Energy Spectroscopic Instrument (DESI). Using a uniform data-processing pipeline, we investigate the relations among continuum luminosity, broad emission-line luminosity, black hole mass, and Eddington ratio. By combining classical scaling relations with principal component analysis and clustering techniques, we identify physically meaningful sub-populations corresponding to distinct accretion states.

We find that DESI extends well-established SDSS trends into the low-luminosity and low-Eddington-ratio regime (log λ<sub>Edd</sub> ∼ −4). Crucially, this captures a population of AGNs near the critical theoretical threshold where the Broad Line Region (BLR) is expected to disappear. While selection effects contribute to the scatter, our results suggest that DESI provides a unique snapshot of AGNs in the 'fading' or 'transitioning' phase, potentially bridging the gap between canonical Type 1 quasars and quiescent galaxies. This hints that the observed differences are driven by distinct evolutionary stages of the accretion flow rather than measurement uncertainties alone.

## Full paper

**[Read the full paper (PDF)](Xi_Ruici_AGN_DESI_SDSS_2025.pdf)** · [Direct download](https://github.com/RoselynXi/agn-desi-sdss/raw/main/Xi_Ruici_AGN_DESI_SDSS_2025.pdf)

## Data sources

- **Sample**: 561 changing-look AGN at z ≤ 0.9 from Guo et al. (2025), ApJS, 278, 28 ([doi:10.3847/1538-4365/adc124](https://doi.org/10.3847/1538-4365/adc124)), built by cross-matching **DESI Data Release 1 (DR1)** with **SDSS Data Release 16 (DR16)**
- **DESI measurements**: `desi/desi-metadata-hdu1.mrt`, the machine-readable table published with Guo et al. (2025) (continuum and broad-line luminosities, black hole masses; 561 rows)
- **SDSS measurements**: `sdss/sdss-metadata-hdu2.mrt`, the corresponding SDSS table from the same paper (561 rows)

Raw spectra are not needed and are not included; all analysis starts from the published measurement tables.

---

## How to run

```
pip install -r requirements.txt
cd code
python desi.py
python sdss.py
```

Each script reads its table (`../desi/desi-metadata-hdu1.mrt` or `../sdss/sdss-metadata-hdu2.mrt`), computes Eddington ratios, fits the line–continuum relations, runs PCA + KMeans clustering, saves `cluster_features_with_labels.csv`, and writes the diagnostic figures.

## Repository structure

- `Xi_Ruici_AGN_DESI_SDSS_2025.pdf` – full paper (AASTeX 7.0.1)
- `code/` – analysis scripts
- `desi/`, `sdss/` – input tables and the clustering output (`cluster_features_with_labels.csv`: `logL5100`, `logLHb`, `zspec`, `log_lambda_Edd`, `cluster`)
- `figures/desi/`, `figures/sdss/` – diagnostic figures, numbered as in the paper (`fig01` … `fig12`)
- `figures/extra/` – figures generated but not used in the paper
