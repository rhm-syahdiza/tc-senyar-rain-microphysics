# tc-senyar-rain-microphysics
Python scripts for "Microphysical Properties of Tropical Cyclone Senyar Outer Rainbands"

## Repository Structure
This repository is organized by manuscript figure.

* Fig1.py: Generates (a) Topography (DEMNAS) + TC Senyar track (ibtracks), (b) Time series — pressure, wind speed, distance to Kototabang (ibtracks), (c) ERA5 Mean 850 hPa Wind + Acc VIMD, (d-f) Himawari-9 TBB + TC Senyar track (ibtracks), (g-i) BMKG Radar CMAX + TC Senyar track (ibtracks)
* Fig2.py: Fig. 2 (a) Mean fall speed vs. drop diameter from Parsivel disdrometer, (b) Number of drops per diameter class. Fig. S4 The sensitivity test for Atlas et al. (1973) fall velocity filter narrower threshold.
* Fig3.py: Equatorial Atmosphere Radar (EAR) time-height cross-sections (a) SNR, (b) vertical velocity and (c) Doppler spectral width.
* Fig4.py: Raindrop size distribution and instantaneous rainfall rate time-series.
* Fig5.py: Parsivel DSD Analysis: (a) Mean N(D) vs D for different rain rate categories (modified BMKG) (b) Rain rate occurrence percentage (c) Contribution to total drop count (Nt) and Rainfall (R) by Diameter Class.
* Fig6.py: Dm vs log10_Nw scatterplot to identify stratiform and convective regimes.
* Fig7.py: (a) shape-slope (mu-Lambda) compare to previous studies, (b) mu vs Rain Rate (Scatter + PDF), (c) Lambda vs Rain Rate (Scatter + PDF)
* Fig8.py: (Z-R) compare to previous studies
* FigS1.py: ERA5 (a) CAPE, (b) CIN, and (c) Vertical velocity (hPa s-1)
* FigS2.py: Zonal Hovmöller diagrams (a) Himawari-9 TBB, (b) BMKG Radar CMAX
* FigS3.py: MAWS Kototabang - 10-Minute Wind Speed Average
* FigS5.py: (a) EAR spectral width scatterplots versus the disdrometer rain rate (R), (b) versus EAR absolute value of the vertical air velocity.
* dsd_common.py Function to read and process disdrometer data
