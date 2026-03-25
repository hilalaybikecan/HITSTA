# HITSTA Interactive Analysis

A Streamlit web app for analyzing HITSTA (High-Throughput In-Situ Transflectance and photoluminescence Apparatus) optical measurement data.

## Features

- **Reflectance analysis**: spectra viewer (single/multi cell), band-edge slope, short-wavelength step, self-similarity tracking over time
- **PL analysis**: spectra with Gaussian fitting, peak wavelength shift tracking, PL intensity over time, self-similarity
- **Condition mapping**: upload a runsheet (Excel) to group cells by experimental conditions and generate box plots

## How to use

1. Open the app (see link below or run locally)
2. Upload your HITSTA `.txt` data file(s) in the sidebar
3. Optionally upload a runsheet (Excel) mapping cell IDs to experimental conditions
4. Browse Reflectance / PL / Conditions tabs

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live app

[Open HITSTA Analysis](https://hilalaybikecan-hitsta.streamlit.app)
