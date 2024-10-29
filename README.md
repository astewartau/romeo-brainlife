[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.444-blue.svg)](https://doi.org/10.25663/bl.app.444)

# ROMEO Phase Unwrapping - Brainlife App

## Overview
This Brainlife App performs phase unwrapping and various related processing steps on multi-echo MRI data using the ROMEO algorithm from the [`MriResearchTools`](https://github.com/korbinian90/MriResearchTools.jl) Julia library. The main function of the app is to take in magnitude and phase images (NIfTI format) and output unwrapped phase images, along with several additional outputs, such as quality maps, T2*/R2* maps, and B0 field maps.

## How it works
The app processes multi-echo NIfTI phase and magnitude images through the following steps:

1. **Phase Offset Removal**: Handles offset removal using mcpc3ds for better downstream processing.
2. **Phase Unwrapping**: Uses ROMEO to unwrap phase images.
3. **Quality Map Generation**: Creates a voxel-wise quality map for the unwrapped phase.
4. **Mask Generation**: Produces a mask based on phase and magnitude data for further analysis.
5. **T2\* and R2\* Mapping**: Calculates T2* and R2* maps based on the unwrapped phase and input echoes.
6. **B0 Field Map Generation**: Computes a B0 map for the unwrapped phase to aid in further analysis or use in QSM.

Each processing step is facilitated by [`MriResearchTools`](https://github.com/korbinian90/MriResearchTools.jl), a Julia library for MRI image processing, phase unwrapping, and data correction methods.

### Authors
- Ashley Stewart (Brainlife App)
- Korbinian Eckstein ([`MriResearchTools`](https://github.com/korbinian90/MriResearchTools.jl))

#### Copyright (c) 2024 brainlife.io The University of Texas at Austin

### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications. Copy and past the following lines into your repository when using this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations
We ask that you the following articles when publishing papers that used data, code or other resources created by the brainlife.io community.

1. **ROMEO**: Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K., Robinson, S.D., 2020. Phase Unwrapping with a Rapid Opensource Minimum Spanning TreE AlgOrithm (ROMEO). Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28563
MCPC-3D-S

2. **ASPIRE**: Eckstein, K., Dymerska, B., Bachrata, B., Bogner, W., Poljanc, K., Trattnig, S., Robinson, S.D., 2018. Computationally Efficient Combination of Multi-channel Phase Data From Multi-echo Acquisitions (ASPIRE). Magnetic Resonance in Medicine 79, 2996–3006. https://doi.org/10.1002/mrm.26963
Homogeneity Correction

3. **NumART2\* - fast T2\* and R2\* fitting**: Hagberg, G.E., Indovina, I., Sanes, J.N., Posse, S., 2002. Real-time quantification of T2* changes using multiecho planar imaging and numerical methods. Magnetic Resonance in Medicine 48(5), 877-882. https://doi.org/10.1002/mrm.10283

4. **Phase-based-masking**: Hagberg, G.E., Eckstein, K., Tuzzi, E., Zhou, J., Robinson, S.D., Scheffler, K., 2022. Phase-based masking for quantitative susceptibility mapping of the human brain at 9.4T. Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.29368

5. **Inhomogeneity correction**: 
  - Eckstein, K., Trattnig, S., Robinson, S.D., 2019. A Simple Homogeneity Correction for Neuroimaging at 7T, in: Proceedings of the 27th Annual Meeting ISMRM. Presented at the ISMRM, Montréal, Québec, Canada. https://index.mirasmart.com/ISMRM2019/PDFfiles/2716.html Eckstein, K., Bachrata, B., Hangel, G., Widhalm, G., Enzinger, C., Barth, M., Trattnig, S., Robinson, S.D., 2021. 
  - Improved susceptibility weighted imaging at ultra-high field using bipolar multi-echo acquisition and optimized image processing: CLEAR-SWI. NeuroImage 237, 118175. https://doi.org/10.1016/j.neuroimage.2021.118175

## Running the App 

### On Brainlife.io

You can submit this App online at [https://doi.org/10.25663/bl.app.444](https://doi.org/10.25663/bl.app.444) via the "Execute" tab.

### Running Locally (on your machine)

1. `git clone` this repo.
2. Inside the cloned directory, create `config.json` with something like the following content with paths to your input files.

```json
{
  "magnitude": [
    "inputs/sub-1_run-1_echo-1_part-mag_MEGRE.nii",
    "inputs/sub-1_run-1_echo-2_part-mag_MEGRE.nii",
    "inputs/sub-1_run-1_echo-3_part-mag_MEGRE.nii",
    "inputs/sub-1_run-1_echo-4_part-mag_MEGRE.nii"
  ],
  "phase": [
    "inputs/sub-1_run-1_echo-1_part-phase_MEGRE.nii",
    "inputs/sub-1_run-1_echo-2_part-phase_MEGRE.nii",
    "inputs/sub-1_run-1_echo-3_part-phase_MEGRE.nii",
    "inputs/sub-1_run-1_echo-4_part-phase_MEGRE.nii"
  ]
}
```

3. Launch the App by executing `main`

```bash
./main
```

### Sample Datasets

If you don't have your own input file, you can download sample datasets from Brainlife.io, or you can use [Brainlife CLI](https://github.com/brain-life/cli).

```
npm install -g brainlife
bl login
mkdir input
bl dataset download 5a0f0fad2c214c9ba8624376#5a050966eec2b300611abff2 && mv 5a0f0fad2c214c9ba8624376#5a050966eec2b300611abff2 .
```

## Output

All output files will be generated inside the current working directory (pwd), inside a specifc directory called `outputFolder`.

### Dependencies

This App requires `wget` and `tar` to run.

