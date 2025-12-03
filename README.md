# AI for SAM

Efficient design of perovskite SAM molecules based on quantum chemical calculations and artificial intelligence.


- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Install Dependencies](#install-dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)


## Overview

AI-SAM is an efficient computational workflow designed for the rational design and screening of perovskite self-assembled monolayer (SAM) molecules. This integrated framework combines artificial intelligence, high-throughput computational chemistry, and retrosynthetic analysis to accelerate the discovery of high-performance SAM materials for perovskite solar cells.

The workflow encompasses five key components: (1) Functional Group Extraction from carbazole-based systems, where representative molecular fragments and functional groups are systematically identified and catalogued; (2) Fragment-based Molecular Generation, which employs combinatorial assembly strategies to construct diverse SAM candidate libraries by connecting extracted functional groups; (3) High-throughput Density Functional Theory (DFT) Calculations, enabling rapid evaluation of electronic properties, energy levels, and molecular geometries across thousands of candidate structures; (4) Retrosynthetic Analysis, which assesses the synthetic accessibility and feasibility of promising candidates through computational retrosynthesis; and (5) Molecular Screening and Ranking, where multi-criteria optimization algorithms prioritize candidates based on their predicted performance metrics, synthetic accessibility, and compatibility with perovskite interfaces.

This systematic approach significantly reduces the time and cost associated with experimental trial-and-error, providing researchers with a curated list of synthetically accessible, high-performance SAM candidates for perovskite device applications.

## System Requirements

### Hardware requirements

AI-SAM package requires only a standard computer with enough RAM to support the in-memory operations.

### Software Requirements

#### OS Requirements

This package is supported for Windows and Linux. The package has been tested on the following systems:

Windows: Windows 11 25H2
Linux: Ubuntu 22.04.1

#### Python Dependencies

```text
# Python = 3.10
rdkit
matplotlib
seaborn
pandas
numpy
cuspy
scipy
```

## Installation Guide

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit the config file: config.json

```json
{
  "sam_data_config": {
    "root": "{DIRECTORY PATH OF THE DATA}",
    "calc_columns": ["HOMO", "LUMO", "HOMO-LUMO Gap", "IE", "Heterosatom", "Aromaticity", "Synthesizability", "$\\lambda_{h}$"],
    "calc_results_fn": "calc_results.csv"
  }
}
```

## Usage

### 1. Functional Groups Extraction

Extract molecules with a carbazole substructure from the PubChem database and identify the groups attached to the carbazole.

```bash
python ai_sam/funtional_groups_extraction_process.py
```

### 2. Molecule Generation

Attach the selected groups to the carbazole, one to two groups at a time, to generate candidate molecules.

```bash
python ai_sam/mol_gen_process.py
```

### 3. High-throughput Theoretical Calculation

Read the generated candidate molecules, parse their molecular structures from SMILES strings, optimize their conformations using the UFFO force field, and then generate Gaussian input files.(The Gaussian calculations need to be performed by the user)

```bash
python ai_sam/dft_process.py
```

### 4. Molecule Screening and Ranking

Rank the candidate molecules based on theoretical calculation metrics such as HOMO, LUMO, etc.

```bash
python ai_sam/sort_process.py
```

### 5. Data Analysis

Data analysis, including correlation analysis and more.

```bash
python ai_sam/analysis_process.py
```

## License
This project is covered under the Apache 2.0 License.
