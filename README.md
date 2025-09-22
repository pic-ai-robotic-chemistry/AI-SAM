# AI for SAM

## Introduction

Efficient design of perovskite SAM molecules based on quantum chemical calculations and artificial intelligence.

## Requirements

Python = 3.10

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
