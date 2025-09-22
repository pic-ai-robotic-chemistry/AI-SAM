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

```bash
python ai_sam/funtional_groups_extraction_process.py
```

### 2. Molecule Generation

```bash
python ai_sam/mol_gen_process.py
```

### 3. High-throughput Theoretical Calculation

```bash
python ai_sam/dft_process.py
```

### 4. Molecule Screening and Ranking

```bash
python ai_sam/sort_process.py
```

### 5. Data Analysis

```bash
python ai_sam/analysis_process.py
```

## References

* Retrosynthetic Analysis:

(1) Zhang, B.; Zhang, X.; Du, W.; Song, Z.; Zhang, G.; Zhang, G.; Wang, Y.; Chen, X.; Jiang, J.; Luo, Y. Chemistry-Informed Molecular Graph as Reaction Descriptor for Machine-Learned Retrosynthesis Planning. Proc. Natl. Acad. Sci. U. S. A. 2022, 119 (41), e2212711119. https://doi.org/10.1073/pnas.2212711119.

