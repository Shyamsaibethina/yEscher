# yEscher

`yEscher` is a Python package designed to simulate metabolic flux distributions, gene knockouts, and dynamic modeling for *S. cerevisiae*. It is built on top of the yeastGEM metabolic model and integrates with various analysis libraries, such as `cobra`, `pytfa`, and `optlang`. This tool allows for both static and dynamic flux balance analysis (FBA) as well as enzyme usage simulations.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Running Knockout Simulations](#running-knockout-simulations)
  - [Performing Flux Variability Analysis](#performing-flux-variability-analysis)
  - [Identifying Essential Genes](#identifying-essential-genes)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Metabolic Simulation**: Run gene knockout simulations and simulate metabolic flux distributions using eTFLa models.
- **Dynamic FBA**: Model the dynamic response of enzyme concentrations and substrate uptakes over time.
- **Flux Variability Analysis**: Perform flux variability analysis to explore alternative metabolic pathways.
- **Essential Gene Identification**: Identify essential genes using single-gene deletion analysis.
- **Integrated Visualization**: Automatically generate metabolic maps using `Escher` for intuitive flux visualization.

## Installation

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.7 or higher
- `pip` (Python package installer)

### Installing yEscher

1. Clone the repository:
    ```bash
    git clone https://github.com/Shyamsaibethina/yEscher.git
    cd yEscher
    ```

2. Install the package and dependencies:
    ```bash
    pip install .
    ```

   Alternatively, you can install the package in development mode using:
    ```bash
    pip install -e .
    ```

### Installing with Conda

If you prefer using Conda, you can create a virtual environment and install the package as follows:

```bash
conda create -n yescher python=3.9
conda activate yescher
pip install .
```

### Additional Dependencies

You may need the following libraries, which are included in `requirements.txt`:

- `optlang`
- `pytfa`
- `cobra`
- `pandas`
- `sympy`
- `escher`
- `gurobipy` (for optimization)

Run this command to install all dependencies:

```bash
pip install -r requirements.txt
```

### Running Knockout Simulations

The `knockout` function allows for simulating gene knockouts and generating metabolic maps. The model path is now referenced using an `Enum` class (`ETFL`).

```python
from yEscher.knockout import knockout
from yEscher.enumETFL import ETFL

# Use the ETFL Enum to get the path to the eTFLa model
etfl_model_path = ETFL.CEFL.value

# Paths for output
map_file_path = './output/maps/'
csv_file_path = './output/csv/'
knockouts = ['gene1', 'gene2']

# Run knockout simulation
knockout(growth_rate=0.5, knockouts=knockouts, map_file_path=map_file_path, 
         csv_file_path=csv_file_path, map_name='yeast_map', etfl_model_path=etfl_model_path)
```

### Performing Flux Variability Analysis

The `perform_flux_variability_analysis` function helps analyze the flux ranges for all reactions in the model.

```python
from yEscher.flux_analysis import perform_flux_variability_analysis
from yEscher.enumETFL import ETFL

# Load eTFLa model using the ETFL enum
etfl_model_path = ETFL.CEFL.value

# Perform FVA
fva_results = perform_flux_variability_analysis(ctrl_model)

# Display FVA results
print(fva_results)
```

### Identifying Essential Genes

The `identify_essential_genes` function identifies genes that are critical for the survival of the organism.

```python
from yEscher.gene_analysis import identify_essential_genes
from yEscher.enumETFL import ETFL

# Load eTFLa model using the ETFL enum
etfl_model_path = ETFL.CEFL.value

# Identify essential genes
essential_genes = identify_essential_genes(ctrl_model)

# Display essential genes
print(essential_genes)
```

## License

yEscher is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

