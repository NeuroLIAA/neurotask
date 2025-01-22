# Neurotask

## Description

**Neurotask** is a Python library designed to facilitate the analysis and processing of neuropsychological task data.

## Installation

You can install **neurotask** via pip: **[WORK IN PROGRESS]**

```bash
pip install pyxations
```

or from local package:

```bash
cd /path/to/your/package
pip install .
```

If you're developing the package and want changes to the code to be immediately reflected without reinstalling, you can
use the -e option to install the package in "editable" mode:

```bash
pip install -e .
```

## Usage Example

Here's a simple example of how to use the **neurotask** library to perform a Trail Making Test (TMT) analysis:

```python
from neurotask.tmt.tmt_analyzer import TMTAnalyzer
from /path/to/your/custom_mapper import CustomMapper
  
# Initialize the TMT Analyzer
analysis = TMTAnalyzer(
    mapper=CustomMapper(),
    dataset_path="/path/to/your/dataset",
    output_path="/path/to/save/results",
    correct_targets_minimum=12,
    consecutive_points=5
)

# Run the analysis
analysis.run()

# Retrieve and display metrics
metrics_df = analysis.get_metrics_dataframe()
print(metrics_df.head())

# Retrive the experiment to access tmt model
experiment_model = analysis.experiment