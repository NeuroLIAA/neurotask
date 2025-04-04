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
from / path / to / your / custom_mapper
import CustomMapper

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
```

## CSV Output Format

After running the TMT analysis, the results are saved as a CSV file. Each row in the CSV represents a trial (valid or
invalid) and contains the following columns:

| Column Name                    | Data Type      | Description                                                                                   |
|--------------------------------|----------------|-----------------------------------------------------------------------------------------------|
| **subject_id**                 | String         | Unique identifier for the subject.                                                            |
| **trial_id**                   | String         | Unique identifier for the trial.                                                              |
| **trial_type**                 | String         | The type of trial, represented by its name.                                                   |
| **age**                        | Number         | Age of the subject (in years).                                                                |
| **gender**                     | String         | Gender of the subject.                                                                        |
| **is_valid**                   | Boolean        | Indicates whether the trial is valid (`True`) or invalid (`False`).                           |
| **trial_order_of_appearance**  | Integer        | The sequential order of the trial within the experiment.                                      |
| **speed_threshold**            | Float          | The speed threshold calculated for the subject.                                               |
| **total_distance**             | Float          | Total distance traveled by the cursor during the trial.                                       |
| **rt**                         | Float          | Reaction time (total duration) of the trial.                                                  |
| **correct_targets_touches**    | Integer or NaN | Number of correct target touches. For invalid trials, this is set to `NaN`.                   |
| **wrong_targets_touches**      | Integer or NaN | Number of wrong target touches. For invalid trials, this is set to `NaN`.                     |
| **mean_speed**                 | Float or NaN   | Average cursor speed during the trial.                                                        |
| **std_speed**                  | Float or NaN   | Standard deviation of the cursor speed.                                                       |
| **peak_speed**                 | Float or NaN   | Maximum cursor speed observed during the trial.                                               |
| **mean_acceleration**          | Float or NaN   | Average acceleration of the cursor during the trial.                                          |
| **std_acceleration**           | Float or NaN   | Standard deviation of the cursor acceleration.                                                |
| **peak_acceleration**          | Float or NaN   | Maximum acceleration observed during the trial.                                               |
| **mean_abs_acceleration**      | Float or NaN   | Average of the absolute acceleration values.                                                  |
| **std_abs_acceleration**       | Float or NaN   | Standard deviation of the absolute acceleration values.                                       |
| **peak_abs_acceleration**      | Float or NaN   | Maximum absolute acceleration observed.                                                       |
| **mean_negative_acceleration** | Float or NaN   | Average of the negative acceleration values (deceleration).                                   |
| **std_negative_acceleration**  | Float or NaN   | Standard deviation of the negative acceleration values.                                       |
| **peak_negative_acceleration** | Float or NaN   | Most negative acceleration value observed.                                                    |
| **hesitation_distance**        | Float or NaN   | Distance over which hesitation was observed (if applicable).                                  |
| **hesitation_time**            | Float or NaN   | Duration of hesitation (if applicable).                                                       |
| **number_of_crosses**          | Integer or NaN | Number of crosses detected in the trial. Calculated only if enabled; otherwise, set to `NaN`. |
| **invalid_cause**              | String         | For invalid trials, a description of why the trial was marked invalid.                        |

> **Note:** For invalid trials, many of the metric columns (e.g., distances, speeds, accelerations) will be set to
> default values or `NaN` to indicate that the trial did not pass validation checks.
