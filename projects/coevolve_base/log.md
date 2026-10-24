# Log for coevolve_base


---
### Run at 2025-11-05 08:27:22
## AI Summary
The experiment log indicates that a preview of the world was saved to the specified location. The project (coevolve_base) is performing evolutionary or optimization experiments, and the world preview is likely a visual representation of the current state of the world. The location of the file, F:\Projects\evolver-forge\projects\coevolve_base\world_preview.png, suggests that it may be a graphical user interface (GUI) or a command-line interface (CLI).


---
### Run at 2025-11-05 14:34:00
## AI Summary
The error message is indicating that the `label_points` function in `main.py` is trying to access a tuple (i.e., a list of values) using a tuple index instead of an integer or slice. Specifically, it's trying to access the elements of the `points` variable, which is a tuple, using a tuple index instead of an integer or slice.

The code is attempting to extract the x and y coordinates from the `points` tuple by using the following lines:
```python
x, y = points[:, 0], points[:, 1]
```
However, this is not valid syntax in Python. The correct way to access the elements of a tuple using integer or slice indices is as follows:
```python
x, y = points[0], points[1]
```
Alternatively, you can use the `zip` function to extract the x and y coordinates from the `points` tuple in a more concise way:
```python
x, y = zip(*points)
```
Once you've fixed this error, the code should run without any issues.


---
### Run at 2025-11-05 14:35:29
## AI Summary
This project (coevolve_base) is a software system that performs evolutionary or optimization experiments. The experiment log provided indicates that the system has successfully saved a preview of the world to the file path "F:\Projects\evolver-forge\projects\coevolve_base\world_preview.png".

Without further context, it is difficult to provide a more specific interpretation of this log. However, based on the context provided, it appears that the system has successfully completed an evolutionary or optimization experiment and saved a preview of the world as a result.


---
### Run at 2025-11-05 14:42:18
## AI Summary
The provided experiment log appears to be a record of an evolutionary or optimization experiment performed using the coevolve_base project. The log includes a reference to a saved preview image, which suggests that the experiment may have involved visualizing the results of the experiment in some way.

Based on this information, it is likely that the experiment was focused on evolving or optimizing some aspect of a virtual world or simulation, with the goal of producing a more realistic or effective outcome. The use of the term "coevolution" in the project name suggests that the experiment may have involved multiple agents or entities interacting and adapting to one another over time.

## Health Summary — 2025-11-07 11:24:11
🟡 Needs tuning
NN accuracy: 0.850 → 0.890 (↑)
Rule fidelity: 0.981 → 0.939 (↓)
Disagreement: 0.058 → 0.071 (↓)
Combined fitness: +0.915 → +0.914 (→)

## Health Summary — 2025-11-08 09:37:19
🟡 Needs tuning
NN accuracy: 0.852 → 0.900 (↑)
Rule fidelity: 0.988 → 0.928 (↓)
Disagreement: 0.054 → 0.068 (↓)
Combined fitness: +0.920 → +0.914 (↓)

## Health Summary — 2025-11-08 10:00:17
🟡 Needs tuning
NN accuracy: 0.829 → 0.900 (↑)
Rule fidelity: 0.990 → 0.928 (↓)
Disagreement: 0.075 → 0.068 (↑)
Combined fitness: +0.909 → +0.914 (→)

## Health Summary — 2025-11-08 10:00:17
🟡 Needs tuning
NN accuracy: 0.855 → 0.871 (↑)
Rule fidelity: 0.944 → 0.928 (↓)
Disagreement: 0.087 → 0.061 (↑)
Combined fitness: +0.899 → +0.900 (→)

## Health Summary — 2025-11-08 10:00:17
🟡 Needs tuning
NN accuracy: 0.825 → 0.858 (↑)
Rule fidelity: 0.975 → 0.928 (↓)
Disagreement: 0.071 → 0.055 (↑)
Combined fitness: +0.900 → +0.893 (↓)

## Health Summary — 2025-11-08 10:00:17
🟡 Needs tuning
NN accuracy: 0.833 → 0.877 (↑)
Rule fidelity: 0.989 → 0.928 (↓)
Disagreement: 0.081 → 0.057 (↑)
Combined fitness: +0.911 → +0.903 (↓)

## Health Summary — 2025-11-08 10:00:18
🟡 Needs tuning
NN accuracy: 0.825 → 0.883 (↑)
Rule fidelity: 0.977 → 0.928 (↓)
Disagreement: 0.077 → 0.054 (↑)
Combined fitness: +0.901 → +0.905 (→)

## Health Summary — 2025-11-08 10:00:27
🟡 Needs tuning
NN accuracy: 0.835 → 0.900 (↑)
Rule fidelity: 0.978 → 0.928 (↓)
Disagreement: 0.072 → 0.068 (→)
Combined fitness: +0.906 → +0.914 (↑)

## Health Summary — 2025-11-08 10:00:27
🟡 Needs tuning
NN accuracy: 0.885 → 0.871 (↓)
Rule fidelity: 0.951 → 0.928 (↓)
Disagreement: 0.076 → 0.061 (↑)
Combined fitness: +0.918 → +0.900 (↓)

## Health Summary — 2025-11-08 10:00:27
🟡 Needs tuning
NN accuracy: 0.807 → 0.858 (↑)
Rule fidelity: 0.968 → 0.928 (↓)
Disagreement: 0.072 → 0.055 (↑)
Combined fitness: +0.888 → +0.893 (↑)

## Health Summary — 2025-11-08 10:00:27
🟡 Needs tuning
NN accuracy: 0.819 → 0.877 (↑)
Rule fidelity: 0.983 → 0.928 (↓)
Disagreement: 0.068 → 0.057 (↑)
Combined fitness: +0.901 → +0.903 (→)

## Health Summary — 2025-11-08 10:00:28
🟡 Needs tuning
NN accuracy: 0.843 → 0.883 (↑)
Rule fidelity: 0.991 → 0.928 (↓)
Disagreement: 0.067 → 0.054 (↑)
Combined fitness: +0.917 → +0.905 (↓)

## Health Summary — 2025-11-08 10:18:42
🔴 Drifting
NN accuracy: 0.794 → 0.780 (↓)
Rule fidelity: 0.985 → 0.928 (↓)
Disagreement: 0.075 → 0.222 (↓)
Combined fitness: +0.889 → +0.854 (↓)

## Health Summary — 2025-11-08 10:48:06
🔴 Drifting
NN accuracy: 0.880 → 0.500 (↓)
Rule fidelity: 0.945 → 1.000 (↑)
Disagreement: 0.084 → 0.318 (↓)
Combined fitness: +0.912 → +0.750 (↓)

## Health Summary — 2025-11-08 10:48:19
🔴 Drifting
NN accuracy: 0.779 → 0.805 (↑)
Rule fidelity: 0.977 → 1.000 (↑)
Disagreement: 0.113 → 0.222 (↓)
Combined fitness: +0.878 → +0.902 (↑)

## Health Summary — 2025-11-08 10:48:31
🔴 Drifting
NN accuracy: 0.781 → 0.845 (↑)
Rule fidelity: 0.985 → 0.999 (↑)
Disagreement: 0.092 → 0.349 (↓)
Combined fitness: +0.883 → +0.922 (↑)

## Health Summary — 2025-11-09 00:32:49
🔴 Drifting
NN accuracy: 0.734 → 0.503 (↓)
Rule fidelity: 0.992 → 1.000 (↑)
Disagreement: 0.108 → 0.002 (↑)
Combined fitness: +0.863 → +0.752 (↓)

## Health Summary — 2025-11-09 00:32:49
🔴 Drifting
NN accuracy: 0.796 → 0.579 (↓)
Rule fidelity: 0.989 → 1.000 (↑)
Disagreement: 0.079 → 0.000 (↑)
Combined fitness: +0.892 → +0.790 (↓)

## Health Summary — 2025-11-09 00:32:49
🔴 Drifting
NN accuracy: 0.845 → 0.501 (↓)
Rule fidelity: 0.957 → 1.000 (↑)
Disagreement: 0.098 → 0.000 (↑)
Combined fitness: +0.901 → +0.750 (↓)

## Health Summary — 2025-11-09 00:32:50
🔴 Drifting
NN accuracy: 0.890 → 0.517 (↓)
Rule fidelity: 0.939 → 1.000 (↑)
Disagreement: 0.071 → 0.000 (↑)
Combined fitness: +0.914 → +0.759 (↓)

## Health Summary — 2025-11-09 00:32:50
🔴 Drifting
NN accuracy: 0.900 → 0.544 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.068 → 0.000 (↑)
Combined fitness: +0.914 → +0.772 (↓)

## Health Summary — 2025-11-09 00:41:46
🔴 Drifting
NN accuracy: 0.871 → 0.501 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.061 → 0.004 (↑)
Combined fitness: +0.900 → +0.750 (↓)

## Health Summary — 2025-11-09 00:41:46
🔴 Drifting
NN accuracy: 0.858 → 0.561 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.055 → 0.000 (↑)
Combined fitness: +0.893 → +0.781 (↓)

## Health Summary — 2025-11-09 00:41:46
🔴 Drifting
NN accuracy: 0.877 → 0.501 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.057 → 0.000 (↑)
Combined fitness: +0.903 → +0.750 (↓)

## Health Summary — 2025-11-09 08:04:24
🔴 Drifting
NN accuracy: 0.900 → 0.502 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.068 → 0.447 (↓)
Combined fitness: +0.914 → +0.751 (↓)

## Health Summary — 2025-11-09 08:04:24
🔴 Drifting
NN accuracy: 0.871 → 0.542 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.061 → 0.295 (↓)
Combined fitness: +0.900 → +0.771 (↓)

## Health Summary — 2025-11-09 08:04:25
🔴 Drifting
NN accuracy: 0.858 → 0.502 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.055 → 0.274 (↓)
Combined fitness: +0.893 → +0.751 (↓)

## Health Summary — 2025-11-09 09:53:35
🔴 Drifting
NN accuracy: 0.883 → 0.500 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.054 → 0.462 (↓)
Combined fitness: +0.905 → +0.750 (↓)

## Health Summary — 2025-11-09 09:53:36
🔴 Drifting
NN accuracy: 0.780 → 0.510 (↓)
Rule fidelity: 0.928 → 1.000 (↑)
Disagreement: 0.222 → 0.441 (↓)
Combined fitness: +0.854 → +0.755 (↓)

## Health Summary — 2025-11-09 09:53:36
🟡 Needs tuning
NN accuracy: 0.500 → 0.500 (→)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.318 → 0.440 (↓)
Combined fitness: +0.750 → +0.750 (→)

## Health Summary — 2025-11-09 09:53:37
🟡 Needs tuning
NN accuracy: 0.805 → 0.502 (↓)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.222 → 0.336 (↓)
Combined fitness: +0.902 → +0.751 (↓)

## Health Summary — 2025-11-09 09:54:05
🔴 Drifting
NN accuracy: 0.503 → 0.694 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.002 → 0.289 (↓)
Combined fitness: +0.752 → +0.847 (↑)

## Health Summary — 2025-11-09 09:54:05
🔴 Drifting
NN accuracy: 0.579 → 0.716 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.290 (↓)
Combined fitness: +0.790 → +0.858 (↑)

## Health Summary — 2025-11-09 09:54:06
🟡 Needs tuning
NN accuracy: 0.501 → 0.704 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.221 (↓)
Combined fitness: +0.750 → +0.852 (↑)

## Health Summary — 2025-11-09 09:54:42
🔴 Drifting
NN accuracy: 0.517 → 0.861 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.237 (↓)
Combined fitness: +0.759 → +0.931 (↑)

## Health Summary — 2025-11-09 09:54:43
🟡 Needs tuning
NN accuracy: 0.544 → 0.830 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.229 (↓)
Combined fitness: +0.772 → +0.915 (↑)

## Health Summary — 2025-11-09 09:54:44
🟡 Needs tuning
NN accuracy: nan → 0.776 (→)
Rule fidelity: nan → 1.000 (→)
Disagreement: nan → 0.225 (→)
Combined fitness: +nan → +0.888 (→)

== Robustness Report ==
Noise: acc_std=0.000, fid_std=0.000, drop@5%LN=0.000
Ablations: Δ(no_rule)=0.000, Δ(no_nn)=0.000, Δ(no_explore)=0.000
Stability score: 1.000

## Health Summary — 2025-11-10 00:21:18
🔴 Drifting
NN accuracy: 0.561 → 0.498 (↓)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.238 (↓)
Combined fitness: +0.781 → +0.749 (↓)

## Health Summary — 2025-11-10 00:21:19
🔴 Drifting
NN accuracy: 0.501 → 0.498 (→)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.000 → 0.151 (↓)
Combined fitness: +0.750 → +0.749 (→)

## Health Summary — 2025-11-10 00:21:19
🔴 Drifting
NN accuracy: nan → 0.560 (→)
Rule fidelity: nan → 1.000 (→)
Disagreement: nan → 0.426 (→)
Combined fitness: +nan → +0.780 (→)

## Health Summary — 2025-11-10 00:21:20
🟡 Needs tuning
NN accuracy: 0.502 → 0.579 (↑)
Rule fidelity: 1.000 → 1.000 (→)
Disagreement: 0.447 → 0.364 (↑)
Combined fitness: +0.751 → +0.789 (↑)

== Causal Attribution ==
arch.explore.lambda_consistency ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.explore.lambda_consistency ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.explore.lambda_consistency ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.explore.mutation_rate ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.explore.mutation_rate ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.explore.mutation_rate ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.explore.uncertainty_k ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.explore.uncertainty_k ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.explore.uncertainty_k ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.nn.activation ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.nn.activation ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.nn.activation ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.nn.dropout ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.nn.dropout ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.nn.dropout ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.nn.hidden_layers ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.nn.hidden_layers ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.nn.hidden_layers ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.nn.normalization ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.nn.normalization ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.nn.normalization ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
arch.rules.prune_ccp_alpha ⇒ nn_accuracy: DiD=+0.071 ↑ | Granger=1.333 (n=1) (low evidence)
arch.rules.prune_ccp_alpha ⇒ rule_fidelity: DiD=+0.012 ↑ | Granger=1.716 (n=1) (low evidence)
arch.rules.prune_ccp_alpha ⇒ combined_fitness: DiD=+0.041 ↑ | Granger=1.525 (n=1) (low evidence)
meta.lambda_consistency ⇒ nn_accuracy: DiD=+0.058 ↑ | Granger=0.834 (n=5)
meta.lambda_consistency ⇒ rule_fidelity: DiD=+0.003 ↑ | Granger=1.076 (n=5)
meta.lambda_consistency ⇒ combined_fitness: DiD=+0.030 ↑ | Granger=0.955 (n=5)
meta.learning_rate ⇒ nn_accuracy: DiD=+0.065 ↑ | Granger=0.834 (n=4)
meta.learning_rate ⇒ rule_fidelity: DiD=+0.005 ↑ | Granger=1.076 (n=4)
meta.learning_rate ⇒ combined_fitness: DiD=+0.035 ↑ | Granger=0.955 (n=4)
meta.rule_mutation_rate ⇒ nn_accuracy: DiD=+0.065 ↑ | Granger=0.834 (n=4)
meta.rule_mutation_rate ⇒ rule_fidelity: DiD=+0.005 ↑ | Granger=1.076 (n=4)
meta.rule_mutation_rate ⇒ combined_fitness: DiD=+0.035 ↑ | Granger=0.955 (n=4)
noise.feature ⇒ nn_accuracy: DiD=-0.014 ↓ | Granger=1.411 (n=1) (low evidence)
noise.feature ⇒ rule_fidelity: DiD=+0.010 ↑ | Granger=1.821 (n=1) (low evidence)
noise.feature ⇒ combined_fitness: DiD=-0.002 ↓ | Granger=1.616 (n=1) (low evidence)
noise.label ⇒ nn_accuracy: DiD=-0.014 ↓ | Granger=1.411 (n=1) (low evidence)
noise.label ⇒ rule_fidelity: DiD=+0.010 ↑ | Granger=1.821 (n=1) (low evidence)
noise.label ⇒ combined_fitness: DiD=-0.002 ↓ | Granger=1.616 (n=1) (low evidence)
noise.target ⇒ nn_accuracy: DiD=-0.014 ↓ | Granger=1.411 (n=1) (low evidence)
noise.target ⇒ rule_fidelity: DiD=+0.010 ↑ | Granger=1.821 (n=1) (low evidence)
noise.target ⇒ combined_fitness: DiD=-0.002 ↓ | Granger=1.616 (n=1) (low evidence)
