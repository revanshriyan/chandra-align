# Scientific dossier decision fields

Every registration run now reports the same decision fields, whether the gate
accepts, marks coarse advisory, or rejects the fit. The app telemetry contains
them; accepted and coarse runs also include them in the scientific dossier
image and transform JSON. Rejected runs include them in the judge-metrics JSON
and telemetry report even though transform values remain suppressed.

| Field | Meaning |
| --- | --- |
| `confidence_score` | A 0–100 heuristic evidence index, not a calibrated probability. It combines inlier support (30%), inlier ratio (25%), normalized entropy (20%), and gate-RMSE quality relative to 2.5 px (25%). |
| `confidence_basis` | Names the score components and states that the score is not a probability. |
| `decision_reason` | Human-readable explanation of the gate verdict or the gate's rejection message. |
| `fallback_path.steps` | Ordered matcher/registration engines used for the run. |
| `fallback_path.primary_engine` | Initial matching engine. |
| `fallback_path.registration_engine` | Engine whose correspondences produced the registration fit. |
| `fallback_path.fallback_triggered` | Whether execution took a fallback route. |
| `fallback_path.fallback_reason` | Why fallback was selected, when applicable. |
| `fallback_path.fallback_error` | Error from the primary or fallback attempt, when applicable. |

The score is descriptive metadata added after gate evaluation. It does not
change thresholds, verdicts, or matcher routing. `results/table_issue03_benchmark.csv`
contains these fields for each recorded Issue #3 matcher run; the source run
records are in `results/issue03_benchmark_runs.json`.
