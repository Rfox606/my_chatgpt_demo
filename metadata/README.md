# Knee-wear experiment metadata

`knee_wear_experiment_metadata.json` is the repository-level, reusable record of the two knee-wear experiments: effective and actual cycle conventions, sensitive gait phase, the established effective-to-actual mapping, and Exp1 surface-morphology anchors.

The morphology values and interval descriptions are interpretation-only metadata. They must not be passed to online state calculation, feature selection, threshold determination, configuration selection, or online updates. Stage labels are likewise prohibited from online analysis.

The nominal sensitive point ranges record the experimental metadata. The feature-generation code can use discretized endpoints determined by the exact waveform length and the normalized 0.45–0.63 interval; analyses that need reproducibility should report both conventions rather than silently substituting one for the other.
