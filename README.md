# Interpretable_Control
Mechanistic interpretability probing experiments for vision-based control policies.
Hypothesis: End-to-end driving models (e.g. TransFuser) outperform two-stage
perception-planning pipelines because they encode only a coarse sense of distance
rather than precise metric depth which is analogous to how humans drive without accurate
depth measurement.
We test this by fitting linear probes on the internal activations of a pretrained
policy and comparing how well each layer encodes exact vs. coarse distance to
the nearest obstacle.
