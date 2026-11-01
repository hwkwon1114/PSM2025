# Pairwise calibration figures

Each pair minimizes the equal-weight joint RMS of its two training cases. The three searches use the same initial 16-point LHS and then adaptive BO, with the same four-parameter bounds including floor [0, 1]. 16 saved initial candidates could be matched across all three; 12: 48 attempted, 44 valid, status early_stop_no_improvement; 23: 92 attempted, 90 valid, status early_stop_no_improvement; 13: 48 attempted, 44 valid, status early_stop_no_improvement. Candidates are dependent adaptive evaluations, not independent experimental replicates.

**Optimization history:** dots show valid candidates; the step is the best valid objective through each completed batch. Crosses at the axis bottom denote failed candidates and have no objective value. The dashed line separates LHS from BO. The axis is logarithmic unless an exact zero is observed, when it is symmetric-log with a 1e-6 mm linear region.

**Parameter observations:** all valid evaluated candidates are shown against each physical parameter; stars mark the minimum joint objective in each search. Growth and beta use logarithmic axes. These are projections of the sampled four-dimensional response, not profiles, confidence intervals or proofs of identifiability. At floor = 1 beta has no mechanical effect. Search stopping and bounds limit all comparisons.

Each chosen fit is frozen before the excluded-case check; outcomes and failures are in pairwise_provenance.json. All three cases were previously seen, so the checks are exploratory, not blind validation. See the CSV for every attempt, including failures, and provenance JSON for input identities and frozen transfer outcomes.
