Dear Professor Federico Marini,

We are pleased to submit the manuscript entitled "Beyond Aggregate Reliability in ADMET Prediction: Protecting Weakest-Class Coverage under Molecular Distribution Shift with a Transport-Audited Multi-View Envelope" for consideration as an Original Research Article in *Chemometrics and Intelligent Laboratory Systems*.

The manuscript presents a two-stage chemometric study of molecular-prediction reliability. First, a frozen, representation-controlled audit evaluates point performance, probability calibration, applicability-domain behavior, marginal and class-conditional conformal validity, prediction-set informativeness, and retained-class composition under repeated random, label-blind scaffold, and similarity-cluster splits. The audit shows why no one aggregate reliability quantity is sufficient. At nominal 90% coverage, for example, ClinTox marginal and shift-weighted conformal procedures achieved overall coverage near target while positive-class coverage was only 0.068-0.158. Mondrian calibration restored positive coverage to approximately 0.93-0.94, but returned ambiguous two-label sets for about 71% of compounds.

Second, the observed failures are translated into explicit design constraints for TAME, a transport-audited multi-view conformal envelope evaluated under the separately frozen RACER-C4 protocol. TAME permits two label-free transport views to affect the ordinary Mondrian baseline only after prespecified effective-sample-size, clipping, domain-discrimination, and balance checks. Its construction always contains the baseline set, emits no empty set, cannot create a new confident singleton, and returns an exact ordinary fallback when a transport audit fails.

Architecture development was restricted to the public Tox21 leaderboard cohort. The method, primary endpoints, fresh seeds, promotion gate, five-percentage-point Macro correct-singleton yield point-estimate guardrail, and negative-result policy were frozen before independent labels were opened. Final EPA predictions and transport audits were written and hashed before the locked label file was acquired and parsed. Across six primary endpoints and five fresh seeds, TAME improved the endpoint-seed-equal mean minimum-class coverage by 1.3649 percentage points; the deterministic hierarchical-bootstrap 95% interval was +0.5827 to +2.0051 points. Macro correct-singleton yield changed by -1.6067 points, inside the frozen efficiency guardrail. The manuscript does not present this secondary point-estimate comparison as a formal non-inferiority test. Both approved transport views passed in all 60 final endpoint-seed cells, whereas the diagnostic ECFP view failed in all 60.

We believe the manuscript fits the journal because its contribution is a non-routine combination of chemometric validation, applicability-domain analysis, calibrated predictive sets, explicit transport diagnostics, paired repeated-split inference, and prospective evidence control. It complements architecture-centered ADMET benchmarks by isolating post-prediction reliability conflicts and by testing a fixed intervention under a prediction-to-label firewall.

The manuscript does not claim exact coverage under arbitrary molecular distribution shift, conditional coverage, clinical safety, formal efficiency non-inferiority, or universal algorithmic superiority. Null and adverse findings were retained, including one primary endpoint with zero minimum-class-coverage change and every negative endpoint-level efficiency change. The public reproducibility repository contains the executable lock, code, tests, cryptographic manifests, frozen summaries, vector figures, LaTeX sources, and rebuilding instructions.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. Both authors have approved the submitted version and declare no competing interests. The use of generative AI and AI-assisted technologies during manuscript preparation is disclosed in the manuscript in accordance with Elsevier policy.

Thank you for considering our work. We believe it will be of interest to readers concerned with chemometric model validation, trustworthy molecular prediction, conformal inference, and applicability domains under structured distribution shift.

Sincerely,

Siyuan Tong

Corresponding author

Department of Artificial Intelligence

Faculty of Computer Science and Information Technology

University of Malaya

50603 Kuala Lumpur, Malaysia

Email: 25064241@siswa.um.edu.my
