Dear Professor Federico Marini,

We are pleased to submit the manuscript entitled “Beyond Accuracy in ADMET Prediction: Applicability-Domain Diagnostics and Conformal Calibration under Chemical Distribution Shift” for consideration as an Original Research Article in *Chemometrics and Intelligent Laboratory Systems*.

The manuscript presents a frozen, representation-controlled chemometric audit of molecular-property reliability. Rather than proposing another predictive architecture or reporting a routine model leaderboard, we evaluate whether commonly used reliability diagnostics provide mutually consistent conclusions under repeated random, label-blind scaffold, and similarity-cluster splits. The protocol integrates point performance, probability calibration, continuous and threshold-sensitive applicability-domain analysis, marginal and class-conditional conformal prediction, prediction-set efficiency, and retention-aware selective prediction within paired endpoint–split–model–seed comparisons.

The study reveals several practically important conflicts that are not visible from aggregate accuracy alone. At nominal 90% coverage, marginal and shift-weighted conformal prediction achieved overall ClinTox coverage near target while positive-class coverage remained only 0.068–0.158. Mondrian calibration increased positive-class coverage to approximately 0.93–0.94, but produced ambiguous two-label sets for about 71% of samples. Chemical similarity ranked risk reliably for some endpoints but was weak or directionally reversed for others, and confidence-based selective prediction could reduce ordinary error partly by preferentially removing minority-class examples.

We believe the manuscript fits the journal because it is a non-routine application of chemometric validation, supervised modelling, robust comparison, applicability-domain analysis, and uncertainty calibration to a chemically structured decision problem. Its contribution is an auditable evaluation protocol and empirical demonstration that predictive accuracy, marginal validity, class-conditional validity, informativeness, chemical domain, and retained-class composition must be interpreted jointly. A recent broad foundation-model ADMET benchmark is explicitly discussed and distinguished: that work maps architecture performance under real-world challenges, whereas our controlled study isolates post-prediction reliability conflicts.

All confirmatory design choices, splits, model families, diagnostic definitions, and stopping rules were frozen before confirmatory outputs were inspected. Development seed 99 was excluded from scientific conclusions, adverse and null findings were retained, and model families were not treated as independent inferential replicates. The public repository provides source code, frozen manuscript-ready result tables, an integrity manifest, figures, LaTeX sources, and complete rebuilding instructions.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. Both authors have approved the submitted version and declare no competing interests. The use of generative AI and AI-assisted technologies during manuscript preparation is disclosed in the manuscript in accordance with Elsevier policy.

Thank you for considering our work. We believe it will be of interest to readers concerned with chemometric model validation, predictive uncertainty, applicability domains, and trustworthy machine learning in chemical and toxicological applications.

Sincerely,

S. Tong
Corresponding author
Department of Artificial Intelligence
Faculty of Computer Science and Information Technology
University of Malaya
50603 Kuala Lumpur, Malaysia
Email: 25064241@siswa.um.edu.my
