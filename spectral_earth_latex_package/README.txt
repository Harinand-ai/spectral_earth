========================================================================
SPECTRALEARTH ACTIVE LEARNING IEEE RESEARCH MANUSCRIPT PACKAGE
========================================================================

Contents:
- paper.tex: Complete IEEEtran conference LaTeX manuscript source
- references.bib: Complete BibTeX bibliography citations
- downstream_tasks_overview.png: System architecture diagram (Fig. 1)
- fig_oa_curve.png: Active learning overall accuracy progression curve (Fig. 2a)
- fig_multiseed_curves.png: 5-seed Mean +/- Std learning curves (Fig. 2b)
- fig_ece_calibration.png: Temperature scaling logit calibration chart (Fig. 2c)
- fig_cross_sensor_transfer.png: AVIRIS to ROSIS few-shot transfer curves (Fig. 3)

Compilation Instructions:
1. Overleaf:
   Simply upload this zip file (spectral_earth_latex_report.zip) directly
   into Overleaf (New Project -> Upload Project). It will automatically
   compile into the final PDF.

2. Local LaTeX (TeX Live / MiKTeX / MacTeX):
   pdflatex paper
   bibtex paper
   pdflatex paper
   pdflatex paper
========================================================================
