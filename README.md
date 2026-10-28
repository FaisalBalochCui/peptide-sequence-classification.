# Peptide Sequence Classification — Reproducible ML Demonstration

## Objective
Build a reproducible pipeline that turns peptide sequences into amino-acid composition features and evaluates a binary classifier.

## Data warning
`data/synthetic_peptides.csv` contains 500 synthetic sequences with artificially generated labels. It is for code/workflow demonstration only; it is not Alzheimer’s, patient, or validated biological data. Do not make disease-related claims from these results.

## Run
```bash
pip install -r requirements.txt
python src/run_analysis.py
```

## Methods
- Validate standard amino-acid symbols and binary labels
- Extract amino-acid composition and sequence length
- Stratified 80/20 train/test split
- Compare a majority-class baseline with Random Forest
- Report accuracy, precision, recall, F1 and ROC-AUC
- Save confusion matrix and feature-importance charts

## Outputs
`outputs/model_metrics.csv`, `outputs/classification_report.txt`, `outputs/confusion_matrix.png`, `outputs/feature_importance.png`.

## Next steps for real research
Replace the demo with a documented, appropriately licensed dataset; define labels and provenance; avoid duplicate-sequence/sample leakage; use independent validation; and involve domain expertise before making scientific claims.
