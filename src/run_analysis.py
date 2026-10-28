from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"synthetic_peptides.csv"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
AAS=list("ACDEFGHIKLMNPQRSTVWY")

def features(seq):
    seq=str(seq).strip().upper()
    if not seq: raise ValueError("Empty sequence found")
    bad=set(seq)-set(AAS)
    if bad: raise ValueError(f"Invalid amino acid symbols: {sorted(bad)}")
    return {f"aac_{a}":seq.count(a)/len(seq) for a in AAS} | {"sequence_length":len(seq)}

def main():
    df=pd.read_csv(DATA).dropna(subset=["peptide_sequence","label"]).copy()
    df["label"]=df["label"].astype(int)
    if not set(df.label.unique()).issubset({0,1}): raise ValueError("label must contain only 0 and 1")
    X=pd.DataFrame([features(s) for s in df.peptide_sequence]); y=df.label
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={"Dummy baseline":DummyClassifier(strategy="most_frequent"),
            "Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")}
    rows={}; fitted={}
    for name,model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); fitted[name]=model
        rows[name]={"model":name,"accuracy":accuracy_score(yte,pred),
                    "precision":precision_score(yte,pred,zero_division=0),
                    "recall":recall_score(yte,pred,zero_division=0),
                    "f1":f1_score(yte,pred,zero_division=0),
                    "roc_auc":roc_auc_score(yte,model.predict_proba(Xte)[:,1]) if hasattr(model,"predict_proba") else np.nan}
    pd.DataFrame(rows.values()).to_csv(OUT/"model_metrics.csv",index=False)
    model=fitted["Random Forest"]; pred=model.predict(Xte)
    (OUT/"classification_report.txt").write_text("SYNTHETIC DEMONSTRATION ONLY — not biological evidence.\\n\\n"+classification_report(yte,pred,zero_division=0),encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(yte,pred); plt.title("Synthetic Peptide Demo — Confusion Matrix"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
    imp=pd.Series(model.feature_importances_,index=X.columns).sort_values().tail(12)
    imp.plot(kind="barh",figsize=(8,5)); plt.title("Feature Importance — Synthetic Demo"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()
    print(pd.DataFrame(rows.values()).to_string(index=False)); print(f"Outputs: {OUT}")

if __name__=="__main__": main()
