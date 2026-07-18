
import os  # klasor oluşturmak için
import json # model bilgilerini .json dosyasına yazmak için
import joblib # eğitilmiş modeli dosyaya kaydetmek için
import mlflow
import mlflow.sklearn 
from training_utils import calculate_threshold_metrics


# model oluşturmak için
from sklearn.pipeline import Pipeline  # pipeline: birden fazla işlemi sıraya koyar 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import(  # metrik importları 
    roc_auc_score,
    average_precision_score,
   )
from preprocess import load_data, split_data
from visualization_utils import plot_threshold_metrics


X,y = load_data()


X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)  

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud_detection")  #MLflow UI’daki experiment adı




models  = {
    
    "Logistic Regression" : Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight = "balanced",
        max_iter = 1000,  # öğrenme sürecinde maksimum iterasyon
        random_state = 42  # sonuçlar her çalıştırmada aynı olsun diye.


    ))

    ]),

   "Random Forest" : RandomForestClassifier(
    n_estimators=100,  # 100 tane karar ağacı kullan
    class_weight = "balanced",
    random_state = 42,
    n_jobs = -1  # bilgisayardaki uygun tüm işlemci çekirdeklerini kullan, training daha hızlı 
   )

 }
 



thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

trained_models = {}   # eğitilmiş modelleri saklar, sözlük 
val_probas = {}       # her modelin validation fraud probability skorlarını saklar
pr_auc_scores = {}    # her modelin PR-AUC skorunu saklar


# AŞAMA 1: MODEL SEÇİMİ - score-based metrik (PR-AUC) ile

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        trained_models[model_name] = model # model eğitildikten sonra dictionary içine koyduk

        y_proba = model.predict_proba(X_val)[:,1]  # validation seti için(traınden egittik simdi modelin görmediği validation verisinde nasıl davrandığını ölçüyoruz.) fraud probability hesaplama

        val_probas[model_name] = y_proba

        roc_auc = roc_auc_score(y_val, y_proba) # Model gerçek fraudlara, normal işlemlerden daha yüksek skor verebiliyor mu
        pr_auc = average_precision_score(y_val, y_proba)
        pr_auc_scores[model_name] = pr_auc

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("selection_metric", "val_pr_auc")

        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.log_metric("val_pr_auc", pr_auc)

        #mlflow.end_run()  wıth kullanmasaydık elle kapatırdık

        print(f"{model_name} -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
  
best_model_name = max(pr_auc_scores, key=pr_auc_scores.get)
best_model = trained_models[best_model_name]  
best_model_proba = val_probas[best_model_name]  


print(f"\n>>> Model seçimi (PR-AUC'a göre): {best_model_name} "
      f"(PR-AUC: {pr_auc_scores[best_model_name]:.4f})")

print("\n==============================")
print(
    f"AŞAMA 2: Threshold Seçimi "
    f"({best_model_name} için, F1 bazlı)"
)
print("==============================")

best_f1 = -1.0  # F1 skoru normalde 0 ile 1 arasında
best_threshold = None
threshold_results = []

for threshold in thresholds:
    metrics = calculate_threshold_metrics(
        y_val,
        best_model_proba,
        threshold
    )

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tp = metrics["tp"]

    print(...)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

if best_threshold is None:
        raise ValueError("Best threshold could not be selected.")


print(
    f"\n>>> Seçilen eşik: {best_threshold:.2f} "
    f"(Validation F1: {best_f1:.4f})"
)

plot_threshold_metrics(threshold_results, best_threshold)



print("\n==============================")
print("Final Test Evaluation")
print("==============================")

final_model = best_model
final_threshold = best_threshold


y_test_proba = final_model.predict_proba(X_test)[:, 1]

test_metrics = calculate_threshold_metrics(
    y_test,
    y_test_proba,

    final_threshold
)

y_test_pred = test_metrics["y_pred"]

precision = test_metrics["precision"]
recall = test_metrics["recall"]
f1 = test_metrics["f1"]

tn = test_metrics["tn"]
fp = test_metrics["fp"]
fn = test_metrics["fn"]
tp = test_metrics["tp"]

roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

print("Final Test Evaluation Results")
print("------------------------")
print("Final Model:", best_model_name)
print("Final Threshold:", final_threshold)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("PR-AUC Score:", pr_auc)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)



os.makedirs("models", exist_ok=True)  # model klasorü oluştur, varsa hata verme

joblib.dump(final_model, "models/fraud_model.pkl")  # Final modeli kaydetme




model_info = {  # model hakkında okunabilir bilgiler
    "model_name": best_model_name,
    "threshold": float(final_threshold),
    "threshold_selection_metric": "f1",
    "validation_best_f1": float(best_f1),
    "feature_columns": list(X_train.columns),

    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_roc_auc": float(roc_auc),
    "test_pr_auc": float(pr_auc),

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

with mlflow.start_run(run_name=f"final-{best_model_name}"):
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_param("threshold", final_threshold)
    mlflow.log_param("threshold_strategy", "max_validation_f1")
    mlflow.log_param("model_selection_metric", "val_pr_auc")

    mlflow.log_metric("validation_best_f1", best_f1)

    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)
    mlflow.log_metric("test_pr_auc", pr_auc)

    mlflow.log_metric("test_tn", tn)
    mlflow.log_metric("test_fp", fp)
    mlflow.log_metric("test_fn", fn)
    mlflow.log_metric("test_tp", tp)

    mlflow.log_artifact("models/fraud_model.pkl")
    mlflow.log_artifact("models/model_info.json")
    mlflow.sklearn.log_model(final_model, name="model")

print("\nModel saved to models/fraud_model.pkl")
print("Model info saved to models/model_info.json")

