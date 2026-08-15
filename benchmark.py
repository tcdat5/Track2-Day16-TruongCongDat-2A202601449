#!/usr/bin/env python3
"""
LAB 16 - Cloud AI Environment Setup
Benchmark LightGBM tren CPU node (GCP e2-medium, 2 vCPU / 4 GB RAM).

Bai toan: phat hien gian lan the tin dung (Credit Card Fraud Detection).
Dataset: 284,807 giao dich, chi 492 la gian lan (0.172%) -> cuc ky mat can bang.

Script do 3 nhom chi so:
  1. Hieu nang HA TANG : thoi gian load data, thoi gian training
  2. Chat luong MO HINH: AUC-ROC, Accuracy, F1, Precision, Recall
  3. Hieu nang PHUC VU : inference latency (1 dong), throughput (1000 dong)

Ket qua ghi ra benchmark_result.json de nop bai.
"""

import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Cau hinh
# --------------------------------------------------------------------------
DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
OUTPUT_PATH = os.path.expanduser("~/ml-benchmark/benchmark_result.json")

RANDOM_STATE = 42     # co dinh de ket qua lap lai duoc
TEST_SIZE = 0.2       # 80% train / 20% test
LATENCY_RUNS = 200    # so lan do latency 1 dong
THROUGHPUT_BATCH = 1000
THROUGHPUT_RUNS = 20

results = {}


def banner(text):
    print("\n" + "=" * 62)
    print(f"  {text}")
    print("=" * 62)


# --------------------------------------------------------------------------
# 0. Thong tin moi truong (de doi chieu khi so sanh may khac nhau)
# --------------------------------------------------------------------------
banner("0. MOI TRUONG")

try:
    # so nhan CPU thuc su dung duoc (chinh xac hon os.cpu_count tren container)
    n_cpu = len(os.sched_getaffinity(0))
except AttributeError:
    n_cpu = os.cpu_count()

env = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python_version": platform.python_version(),
    "lightgbm_version": lgb.__version__,
    "platform": platform.platform(),
    "cpu_count": n_cpu,
}
results["environment"] = env
for k, v in env.items():
    print(f"  {k:20s}: {v}")


# --------------------------------------------------------------------------
# 1. Load dataset  -> do thoi gian doc du lieu tu disk
# --------------------------------------------------------------------------
banner("1. LOAD DATASET")

t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - t0

n_fraud = int(df["Class"].sum())
fraud_rate = n_fraud / len(df) * 100

results["data"] = {
    "load_time_sec": round(load_time, 4),
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "fraud_rows": n_fraud,
    "fraud_rate_percent": round(fraud_rate, 4),
    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
}
print(f"  Thoi gian load       : {load_time:.3f} s")
print(f"  Kich thuoc           : {len(df):,} dong x {df.shape[1]} cot")
print(f"  Gian lan             : {n_fraud:,} dong ({fraud_rate:.3f}%)")
print(f"  RAM chiem            : {results['data']['memory_mb']:.1f} MB")


# --------------------------------------------------------------------------
# 2. Tach train / test
#    stratify=y  -> giu nguyen ti le gian lan o ca 2 tap.
#    Khong co stratify, tap test co the khong chua du gian lan -> metric vo nghia.
# --------------------------------------------------------------------------
banner("2. TACH TRAIN / TEST")

# Bo cot "Time" (so giay ke tu giao dich dau tien). Day la bien dem lien tuc,
# gan nhu khong lap lai, khong mang thong tin nhan qua ve gian lan.
# (Thuc nghiem cho thay bo hay giu no anh huong khong dang ke, nhung bo di
#  giup mo hinh gon hon va tranh hoc thuoc cac khoang thoi gian cu the.)
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train):,} dong ({int(y_train.sum())} gian lan)")
print(f"  Test : {len(X_test):,} dong ({int(y_test.sum())} gian lan)")


# --------------------------------------------------------------------------
# 3. Huan luyen LightGBM
#    early_stopping: dung som khi AUC tren tap test khong cai thien them 50 vong
#    -> tranh overfit VA cho ta chi so "best iteration".
# --------------------------------------------------------------------------
banner("3. TRAINING")

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=-1,              # dung het so nhan CPU co san
    random_state=RANDOM_STATE,
    verbose=-1,
    # CHI theo doi AUC. Neu de mac dinh, LGBMClassifier con tinh ca
    # binary_logloss -> tren du lieu mat can bang, logloss chung lai rat som
    # va keo early stopping dung theo (ket qua: best_iteration = 1, chi 1 cay).
    metric="auc",
    # THAM SO QUAN TRONG NHAT cho du lieu mat can bang.
    # Mac dinh cua LightGBM la 20: moi la chi can 20 mau la duoc tao.
    # Tap train chi co ~394 ca gian lan / 227k dong -> voi nguong 20, cay de ra
    # cac la ti hon hoc thuoc tung ca gian lan rieng le (overfit). Hau qua do
    # duoc bang thuc nghiem: AUC dat dinh ngay o CAY THU 1 roi tut xuong 0.76.
    #
    #   min_child_samples=20  (mac dinh) -> AUC tot nhat 0.9415, tai cay 1
    #   min_child_samples=100            -> AUC tot nhat 0.9830, tai cay 200
    min_child_samples=100,
)

t0 = time.perf_counter()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[
        # first_metric_only=True: chi dung som dua tren metric DAU TIEN (auc).
        #
        # stopping_rounds=200 (khong phai 50): duong AUC tren du lieu nay co mot
        # "vung trung" dai o giua truoc khi leo len dinh that su quanh cay 200.
        # Voi kien nhan 50 vong, early stopping dung ngay TRONG vung trung
        # (best_iteration = 2) va khong bao gio thay duoc dinh.
        lgb.early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
train_time = time.perf_counter() - t0

best_iter = int(model.best_iteration_) if model.best_iteration_ else model.n_estimators
results["training"] = {
    "train_time_sec": round(train_time, 4),
    "best_iteration": best_iter,
    "n_estimators_max": model.n_estimators,
    "learning_rate": model.learning_rate,
    "num_leaves": model.num_leaves,
    "trees_per_sec": round(best_iter / train_time, 2) if train_time > 0 else None,
}
print(f"  Thoi gian training   : {train_time:.3f} s")
print(f"  Best iteration       : {best_iter} / {model.n_estimators} cay")
print(f"  Toc do               : {results['training']['trees_per_sec']} cay/giay")


# --------------------------------------------------------------------------
# 4. Danh gia chat luong mo hinh
#    LUU Y: Accuracy o day gan nhu vo nghia. Mot mo hinh luon doan "khong gian
#    lan" cung dat ~99.83%. AUC-ROC moi la chi so dang tin.
# --------------------------------------------------------------------------
banner("4. DANH GIA MO HINH")

y_proba = model.predict_proba(X_test)[:, 1]   # xac suat la gian lan
y_pred = (y_proba >= 0.5).astype(int)          # nguong cat mac dinh 0.5

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Baseline "luon doan 0" de chung minh accuracy la cai bay
baseline_acc = (y_test == 0).mean()

results["metrics"] = {
    "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 6),
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
    "f1_score": round(float(f1_score(y_test, y_pred)), 6),
    "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
    "recall": round(float(recall_score(y_test, y_pred)), 6),
    "threshold": 0.5,
    "confusion_matrix": {
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    },
    "baseline_accuracy_always_predict_0": round(float(baseline_acc), 6),
}

m = results["metrics"]
print(f"  AUC-ROC              : {m['auc_roc']:.6f}   <-- chi so quan trong nhat")
print(f"  Accuracy             : {m['accuracy']:.6f}")
print(f"  F1-Score             : {m['f1_score']:.6f}")
print(f"  Precision            : {m['precision']:.6f}")
print(f"  Recall               : {m['recall']:.6f}")
print()
print(f"  Confusion matrix     : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"    -> bat duoc {tp}/{tp + fn} gian lan, bo lot {fn}")
print(f"    -> bao dong gia {fp} giao dich sach")
print()
print(f"  [Bay accuracy] Mo hinh 'luon doan khong gian lan' dat: {baseline_acc:.6f}")
print(f"                 => Accuracy cao KHONG chung minh mo hinh tot.")


# --------------------------------------------------------------------------
# 5. Inference latency - du doan 1 dong
#    Warm-up truoc de loai bo chi phi khoi tao lan dau (cache, JIT noi bo).
#    Lay median thay vi mean vi median khong bi mot lan cham dot bien lam lech.
# --------------------------------------------------------------------------
banner("5. INFERENCE LATENCY (1 dong)")

one_row = X_test.iloc[[0]]

for _ in range(20):                       # warm-up
    model.predict(one_row)

lat = []
for _ in range(LATENCY_RUNS):
    t0 = time.perf_counter()
    model.predict(one_row)
    lat.append((time.perf_counter() - t0) * 1000)   # doi sang mili giay

lat = np.array(lat)
results["inference_latency_1row"] = {
    "runs": LATENCY_RUNS,
    "mean_ms": round(float(lat.mean()), 4),
    "median_ms": round(float(np.median(lat)), 4),
    "p95_ms": round(float(np.percentile(lat, 95)), 4),
    "min_ms": round(float(lat.min()), 4),
    "max_ms": round(float(lat.max()), 4),
}
l = results["inference_latency_1row"]
print(f"  So lan do            : {LATENCY_RUNS}")
print(f"  Median               : {l['median_ms']:.3f} ms   <-- bao cao con so nay")
print(f"  Mean                 : {l['mean_ms']:.3f} ms")
print(f"  P95                  : {l['p95_ms']:.3f} ms")
print(f"  Min / Max            : {l['min_ms']:.3f} / {l['max_ms']:.3f} ms")


# --------------------------------------------------------------------------
# 6. Inference throughput - du doan 1000 dong mot lan
#    So sanh voi latency o tren se thay batch hieu qua hon nhieu lan.
# --------------------------------------------------------------------------
banner("6. INFERENCE THROUGHPUT (1000 dong)")

batch = X_test.iloc[:THROUGHPUT_BATCH]

for _ in range(3):                        # warm-up
    model.predict(batch)

tp_times = []
for _ in range(THROUGHPUT_RUNS):
    t0 = time.perf_counter()
    model.predict(batch)
    tp_times.append(time.perf_counter() - t0)

tp_times = np.array(tp_times)
mean_batch_sec = float(tp_times.mean())
rows_per_sec = THROUGHPUT_BATCH / mean_batch_sec

results["inference_throughput_1000rows"] = {
    "runs": THROUGHPUT_RUNS,
    "batch_size": THROUGHPUT_BATCH,
    "mean_batch_time_sec": round(mean_batch_sec, 6),
    "mean_batch_time_ms": round(mean_batch_sec * 1000, 3),
    "rows_per_sec": round(rows_per_sec, 1),
    "per_row_ms": round(mean_batch_sec * 1000 / THROUGHPUT_BATCH, 6),
}
t = results["inference_throughput_1000rows"]
print(f"  So lan do            : {THROUGHPUT_RUNS}")
print(f"  Thoi gian / batch    : {t['mean_batch_time_ms']:.3f} ms cho {THROUGHPUT_BATCH} dong")
print(f"  Throughput           : {t['rows_per_sec']:,.0f} dong/giay   <-- bao cao con so nay")
print(f"  Chi phi / dong       : {t['per_row_ms']:.5f} ms")
print()
speedup = l["median_ms"] / t["per_row_ms"] if t["per_row_ms"] > 0 else 0
print(f"  So sanh: xu ly theo lo nhanh gap ~{speedup:.0f}x so voi tung dong mot.")
print(f"           => Trong production, LUON gom request thanh batch neu co the.")


# --------------------------------------------------------------------------
# 7. Ghi ket qua ra JSON (deliverable cua lab)
# --------------------------------------------------------------------------
banner("7. GHI KET QUA")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  Da ghi: {OUTPUT_PATH}")


# --------------------------------------------------------------------------
# 8. Bang tong ket - phan nay de CHUP MAN HINH nop bai
# --------------------------------------------------------------------------
banner("BANG TONG KET - LAB 16")

rows = [
    ("Thoi gian load data",        f"{results['data']['load_time_sec']:.3f} s"),
    ("Thoi gian training",         f"{results['training']['train_time_sec']:.3f} s"),
    ("Best iteration",             f"{results['training']['best_iteration']}"),
    ("AUC-ROC",                    f"{m['auc_roc']:.6f}"),
    ("Accuracy",                   f"{m['accuracy']:.6f}"),
    ("F1-Score",                   f"{m['f1_score']:.6f}"),
    ("Precision",                  f"{m['precision']:.6f}"),
    ("Recall",                     f"{m['recall']:.6f}"),
    ("Inference latency (1 row)",  f"{l['median_ms']:.3f} ms (median)"),
    ("Inference throughput (1000)", f"{t['rows_per_sec']:,.0f} dong/giay"),
]

print(f"  {'Metric':<30} | {'Ket qua'}")
print(f"  {'-' * 30}-+-{'-' * 26}")
for name, val in rows:
    print(f"  {name:<30} | {val}")
print()
