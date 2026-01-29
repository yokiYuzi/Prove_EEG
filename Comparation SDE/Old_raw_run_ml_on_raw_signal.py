# run_ml_on_raw_signal.py
# -*- coding: utf-8 -*-

"""
直接使用 NPZ 文件中的原始信号进行经典机器学习分类。

功能要点：
- 数据加载：直接读取 500HZ_NIFEA_DB_processed_data.npz 文件。
- 预处理：
    1. 统一信号长度：将所有样本截断到数据集中最短的信号长度。
    2. 展平信号：将每个 (T, C) 的信号样本展平成一个一维向量。
- 机器学习：
    - 应用与原脚本类似的嵌套交叉验证流程，对展平后的原始信号进行分类。
    - 对比多种模型（LogReg/SVM/RF）。
- 保存报告：在 /mnt/data/ml_reports/ 下保存 CSV 报告。

运行示例：
  python run_ml_on_raw_signal.py \
      --data-path 500HZ_NIFEA_DB_processed_data.npz \
      --models logreg,rf,svm \
      --n-splits 5 --inner-splits 3 --repeats 1 \
      --subject-metric roc_auc
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             balanced_accuracy_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==============================================================================
#  Helper Functions (无变动)
# ==============================================================================

def ensure_dir(p):
    # 确保目录存在
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def build_models(which: List[str]) -> Dict[str, Tuple[Pipeline, Dict]]:
    # 构建机器学习模型和参数网格
    models = {}
    for name in which:
        name = name.strip().lower()
        if name == "logreg":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))
            ])
            grid = {"clf__C": [0.1, 1.0, 10.0]}
        elif name == "svm":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", SVC(probability=True))
            ])
            grid = {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale"], "clf__kernel": ["rbf"]}
        elif name == "rf":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
            ])
            grid = {"clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 5]}
        elif name == "gb":
            # 尽管默认不使用，但保留构建代码的灵活性
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier())
            ])
            grid = {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2,3]}
        else:
            raise ValueError(f"未知模型: {name}")
        models[name] = (pipe, grid)
    return models

def score_dict(y_true, y_prob, y_pred) -> Dict[str, float]:
    # 计算评估指标
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
    except Exception:
        auc, ap = np.nan, np.nan
    
    return {
        "roc_auc": float(auc),
        "average_precision":  float(ap),
        "f1":  float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }

# ==============================================================================
#  数据加载与预处理部分 (无变动)
# ==============================================================================

def load_and_prepare_raw_data(path_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 NPZ 文件加载原始信号，并进行预处理。
    """
    print(f"正在从 {path_npz} 加载原始信号数据...")
    data = np.load(path_npz, allow_pickle=True)
    signals = data["signals"]
    labels  = data["labels"].astype(int)

    min_len = min(s.shape[0] for s in signals)
    print(f"找到最短信号长度为: {min_len} 个时间步。所有样本将被截断至此长度。")

    num_samples = len(signals)
    num_channels = signals[0].shape[1]
    
    X = np.zeros((num_samples, min_len * num_channels), dtype=np.float32)

    for i, sig in enumerate(signals):
        truncated_sig = sig[:min_len, :]
        X[i, :] = truncated_sig.flatten()
    
    print(f"数据预处理完成。最终特征矩阵 X 的形状: {X.shape}")
    
    return X, labels

# ==============================================================================
#  机器学习流程部分 (无变动)
# ==============================================================================

def run_nested_cv(X: np.ndarray, y: np.ndarray, out_dir: str,
                  models: Dict[str, Tuple[Pipeline, Dict]], n_splits=5, inner_splits=3, repeats=1,
                  scoring_metric="roc_auc"):
    """
    在展平的原始信号上运行嵌套交叉验证。
    """
    ensure_dir(out_dir)
    
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=42)
    all_results = {}

    for model_name, (pipe, grid) in models.items():
        print(f"\n    正在运行模型: {model_name}...")
        fold_rows = []
        for fold_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=fold_id)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring=scoring_metric, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            y_prob = best.predict_proba(X_te)[:,1]
            y_pred = (y_prob >= 0.5).astype(int)
            sc = score_dict(y_te, y_prob, y_pred)
            fold_rows.append({
                "fold_id": fold_id,
                "best_params": json.dumps(gs.best_params_),
                **sc
            })

        df_res = pd.DataFrame(fold_rows)
        df_res.to_csv(os.path.join(out_dir, f"raw_signal_{model_name}_outer_scores.csv"), index=False, encoding="utf-8")
        all_results[model_name] = df_res

    means = {m: v.drop(columns=['best_params']).mean() for m, v in all_results.items()}
    pd.DataFrame([{"model": k, f"mean_{scoring_metric}": v[scoring_metric]} for k, v in means.items()])\
        .to_csv(os.path.join(out_dir, "raw_signal_model_selection.csv"), index=False, encoding="utf-8")
    
    return all_results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="500HZ_NIFEA_DB_processed_data.npz", help="指向原始信号 NPZ 文件的路径")
    ### --- 修改 --- ###
    # 删除了 'gb' 模型
    ap.add_argument("--models", type=str, default="logreg,rf,svm", help="要运行的模型列表，以逗号分隔")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--subject-metric", type=str, default="roc_auc", choices=["roc_auc","average_precision","f1","accuracy","balanced_accuracy"])
    args = ap.parse_args()

    out_root = "ml_reports_raw_signal" # 使用相对路径，方便查找
    ensure_dir(out_root)

    # === 1. 加载并预处理原始信号 ===
    X_raw, y_raw = load_and_prepare_raw_data(args.data_path)
    
    # === 2. 构建模型 ===
    model_list = [m.strip() for m in args.models.split(",")]
    models = build_models(model_list)

    # === 3. 运行嵌套交叉验证 ===
    print("\n===== 在展平的原始信号上运行嵌套交叉验证 =====")
    results = run_nested_cv(X_raw, y_raw, out_root, models,
                                n_splits=args.n_splits,
                                inner_splits=args.inner_splits,
                                repeats=args.repeats,
                                scoring_metric=args.subject_metric)
    
    # === 4. 生成总结报告 ===
    summary_rows = []
    print("\n\n" + "="*80)
    print(" " * 25 + "模型独立性能报告")
    print("="*80)

    for model_name, result_df in results.items():
        ### --- 修改：核心修复点 --- ###
        # 在计算均值前，使用 .drop() 方法丢弃非数值类型的 'best_params' 列
        mean_scores = result_df.drop(columns=['best_params']).mean()
        
        ### --- 新增：按要求单独打印每个模型的结果 --- ###
        print(f"\n--- 模型: {model_name} ---")
        print(f"  平均 ROC AUC:         {mean_scores.get('roc_auc', 0):.4f}")
        print(f"  平均 Accuracy:        {mean_scores.get('accuracy', 0):.4f}")
        print(f"  平均 F1-Score:        {mean_scores.get('f1', 0):.4f}")
        print(f"  平均 Balanced Acc:    {mean_scores.get('balanced_accuracy', 0):.4f}")
        print("-" * (20 + len(model_name)))
        
        # 为最终的总结表格准备数据
        summary_rows.append({
            "model": model_name,
            "auc_mean": f"{mean_scores.get('roc_auc', 0):.4f}",
            "accuracy_mean": f"{mean_scores.get('accuracy', 0):.4f}",
            "f1_mean": f"{mean_scores.get('f1', 0):.4f}",
            "balanced_accuracy_mean": f"{mean_scores.get('balanced_accuracy', 0):.4f}"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_root, "raw_signal_summary.csv"), index=False, encoding="utf-8")

    # 打印最终的整合表格
    print("\n\n" + "="*80)
    print(" " * 25 + "最终整合总结报告")
    print("="*80)
    print(df_summary.to_string())
    print("="*80)
    print(f"\n所有报告和详细分数已保存至: '{out_root}' 目录")


if __name__ == "__main__":
    main()