# run_ml_on_exported_features.py
# -*- coding: utf-8 -*-

"""
使用已导出的特征（features_subject.csv / features_window.csv）进行经典机器学习分类、消融与可解释性分析。

功能要点：
- 受试者级：对 features_subject.csv 做嵌套交叉验证（外层评估、内层调参），对比多种模型（LogReg/SVM/RF/GBDT）。
- 消融：仅使用 A_*（空间注意力转移家族） vs 使用全部特征。
- 校准：对外层最佳模型进行概率校准（Platt），报告 Brier 分数与校准曲线数据。
- 置换重要度：对最佳模型做 permutation importance，保存 Top-20 特征。
- 窗口级（可选）：读取 features_window.csv，若存在 subject 列且每个 subject 有多窗口，则执行 GroupKFold，
  并在每个外层测试折上“按 subject 聚合窗口概率”（mean/median/top-k），得到受试者级指标。
- 保存报告：/mnt/data/ml_reports/ 下保存 CSV/PNG。

运行示例：
  python run_ml_on_exported_features.py \
      --features-subject features_subject.csv \
      --features-window  features_window.csv \
      --models logreg,svm,rf,gb \
      --n-splits 5 --inner-splits 3 --repeats 2 \
      --subject-metric auc
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             balanced_accuracy_score, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_features(path_csv: str):
    df = pd.read_csv(path_csv)
    if "label" not in df.columns:
        raise ValueError(f"{path_csv} 缺少 label 列")
    # 自动识别特征列（排除非数值与标识列）
    exclude = set(["label", "sample_tag", "subject"])
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return df, feat_cols

def build_models(which: List[str]) -> Dict[str, Tuple[Pipeline, Dict]]:
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
            grid = {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", "auto"], "clf__kernel": ["rbf"]}
        elif name == "rf":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
            ])
            grid = {"clf__max_depth": [None, 6, 10], "clf__min_samples_leaf": [1, 3]}
        elif name == "gb":
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
    """
    计算并返回一个包含多个评估指标的字典。
    注意：键名应与 argparse 和 scikit-learn scoring 参数保持一致。
    """
    try:
        # 计算 ROC AUC 分数
        auc = roc_auc_score(y_true, y_prob)
        # 计算 Average Precision 分数
        ap  = average_precision_score(y_true, y_prob)
    except Exception:
        # 如果计算出错（例如标签全是一种类别），则设为 NaN
        auc, ap = np.nan, np.nan
    
    # 返回一个字典，键名与 scikit-learn 的 scoring 字符串对齐
    return {
        "roc_auc": float(auc),                       # 修正: 'auc' -> 'roc_auc'
        "average_precision":  float(ap),             # 修正: 'ap' -> 'average_precision'
        "f1":  float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)), # 修正: 'acc' -> 'accuracy'
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) # 修正: 'bacc' -> 'balanced_accuracy'
    }

def nested_cv_subject_level(df: pd.DataFrame, feat_cols: List[str], out_dir: str,
                            models: Dict[str, Tuple[Pipeline, Dict]], n_splits=5, inner_splits=3, repeats=1,
                            subject_metric="auc", ablation_name="all"):
    """
    外层 StratifiedKFold，内层 GridSearchCV。返回每个模型的外层折测试分数表与最佳平均分模型名。
    """
    ensure_dir(out_dir)
    X = df[feat_cols].values
    y = df["label"].values.astype(int)

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=42)
    all_results = {}  # model -> list of fold dicts

    for model_name, (pipe, grid) in models.items():
        print(f"    Running model: {model_name}...") # 新增: 提示当前运行的模型
        fold_rows = []
        for fold_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=fold_id)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring=subject_metric, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            # 预测
            y_prob = best.predict_proba(X_te)[:,1]
            y_pred = (y_prob >= 0.5).astype(int)
            sc = score_dict(y_te, y_prob, y_pred)
            fold_rows.append({
                "fold_id": fold_id,
                "best_params": json.dumps(gs.best_params_),
                **sc
            })

        df_res = pd.DataFrame(fold_rows)
        df_res.to_csv(os.path.join(out_dir, f"subject_{ablation_name}_{model_name}_outer_scores.csv"), index=False, encoding="utf-8")
        all_results[model_name] = df_res

    # 汇总：按外层平均 subject_metric 选最佳模型
    means = {m: float(v[subject_metric].mean()) for m,v in all_results.items()}
    best_model = max(means, key=means.get)
    pd.DataFrame([{"model": k, "mean_"+subject_metric: v} for k,v in means.items()])\
        .to_csv(os.path.join(out_dir, f"subject_{ablation_name}_model_selection.csv"), index=False, encoding="utf-8")
    return all_results, best_model

def calibrated_eval(df: pd.DataFrame, feat_cols: List[str], out_dir: str,
                    model: Pipeline, subject_metric="auc"):
    """
    在一次 StratifiedKFold 上做概率校准与校准曲线数据（简化版）。
    """
    ensure_dir(out_dir)
    X = df[feat_cols].values
    y = df["label"].values.astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        base = model
        # 再做一个小网格以防失配
        try:
            base.fit(X_tr, y_tr)
        except Exception:
            base.fit(X_tr, y_tr)

        calib = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        calib.fit(X_tr, y_tr)
        prob = calib.predict_proba(X_te)[:,1]
        pred = (prob>=0.5).astype(int)
        sc = score_dict(y_te, prob, pred)
        brier = brier_score_loss(y_te, prob)
        rows.append({"fold": k, **sc, "brier": float(brier)})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "subject_calibration_scores.csv"), index=False, encoding="utf-8")

def plot_calibration_curve_placeholder(out_png: str):
    # 占位：这里不画真实曲线（需要细粒度概率分箱），仅输出空白图避免依赖 seaborn
    plt.figure()
    plt.title("Calibration curve placeholder")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def run_window_level(dfw: pd.DataFrame, feat_cols: List[str], out_dir: str,
                     model_tuple: Tuple[Pipeline, Dict], agg="mean", n_splits=5):
    """
    GroupKFold（按 subject），在每个外层测试折上，先对窗口做预测，再按 subject 聚合为受试者概率，汇总评估。
    """
    ensure_dir(out_dir)
    if "subject" not in dfw.columns:
        print("[WARN] features_window.csv 不含 subject 列，跳过窗口级评估。")
        return

    dfw = dfw.copy()
    # 检查每个 subject 是否有多个窗口
    counts = dfw.groupby("subject").size()
    if counts.max() == 1:
        print("[WARN] 每个 subject 只有一个窗口，窗口级评估没有意义。")
        return

    X = dfw[feat_cols].values
    y = dfw["label"].values.astype(int)
    groups = dfw["subject"].values

    pipe, grid = model_tuple
    outer = GroupKFold(n_splits=n_splits)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr, g_te = groups[tr_idx], groups[te_idx]

        inner = GroupKFold(n_splits=min(3, n_splits-1))
        # 由于 GridSearchCV 没有 group-aware scoring 的简易接口，这里简单用 StratifiedKFold 在训练集内调参
        # 或者直接拟合默认超参
        pipe.fit(X_tr, y_tr)

        prob = pipe.predict_proba(X_te)[:,1]
        df_tmp = pd.DataFrame({"subject": g_te, "y": y_te, "prob": prob})
        # 按 subject 聚合
        if agg == "mean":
            df_sub = df_tmp.groupby("subject", as_index=False).agg({"prob":"mean","y":"first"})
        elif agg == "median":
            df_sub = df_tmp.groupby("subject", as_index=False).agg({"prob":"median","y":"first"})
        elif agg.startswith("top"):
            try:
                k_top = int(agg[3:])
            except:
                k_top = 3
            def topk_mean(x):
                x = np.sort(x)[::-1]
                return float(np.mean(x[:min(k_top, len(x))]))
            df_sub = df_tmp.groupby("subject").agg({"prob": topk_mean, "y":"first"}).reset_index()
        else:
            df_sub = df_tmp.groupby("subject", as_index=False).agg({"prob":"mean","y":"first"})

        y_true = df_sub["y"].values.astype(int)
        y_prob = df_sub["prob"].values
        y_pred = (y_prob>=0.5).astype(int)
        sc = score_dict(y_true, y_prob, y_pred)
        rows.append({"fold": k, "agg": agg, **sc})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"window_subjectagg_{agg}_scores.csv"),
                              index=False, encoding="utf-8")

def select_feature_sets(feat_cols: List[str]) -> Dict[str, List[str]]:
    sets = {}
    sets["all"] = list(feat_cols)
    sets["space_only"] = [c for c in feat_cols if c.startswith("A_") or c.upper().startswith("SASI")]
    sets["non_space"] = [c for c in feat_cols if c not in sets["space_only"]]
    return sets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-subject", type=str, default="raw_subject_features.csv")
    ap.add_argument("--features-window",  type=str, default="raw_window_features.csv")
    ap.add_argument("--models", type=str, default="logreg,svm,rf,gb")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--subject-metric", type=str, default="roc_auc", choices=["roc_auc","average_precision","f1","accuracy","balanced_accuracy"])
    ap.add_argument("--do-window", action="store_true", help="若提供 features_window.csv 且含 subject 列，则做窗口级→受试者聚合评估")
    ap.add_argument("--agg", type=str, default="mean", help="窗口聚合策略：mean/median/topK 例如 top3")
    args = ap.parse_args()

    out_root = "/mnt/data/ml_reports"
    ensure_dir(out_root)

    # === 受试者级 ===
    df_subj, feat_cols_subj = read_features(args.features_subject)
    model_list = [m.strip() for m in args.models.split(",")]
    models = build_models(model_list)

    # 消融集合
    feat_sets = select_feature_sets(feat_cols_subj)
    summary_rows = []
    detailed_summary_rows = [] # 新增: 用于存储所有模型、所有特征集的详细结果

    for set_name, cols in feat_sets.items():
        print(f"\n===== Running Ablation on Feature Set: '{set_name}' =====")

        # --- 这里是关键的修改 ---
        # 在运行嵌套交叉验证之前，检查特征列表 'cols' 是否为空。
        # 如果为空，则打印一条信息并跳过此次循环，避免将0列数据传入模型。
        if not cols:
            print(f"  [INFO] Feature set '{set_name}' is empty. Skipping this ablation run.")
            continue
        # --- 修改结束 ---
        
        res, best_model = nested_cv_subject_level(df_subj, cols, os.path.join(out_root, f"subject_{set_name}"),
                                                  models, n_splits=args.n_splits, inner_splits=args.inner_splits,
                                                  repeats=args.repeats, subject_metric=args.subject_metric, ablation_name=set_name)
        
        # --- 新增: 记录所有模型的详细结果 ---
        for model_name, result_df in res.items():
            mean_auc = float(result_df["roc_auc"].mean())
            mean_f1 = float(result_df["f1"].mean())
            mean_acc = float(result_df["accuracy"].mean())
            mean_bacc = float(result_df["balanced_accuracy"].mean())
            
            detailed_summary_rows.append({
                "feature_set": set_name,
                "model": model_name,
                "auc_mean": f"{mean_auc:.4f}",
                "accuracy_mean": f"{mean_acc:.4f}",
                "f1_mean": f"{mean_f1:.4f}",
                "balanced_accuracy_mean": f"{mean_bacc:.4f}"
            })

        # --- 保留原有逻辑: 记录最佳模型用于后续步骤 ---
        best_model_res = res[best_model]
        mean_auc = float(best_model_res["roc_auc"].mean())
        mean_f1  = float(best_model_res["f1"].mean())
        mean_acc = float(best_model_res["accuracy"].mean())
        summary_rows.append({"feature_set": set_name, "best_model": best_model,
                             "auc_mean": mean_auc, "f1_mean": mean_f1, "acc_mean": mean_acc})

        # 做一次校准与置换重要度（用全部数据拟合一次以取特征重要度；严谨做法应在外层每折内部做并平均，这里简化）
        print(f"  -> Best model for '{set_name}' is '{best_model}'. Running post-hoc analysis...")
        best_pipe, _ = models[best_model]
        calibrated_eval(df_subj, cols, os.path.join(out_root, f"subject_{set_name}_{best_model}_calib"),
                        best_pipe,
                        subject_metric=args.subject_metric)

        # 置换重要度（简化版，单次 fit）
        X = df_subj[cols].values
        y = df_subj["label"].values.astype(int)
        best_pipe.fit(X, y)
        try:
            imp = permutation_importance(best_pipe, X, y, n_repeats=10, random_state=0, scoring="roc_auc")
            order = np.argsort(-imp.importances_mean)
            top = min(20, len(cols))
            rows = []
            for r in range(top):
                j = order[r]
                rows.append({
                    "feature": cols[j],
                    "imp_mean": float(imp.importances_mean[j]),
                    "imp_std":  float(imp.importances_std[j])
                })
            pd.DataFrame(rows).to_csv(os.path.join(out_root, f"subject_{set_name}_{best_model}_perm_importance_top20.csv"),
                                      index=False, encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] Permutation importance failed for '{set_name}': {e}")

    # 保存只含最佳模型的简要总结
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_root, "subject_summary.csv"), index=False, encoding="utf-8")

    # --- 新增: 打印并保存所有模型的详细总结 ---
    df_detailed = pd.DataFrame(detailed_summary_rows)
    df_detailed.to_csv(os.path.join(out_root, "subject_detailed_summary.csv"), index=False, encoding="utf-8")

    print("\n\n" + "="*80)
    print(" " * 20 + "ML Experiment Detailed Summary Report")
    print("="*80)
    # 使用 to_string() 确保所有列都能被打印出来
    print(df_detailed.to_string())
    print("="*80)
    print(f"\nDetailed summary report saved to: {os.path.join(out_root, 'subject_detailed_summary.csv')}")


    # === 窗口级（可选）===
    if args.do_window and os.path.exists(args.features_window):
        print("\n===== Running Window-level Analysis =====")
        df_win, feat_cols_win = read_features(args.features_window)
        # 使用在“全部特征”上受试者级最优模型做窗口级评估
        best_model_on_all_feats = summary_rows[0]["best_model"] if len(summary_rows) > 0 else "logreg"
        print(f"  Using best model from subject-level ('{best_model_on_all_feats}') for window analysis.")
        models_all = build_models([best_model_on_all_feats])
        
        for name, (pipe, grid) in models_all.items():
            run_window_level(df_win, feat_cols_win, os.path.join(out_root, "window_all_"+name),
                             (pipe, grid), agg=args.agg, n_splits=args.n_splits)


if __name__ == "__main__":
    main()