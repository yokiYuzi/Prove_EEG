# channel_select_2a_anova.py
# ------------------------------------------------------------
# Channel Selection for BCICIV-2a (Within-subject, Session T only)
# Scheme A: log bandpower (mu/beta) + one-way ANOVA F-statistic
# Output: channel ranking + Top-K selection + CSV/PNG
# ------------------------------------------------------------

import os
import sys
import json
import argparse
import numpy as np

from sklearn.feature_selection import f_classif

# ---- make sure we can import get_data exactly like your training scripts ----
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
DATALOAD_DIR = os.path.join(CUR_DIR, "dataLoad")
if DATALOAD_DIR not in sys.path and os.path.isdir(DATALOAD_DIR):
    sys.path.append(DATALOAD_DIR)

try:
    # training scripts often use: from dataLoad.preprocess import get_data
    from dataLoad.preprocess import get_data
except Exception:
    # plotting scripts in dataLoad folder use: from preprocess import get_data
    from preprocess import get_data


# 22ch names used in your plot_eeg_60s.py (BCICIV-2a)
CH_NAMES_2A = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


def parse_bands(band_str: str):
    """
    band_str: e.g. "8-12,13-30"
    return: [(8,12),(13,30)]
    """
    bands = []
    for item in band_str.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-")
        bands.append((float(a), float(b)))
    if len(bands) == 0:
        raise ValueError("No valid bands parsed. Example: --bands 8-12,13-30")
    return bands


def extract_log_bandpower_features(
    X: np.ndarray,
    fs: int,
    win_sec: float = 1.0,
    bands=((8.0, 12.0), (13.0, 30.0)),
    nfft: int = 256,
    eps: float = 1e-12,
):
    """
    X: [N_trial, N_ch, T]
    Return:
      feat: [N_trial, N_ch, (n_win * n_bands)]
    Implementation uses vectorized rFFT for speed (no per-trial Welch loops).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be [trial, ch, time], got {X.shape}")

    N, C, T = X.shape
    win_len = int(round(win_sec * fs))
    if win_len <= 0:
        raise ValueError(f"win_sec too small: {win_sec}")
    n_win = T // win_len
    if n_win <= 0:
        raise ValueError(f"Time length T={T} too short for win_len={win_len}")

    # truncate to full windows
    T_use = n_win * win_len
    X_use = X[..., :T_use]  # [N,C,T_use]
    Xw = X_use.reshape(N, C, n_win, win_len)  # [N,C,W,L]

    # apply Hann window
    hann = np.hanning(win_len).astype(np.float32)  # [L]
    Xw = Xw * hann  # broadcast

    # rFFT
    fft = np.fft.rfft(Xw, n=nfft, axis=-1)  # [N,C,W,F]
    psd = (np.abs(fft) ** 2).astype(np.float32)  # proportional to power

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)  # [F]

    feats = []
    for (fmin, fmax) in bands:
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size == 0:
            raise ValueError(f"Band {fmin}-{fmax}Hz has no FFT bins. Try larger nfft.")
        bp = psd[..., idx].sum(axis=-1)  # [N,C,W]
        feats.append(np.log(bp + eps))   # [N,C,W]

    feat = np.stack(feats, axis=-1)          # [N,C,W,B]
    feat = feat.reshape(N, C, n_win * len(bands))  # [N,C,W*B]
    return feat


def channel_scores_anova(feat: np.ndarray, y: np.ndarray):
    """
    feat: [N, C, D]  (D = n_win*n_bands)
    y: [N] int labels in {0,1,2,3}
    return:
      scores: [C] channel score = mean_j(F_j)
      F_all : [C, D] F-stat for analysis/debug
    """
    if y.ndim != 1:
        y = y.reshape(-1)
    N, C, D = feat.shape
    scores = np.zeros(C, dtype=np.float32)
    F_all = np.zeros((C, D), dtype=np.float32)

    for ch in range(C):
        F, _p = f_classif(feat[:, ch, :], y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        F_all[ch] = F
        scores[ch] = float(np.mean(F))
    return scores, F_all


def select_topk_with_corr(
    scores: np.ndarray,
    feat: np.ndarray,
    k: int,
    corr_thr: float = 0.90,
):
    """
    Greedy selection by score with redundancy control:
    - sort by score desc
    - add channel if correlation (flattened features) with any selected < corr_thr
    If cannot fill k due to corr constraint, fallback to fill remaining by score.
    """
    C = scores.shape[0]
    order = np.argsort(scores)[::-1]

    selected = []
    selected_set = set()

    def flat_vec(ch_idx: int):
        return feat[:, ch_idx, :].reshape(-1)

    for ch in order:
        if len(selected) >= k:
            break
        if len(selected) == 0:
            selected.append(int(ch))
            selected_set.add(int(ch))
            continue

        v = flat_vec(int(ch))
        ok = True
        for s in selected:
            vs = flat_vec(int(s))
            # handle zero-variance cases
            if np.std(v) < 1e-8 or np.std(vs) < 1e-8:
                continue
            corr = np.corrcoef(v, vs)[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) >= corr_thr:
                ok = False
                break
        if ok:
            selected.append(int(ch))
            selected_set.add(int(ch))

    # fallback fill
    if len(selected) < k:
        for ch in order:
            ch = int(ch)
            if ch not in selected_set:
                selected.append(ch)
                selected_set.add(ch)
            if len(selected) >= k:
                break

    return np.array(selected, dtype=np.int64)


def get_ch_names(n_ch: int, data_type: str):
    if data_type == "2a" and n_ch == 22:
        return CH_NAMES_2A
    return [f"Ch{idx+1}" for idx in range(n_ch)]


def save_rank_csv(save_path: str, ch_names, scores):
    import csv
    order = np.argsort(scores)[::-1]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "channel_index", "channel_name", "score"])
        for r, ch in enumerate(order, start=1):
            w.writerow([r, int(ch), ch_names[int(ch)], float(scores[int(ch)])])


def plot_scores_bar(save_path: str, ch_names, scores, topn: int = 22):
    import matplotlib.pyplot as plt

    order = np.argsort(scores)[::-1][:topn]
    names = [ch_names[int(i)] for i in order]
    vals = [scores[int(i)] for i in order]

    plt.figure(figsize=(max(10, topn * 0.45), 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=60, ha="right")
    plt.ylabel("ANOVA F score (mean over features)")
    plt.title(f"Top-{topn} channel discriminability (log bandpower + ANOVA)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_one_subject(args, subject: int):
    # 1) Load data (Session T for training, Session E for testing)
    # IMPORTANT: channel selection uses Session T only to avoid leakage.
    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=args.data_dir,
        subject=subject,
        LOSO=False,
        Transfer=False,
        onLine_2a=False,
        data_model="one_session",
        isStandard=args.is_standard,   # default False for bandpower-based selection
        data_type="2a",
    )

    if X_train is None or len(X_train) == 0:
        raise RuntimeError(f"[S{subject}] X_train is empty. Check --data_dir and subject.")
    if X_train.ndim != 3:
        raise RuntimeError(f"[S{subject}] X_train shape must be [trial,ch,time], got {X_train.shape}")

    y_train = np.asarray(y_train).reshape(-1).astype(int)
    N, C, T = X_train.shape

    # 2) Infer fs from your pipeline:
    # In preprocess.py, fs=250 and crop 2-6s => 4 seconds => T should be 1000 typically.
    # We keep it robust:
    if args.fs is not None:
        fs = int(args.fs)
    else:
        # assume 4-second MI window after crop (2-6s)
        fs = int(round(T / 4.0))

    ch_names = get_ch_names(C, "2a")

    # 3) Feature extraction: log bandpower per channel, per window, per band
    bands = parse_bands(args.bands)
    feat = extract_log_bandpower_features(
        X_train.astype(np.float32),
        fs=fs,
        win_sec=args.win_sec,
        bands=bands,
        nfft=args.nfft,
        eps=args.eps,
    )  # [N,C,D]

    # 4) ANOVA scoring
    scores, F_all = channel_scores_anova(feat, y_train)

    # 5) select top-k (+ optional redundancy control)
    if args.corr_thr is not None and args.corr_thr > 0:
        selected_idx = select_topk_with_corr(scores, feat, k=args.k, corr_thr=args.corr_thr)
    else:
        order = np.argsort(scores)[::-1]
        selected_idx = order[: args.k].astype(np.int64)

    selected_names = [ch_names[int(i)] for i in selected_idx]

    # 6) Save results
    sub_dir = os.path.join(args.save_dir, f"sub{subject:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    save_rank_csv(os.path.join(sub_dir, "channel_rank.csv"), ch_names, scores)

    if args.plot:
        plot_scores_bar(os.path.join(sub_dir, "channel_scores_top.png"), ch_names, scores, topn=min(args.topn, C))

    out_json = {
        "subject": subject,
        "fs_inferred": fs,
        "X_train_shape": [int(N), int(C), int(T)],
        "bands": bands,
        "win_sec": args.win_sec,
        "nfft": args.nfft,
        "is_standard": bool(args.is_standard),
        "k": int(args.k),
        "corr_thr": None if args.corr_thr is None else float(args.corr_thr),
        "selected_idx": [int(i) for i in selected_idx],
        "selected_names": selected_names,
        "scores": {ch_names[i]: float(scores[i]) for i in range(C)},
    }
    with open(os.path.join(sub_dir, "selected_channels.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # 7) Print summary
    print("=" * 80)
    print(f"[Subject {subject}] X_train={X_train.shape}, fs≈{fs}Hz, bands={bands}, win_sec={args.win_sec}")
    print(f"Top-{args.k} selected channels (idx -> name):")
    for i in selected_idx:
        print(f"  {int(i):2d} -> {ch_names[int(i)]}   (score={scores[int(i)]:.4f})")
    print(f"Saved to: {sub_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="BCICIV_2a root dir, e.g. /path/to/dataLoad/BCICIV_2a/ (must end with /)")
    parser.add_argument("--subject", type=str, default="1",
                        help="1..9 or 'all'")
    parser.add_argument("--k", type=int, default=8, help="Top-K channels to select")
    parser.add_argument("--bands", type=str, default="8-12,13-30",
                        help="Frequency bands, e.g. '8-12,13-30'")
    parser.add_argument("--win_sec", type=float, default=1.0, help="Window length in seconds (default 1.0 => 4 windows)")
    parser.add_argument("--fs", type=int, default=None, help="Force sampling rate. If None, infer from T/4.")
    parser.add_argument("--nfft", type=int, default=256, help="FFT size for bandpower (default 256)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for log()")
    parser.add_argument("--is_standard", action="store_true",
                        help="If set, use StandardScaler in get_data (default False recommended for bandpower)")
    parser.add_argument("--corr_thr", type=float, default=0.90,
                        help="Redundancy control threshold. Set <=0 to disable. Default 0.90.")
    parser.add_argument("--save_dir", type=str, default="channel_selection_results",
                        help="Output dir")
    parser.add_argument("--plot", action="store_true", help="Save bar plot PNG")
    parser.add_argument("--topn", type=int, default=22, help="How many channels to show in plot")

    args = parser.parse_args()

    if not args.data_dir.endswith("/") and not args.data_dir.endswith("\\"):
        # keep consistent with your preprocess path concatenation style
        args.data_dir = args.data_dir + "/"

    if args.subject.lower() == "all":
        for s in range(1, 10):
            run_one_subject(args, s)
    else:
        s = int(args.subject)
        run_one_subject(args, s)


if __name__ == "__main__":
    main()
