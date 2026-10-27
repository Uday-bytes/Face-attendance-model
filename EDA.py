#!/usr/bin/env python3
"""
EDA.py - Exploratory Data Analysis for the Face Attendance project
====================================================================

Project requirement (Image Processing: Face Attendance)
    "EDA: image counts per person, lighting variation and failed detections."
    "Metrics: recognition accuracy, false accepted, false rejected."
    "Test under different light, distance and angle."

This script turns those requirements into concrete analyses, using the plot
types from the Data Visualization lecture (bar, histogram with Freedman-Diaconis
bins, KDE/ECDF, box/violin, heatmap incl. the "missing value" heatmap,
correlation matrix, pairplot, line plot).

ANALYSIS PLAN
-------------
 Question                                   Method (lecture topic)                       Output file
 -----------------------------------------  -------------------------------------------  ----------------------------------
 How many images does each person have?    Bar plot, bar of count categories            01_images_per_person.png
 Is the data balanced?                      Histogram (integer bins) + ECDF              02_count_distribution.png
 Are image sizes consistent?                Hexbin / joint plot (only if sizes differ)   03_image_sizes.png
 How much does lighting/quality vary?       Histogram (FD bins) + KDE, threshold lines   04_lighting_quality_distributions.png
 Does it vary within one person?            Box plots per person (compare + outliers)    05_variation_within_person.png
 How often does detection fail, and why?   Bar plot of outcomes, rate by lighting bin   06_detection_overview.png
 Which images fail?                         Missing-value style heatmap (person x img)   07_detection_heatmap.png
 Do failures relate to image quality?       Box plots: detected vs not detected          08_failure_vs_quality.png
 Distance / angle of detected faces         Histograms (face size, score, |yaw|)         09_face_geometry.png
 Which measurements move together?          Correlation matrix heatmap                   10_correlation_heatmap.png
 Pairwise structure, split by outcome       Pairplot with hue                            11_pairplot.png
 Same person vs different person scores     Overlaid histogram + KDE                     12_similarity_distributions.png
 Best recognition threshold, FAR / FRR      Line plot (FAR and FRR vs threshold)         13_far_frr_curve.png
 Which people look alike to the model?      Heatmap of similarity between people         14_person_similarity_heatmap.png

PRIVACY / CONSENT RULES (please keep)
-------------------------------------
 * Only use images of people who have given consent, or a public dataset such as LFW.
 * Everything runs locally. No image is uploaded or copied; no face crops are saved.
 * Use --anonymize to replace names with P001, P002... in plots, the CSV and the report.
 * Do NOT commit the photos (registered_faces/, unknown_faces/, the dataset) to GitHub.
   Add them to .gitignore. Share only anonymized outputs of this script.

USAGE
-----
   python EDA.py --data path/to/lfw-deepfunneled --min-images 5 --anonymize
   python EDA.py --data registered_faces --anonymize
   python EDA.py --data dataset --no-detect            # image counts + lighting only (fast)
   python EDA.py --data dataset --detector haar        # without InsightFace

Folder layouts understood:
   dataset/<Person>/<image>.jpg      (LFW and most datasets: person = folder name)
   dataset/<Person>_<number>.jpg     (flat folder: person = file name without trailing number)

Requirements: numpy, pandas, matplotlib, seaborn, opencv-python  (+ insightface, onnxruntime
for detection scores, embeddings and the FAR/FRR analysis).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

import matplotlib

if "--show" not in sys.argv:          # no window needed when we only save figures
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="colorblind")

# ----------------------------------------------------------------------------
# Settings (same values as the attendance app where they exist)
# ----------------------------------------------------------------------------
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
APP_THRESHOLD = 0.45      # THRESHOLD used in face_attendance_gui.py
MIN_DET_SCORE = 0.6       # MIN_DET_SCORE used in face_attendance_gui.py

# Heuristic limits used only to *label* images in the EDA. Adjust after looking at the plots.
DARK_T = 60               # mean gray level below this  -> "dark"
BRIGHT_T = 190            # mean gray level above this  -> "bright"
LOW_CONTRAST_T = 35       # std of gray levels below this -> "low contrast"
BLUR_T = 100              # variance of Laplacian below this -> "blurry" (depends on resolution)
SMALL_FACE_T = 0.05       # face area / image area below this -> "small / far away"

OUT_OK = "detected (1 face)"
OUT_MULTI = "detected (multiple faces)"
OUT_LOW = "low confidence"
OUT_NONE = "no face"
OUT_BAD = "unreadable"
OUTCOME_ORDER = [OUT_OK, OUT_MULTI, OUT_LOW, OUT_NONE, OUT_BAD]
OUTCOME_COLORS = {OUT_OK: "#2ca02c", OUT_MULTI: "#98df8a", OUT_LOW: "#ff7f0e",
                  OUT_NONE: "#d62728", OUT_BAD: "#7f7f7f"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def fd_bins(x):
    """Freedman-Diaconis bin edges (lecture): robust to outliers. Falls back safely."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.ptp(x) == 0:
        return 10
    edges = np.histogram_bin_edges(x, bins="fd")
    if len(edges) > 201:                       # FD can explode on heavy tails
        edges = np.histogram_bin_edges(x, bins=200)
    return edges


def iqr_outliers(x):
    """Box-plot rule: points beyond 1.5 * IQR. Returns (count, lower, upper)."""
    x = pd.Series(x).dropna()
    if x.empty:
        return 0, np.nan, np.nan
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((x < lo) | (x > hi)).sum()), lo, hi


def save(fig, outdir, name, show=False):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  saved {path}")
    return path


def safe(step_name, fn, *args, **kwargs):
    """Run one analysis step; a failing plot must not stop the whole EDA."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"  [skipped] {step_name}: {type(e).__name__}: {e}")
        if os.environ.get("EDA_DEBUG"):
            traceback.print_exc()
        plt.close("all")
        return None


# ----------------------------------------------------------------------------
# 1. Inventory: which images exist and who do they belong to?
# ----------------------------------------------------------------------------
def person_from_path(path, root):
    parent = os.path.dirname(path)
    if os.path.abspath(parent) != os.path.abspath(root):
        return os.path.basename(parent)
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_\-\s]*\d+$", "", stem) or stem


def build_inventory(root):
    rows = []
    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                p = os.path.join(folder, f)
                rows.append((p, person_from_path(p, root)))
    inv = pd.DataFrame(rows, columns=["path", "person"])
    return inv.sort_values("path").reset_index(drop=True)


def anonymize_inventory(inv):
    names = sorted(inv["person"].unique())
    mapping = {n: f"P{i + 1:03d}" for i, n in enumerate(names)}
    inv = inv.copy()
    inv["person"] = inv["person"].map(mapping)
    return inv


def select_subset(inv, min_images, max_images, per_person_cap, seed):
    """
    Detection is slow on a CPU, so analyse a subset. Whole people are selected (not random
    single images) so that every selected person keeps several images - this is needed for
    same-person comparisons later.
    """
    rng = np.random.default_rng(seed)
    groups = inv.groupby("person").indices            # person -> row positions
    eligible = [p for p, idx in groups.items() if len(idx) >= min_images]
    if not eligible:
        print(f"  No person has >= {min_images} images; using all images instead.")
        eligible = list(groups)
    rng.shuffle(eligible)

    picked, total = [], 0
    for p in eligible:
        idx = np.asarray(groups[p])
        if max_images > 0 and per_person_cap > 0 and len(idx) > per_person_cap:
            idx = rng.choice(idx, per_person_cap, replace=False)
        picked.extend(idx.tolist())
        total += len(idx)
        if max_images > 0 and total >= max_images:
            break
    if max_images > 0:
        picked = picked[:max_images]
    return inv.iloc[picked].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Per-image measurements: size, lighting, sharpness, detection, embedding
# ----------------------------------------------------------------------------
class FaceDetector:
    """InsightFace (same model as the app) with an OpenCV Haar fallback."""

    def __init__(self, kind="insightface", min_score=MIN_DET_SCORE):
        self.kind = kind
        self.min_score = min_score
        self.app = None
        self.haar = None
        if self.kind == "insightface":
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:  # noqa: BLE001
                print(f"  InsightFace not available ({type(e).__name__}); using OpenCV Haar cascade."
                      "\n  (No detection scores / embeddings, so FAR/FRR analysis will be skipped.)")
                self.kind = "haar"
        if self.kind == "haar":
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def analyse(self, img):
        h, w = img.shape[:2]
        res = dict(outcome=OUT_NONE, n_faces=0, det_score=np.nan,
                   face_area_ratio=np.nan, yaw_abs=np.nan, embedding=None)

        if self.kind == "insightface":
            faces = self.app.get(img)
            res["n_faces"] = len(faces)
            if not faces:
                return res
            good = [f for f in faces if f.det_score >= self.min_score]
            best = max(good or faces, key=lambda f: f.det_score if not good else
                       (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = best.bbox
            res["det_score"] = float(best.det_score)
            res["face_area_ratio"] = float(max(0, x2 - x1) * max(0, y2 - y1) / (w * h))
            pose = getattr(best, "pose", None)
            if pose is not None:
                res["yaw_abs"] = float(abs(pose[1]))
            if not good:
                res["outcome"] = OUT_LOW
                return res
            res["n_faces"] = len(good)
            res["outcome"] = OUT_OK if len(good) == 1 else OUT_MULTI
            res["embedding"] = best.normed_embedding
            return res

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        res["n_faces"] = len(rects)
        if len(rects) == 0:
            return res
        x, y, fw, fh = max(rects, key=lambda r: r[2] * r[3])
        res["face_area_ratio"] = float(fw * fh / (w * h))
        res["outcome"] = OUT_OK if len(rects) == 1 else OUT_MULTI
        return res


def measure_images(subset, data_root, detector, anonymize):
    rows, embs = [], []
    n = len(subset)
    for i, rec in enumerate(subset.itertuples(index=False)):
        if i % 50 == 0:
            print(f"  analysing image {i + 1}/{n}")
        row = dict(image_id=i, person=rec.person,
                   width=np.nan, height=np.nan, brightness=np.nan, contrast=np.nan,
                   sharpness=np.nan, file_hash=None, outcome=OUT_BAD, n_faces=0,
                   det_score=np.nan, face_area_ratio=np.nan, yaw_abs=np.nan)
        if not anonymize:
            row["file"] = os.path.relpath(rec.path, data_root)

        emb = None
        try:
            raw = np.fromfile(rec.path, dtype=np.uint8)      # also works for non-ASCII paths
            row["file_hash"] = hashlib.md5(raw.tobytes()).hexdigest()
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            row["height"], row["width"] = img.shape[:2]
            row["brightness"] = float(gray.mean())
            row["contrast"] = float(gray.std())
            row["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if detector is not None:
                r = detector.analyse(img)
                emb = r.pop("embedding")
                row.update(r)
            else:
                row["outcome"] = "not run"
        rows.append(row)
        embs.append(emb)

    df = pd.DataFrame(rows)
    df["sharpness_log"] = np.log10(1 + df["sharpness"])
    df["light_bin"] = pd.cut(df["brightness"], [-np.inf, DARK_T, BRIGHT_T, np.inf],
                             labels=["dark", "normal", "bright"])
    df["detected"] = df["outcome"].isin([OUT_OK, OUT_MULTI])
    df["detected_label"] = np.where(df["detected"], "detected", "not detected")
    return df, embs


# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------
def plot_counts(inv, out, show):
    """Bar plot: images per person (+ how many people reach 5-10 images)."""
    counts = inv["person"].value_counts()
    top = counts.head(20)[::-1]
    cats = ["1", "2-4", "5-10", "11+"]
    grouped = pd.cut(counts, [0, 1, 4, 10, np.inf], labels=cats).value_counts().reindex(cats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})
    axes[0].barh(top.index.astype(str), top.values, color=sns.color_palette()[0])
    axes[0].set(xlabel="Number of images", title=f"Top {len(top)} people by number of images")
    sns.barplot(x=cats, y=grouped.values, hue=cats, legend=False, ax=axes[1])
    for i, v in enumerate(grouped.values):
        axes[1].text(i, v, str(int(v)), ha="center", va="bottom")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set(xlabel="Images per person", ylabel="Number of people",
                title="Do people have enough images? (target: 5-10)")
    save(fig, out, "01_images_per_person.png", show)


def plot_count_distribution(inv, out, show):
    """Histogram + ECDF of images per person (is the data balanced?)."""
    counts = inv["person"].value_counts()
    clipped = counts.clip(upper=30)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # Counts are whole numbers with a very heavy tail, so integer bins are clearer than FD bins here.
    sns.histplot(clipped, bins=np.arange(0.5, 31.5, 1), ax=axes[0])
    axes[0].set(yscale="log", xlabel="Images per person (30 = 30 or more)",
                ylabel="Number of people (log scale)", title="Distribution of images per person")
    sns.ecdfplot(counts, ax=axes[1], log_scale=True)
    axes[1].axvline(5, color="red", ls="--", label="5 images")
    axes[1].legend()
    axes[1].set(xlabel="Images per person (log scale)", ylabel="Fraction of people",
                title="ECDF: share of people with at most N images")
    save(fig, out, "02_count_distribution.png", show)


def plot_image_sizes(df, out, show):
    """Joint plot of width vs height - only useful when sizes differ."""
    sizes = df.dropna(subset=["width", "height"])
    if sizes[["width", "height"]].drop_duplicates().shape[0] < 3:
        print("  image sizes are (almost) identical -> size plot skipped")
        return
    g = sns.jointplot(data=sizes, x="width", y="height", kind="hex")
    g.figure.suptitle("Image size (hexbin)", y=1.02)
    save(g.figure, out, "03_image_sizes.png", show)


def plot_lighting(df, out, show):
    """Histograms (FD bins) + KDE of brightness, contrast, sharpness with heuristic limits."""
    spec = [("brightness", "Brightness (mean gray level)", [DARK_T, BRIGHT_T]),
            ("contrast", "Contrast (std of gray levels)", [LOW_CONTRAST_T]),
            ("sharpness_log", "Sharpness: log10(1 + variance of Laplacian)", [np.log10(1 + BLUR_T)])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (col, title, lines) in zip(axes, spec):
        x = df[col].dropna()
        sns.histplot(x, bins=fd_bins(x), kde=x.nunique() > 2, ax=ax)
        for v in lines:
            ax.axvline(v, color="red", ls="--")
        ax.set(title=title, xlabel="")
    fig.suptitle("Lighting and quality variation (red dashed = heuristic limits)", y=1.02)
    save(fig, out, "04_lighting_quality_distributions.png", show)


def plot_variation(df, out, show, top=10):
    """Box plots per person: does lighting/sharpness change a lot for the same person?"""
    counts = df["person"].value_counts()
    order = list(counts[counts >= 3].head(top).index)
    if len(order) < 2:
        print("  fewer than 2 people with >= 3 analysed images -> within-person plot skipped")
        return
    sub = df[df["person"].isin(order)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col, title in zip(axes, ["brightness", "sharpness_log"],
                              ["Brightness per person", "Sharpness per person (log10)"]):
        sns.boxplot(data=sub, x="person", y=col, order=order, hue="person",
                    legend=False, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set(title=title, xlabel="")
    save(fig, out, "05_variation_within_person.png", show)


def plot_detection_overview(df, out, show):
    """Bar plot of detection outcomes + detection rate by lighting bin."""
    counts = df["outcome"].value_counts().reindex(OUTCOME_ORDER).dropna().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(counts.index, counts.values, color=[OUTCOME_COLORS[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="Number of images", title="Face detection outcomes")

    rate = (df.dropna(subset=["light_bin"]).groupby("light_bin", observed=True)["detected"]
            .agg(["mean", "size"]))
    rate = rate[rate["size"] > 0]
    sns.barplot(x=rate.index.astype(str), y=rate["mean"] * 100, hue=rate.index.astype(str),
                legend=False, ax=axes[1])
    for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
        axes[1].text(i, m * 100, f"{m * 100:.1f}%\n(n={n})", ha="center", va="bottom")
    axes[1].set(ylim=(0, 115), xlabel="Lighting category", ylabel="Detection rate (%)",
                title="Detection rate by lighting")
    save(fig, out, "06_detection_overview.png", show)


def plot_detection_heatmap(df, out, show, max_people=25, max_cols=30):
    """Missing-value style heatmap: rows = people, columns = their images, colour = detected?"""
    top = df["person"].value_counts().head(max_people).index
    rows = []
    for p in top:
        v = df.loc[df["person"] == p, "detected"].astype(float).to_numpy()[:max_cols]
        r = np.full(max_cols, np.nan)
        r[:len(v)] = v
        rows.append(r)
    m = np.array(rows)
    m = m[:, ~np.all(np.isnan(m), axis=0)]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(top) + 2)))
    sns.heatmap(m, mask=np.isnan(m), cmap=ListedColormap(["#d62728", "#2ca02c"]), vmin=0, vmax=1,
                cbar=False, linewidths=0.4, linecolor="white", yticklabels=list(top),
                xticklabels=False, ax=ax)
    ax.grid(False)
    ax.set(xlabel="Images of the person",
           title="Detection map: green = face detected, red = not detected, blank = no image")
    save(fig, out, "07_detection_heatmap.png", show)


def plot_failure_vs_quality(df, out, show):
    """Box plots: are failed detections darker / blurrier / lower contrast?"""
    d = df[df["outcome"] != OUT_BAD]
    if d["detected"].nunique() < 2:
        print("  no failed (or no successful) detections -> failure-vs-quality plot skipped")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(axes, ["brightness", "contrast", "sharpness_log"],
                              ["Brightness", "Contrast", "Sharpness (log10)"]):
        sns.boxplot(data=d, x="detected_label", y=col, hue="detected_label", legend=False,
                    order=["detected", "not detected"], ax=ax)
        ax.set(title=title, xlabel="")
    fig.suptitle("Do failed detections have different image quality?", y=1.02)
    save(fig, out, "08_failure_vs_quality.png", show)


def plot_geometry(df, out, show):
    """Distance (face size), detector confidence and angle (yaw) of detected faces."""
    d = df[df["detected"]]
    spec = [("face_area_ratio", "Face size / image size (distance proxy)", SMALL_FACE_T),
            ("det_score", "Detector confidence", MIN_DET_SCORE),
            ("yaw_abs", "|Yaw| in degrees (angle proxy)", None)]
    spec = [s for s in spec if d[s[0]].notna().sum() >= 5]
    if not spec:
        print("  no face-geometry data -> plot skipped")
        return
    fig, axes = plt.subplots(1, len(spec), figsize=(5.5 * len(spec), 4.6), squeeze=False)
    for ax, (col, title, line) in zip(axes[0], spec):
        x = d[col].dropna()
        sns.histplot(x, bins=fd_bins(x), kde=x.nunique() > 2, ax=ax)
        if line is not None:
            ax.axvline(line, color="red", ls="--")
        ax.set(title=title, xlabel="")
    save(fig, out, "09_face_geometry.png", show)


def plot_correlation(df, out, show):
    """Correlation matrix heatmap (Pearson). Remember: correlation is not causation."""
    cols = [c for c in ["brightness", "contrast", "sharpness_log", "width", "height",
                        "face_area_ratio", "det_score", "yaw_abs"]
            if df[c].notna().sum() >= 10 and df[c].std() > 0]
    if len(cols) < 3:
        print("  not enough varying numeric columns -> correlation plot skipped")
        return None
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation matrix (Pearson)")
    save(fig, out, "10_correlation_heatmap.png", show)
    return corr


def plot_pairplot(df, out, show, max_points=500, seed=0):
    """Pairplot of the quality measures, coloured by detection outcome."""
    cols = [c for c in ["brightness", "contrast", "sharpness_log"] if df[c].notna().sum() >= 10]
    if len(cols) < 2:
        return
    sample = df.dropna(subset=cols)
    sample = sample.sample(min(max_points, len(sample)), random_state=seed)
    hue = "detected_label" if sample["detected_label"].nunique() > 1 else None
    g = sns.pairplot(sample, vars=cols, hue=hue, diag_kind="hist", corner=True,
                     plot_kws=dict(alpha=0.5, s=18))
    g.figure.suptitle("Pairplot of quality measures", y=1.02)
    save(g.figure, out, "11_pairplot.png", show)


# ----------------------------------------------------------------------------
# 4. Recognition analysis: same person vs different person (FAR / FRR)
# ----------------------------------------------------------------------------
def similarity_analysis(emb, persons, max_pairs_per_person=30, n_impostor=20000, seed=0):
    """
    emb: (n, d) L2-normalised embeddings, persons: (n,) labels.
    genuine  = similarity between two images of the SAME person
    impostor = similarity between images of DIFFERENT people
    The number of genuine pairs is capped per person so one person with hundreds of images
    does not dominate.
    """
    rng = np.random.default_rng(seed)
    persons = np.asarray(persons)
    genuine = []
    for p in np.unique(persons):
        idx = np.flatnonzero(persons == p)
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        if len(a) > max_pairs_per_person:
            sel = rng.choice(len(a), max_pairs_per_person, replace=False)
            a, b = a[sel], b[sel]
        genuine.append(np.sum(emb[idx[a]] * emb[idx[b]], axis=1))
    genuine = np.concatenate(genuine) if genuine else np.array([])

    n = len(emb)
    i = rng.integers(0, n, size=n_impostor * 3)
    j = rng.integers(0, n, size=n_impostor * 3)
    keep = persons[i] != persons[j]
    i, j = i[keep][:n_impostor], j[keep][:n_impostor]
    impostor = np.sum(emb[i] * emb[j], axis=1)
    return genuine, impostor


def threshold_table(genuine, impostor):
    """
    A face is accepted when similarity >= threshold (same rule as the app).
    FRR (false reject)  = genuine pairs below the threshold
    FAR (false accept)  = impostor pairs at/above the threshold
    """
    ts = np.round(np.arange(0, 1.0001, 0.01), 2)
    frr = np.array([(genuine < t).mean() for t in ts])
    far = np.array([(impostor >= t).mean() for t in ts])
    return pd.DataFrame({"threshold": ts, "FAR": far, "FRR": frr,
                         "balanced_accuracy": 1 - (far + frr) / 2})


def plot_similarity(genuine, impostor, table, out, show):
    eer_t = table.loc[(table["FAR"] - table["FRR"]).abs().idxmin(), "threshold"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(impostor, bins=60, stat="density", kde=True, color="#d62728", alpha=0.35,
                 label=f"different people (n={len(impostor)})", ax=ax)
    sns.histplot(genuine, bins=60, stat="density", kde=True, color="#2ca02c", alpha=0.35,
                 label=f"same person (n={len(genuine)})", ax=ax)
    ax.axvline(APP_THRESHOLD, color="black", ls="--", label=f"app threshold {APP_THRESHOLD}")
    ax.axvline(eer_t, color="blue", ls=":", label=f"equal-error threshold {eer_t:.2f}")
    ax.set(xlabel="Cosine similarity", title="Similarity of same-person vs different-person pairs")
    ax.legend()
    save(fig, out, "12_similarity_distributions.png", show)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table["threshold"], table["FAR"] * 100, label="FAR - strangers accepted (%)", color="#d62728")
    ax.plot(table["threshold"], table["FRR"] * 100, label="FRR - real students rejected (%)", color="#2ca02c")
    ax.axvline(APP_THRESHOLD, color="black", ls="--", label=f"app threshold {APP_THRESHOLD}")
    ax.axvline(eer_t, color="blue", ls=":", label=f"equal-error threshold {eer_t:.2f}")
    ax.set(xlabel="Threshold", ylabel="Error rate (%)", title="False accept vs false reject rate")
    ax.legend()
    save(fig, out, "13_far_frr_curve.png", show)
    return eer_t


def plot_person_similarity(emb, persons, out, show, max_people=20):
    persons = np.asarray(persons)
    counts = pd.Series(persons).value_counts()
    chosen = list(counts[counts >= 2].head(max_people).index)
    if len(chosen) < 3:
        return
    means = []
    for p in chosen:
        m = emb[persons == p].mean(axis=0)
        means.append(m / np.linalg.norm(m))
    means = np.array(means)
    sim = means @ means.T
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(sim, xticklabels=chosen, yticklabels=chosen, cmap="viridis", vmin=0, vmax=1,
                annot=len(chosen) <= 10, fmt=".2f", ax=ax)
    ax.set_title("Similarity between people (mean face embedding)\n"
                 "bright off-diagonal cells = people the model may confuse")
    save(fig, out, "14_person_similarity_heatmap.png", show)


# ----------------------------------------------------------------------------
# 5. Report
# ----------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        print(text)
        self.lines.append(text)

    def head(self, text):
        self.add("")
        self.add(text)
        self.add("-" * len(text))

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def describe_dataset(rep, inv):
    c = inv["person"].value_counts()
    rep.head("1. Dataset inventory (all images)")
    rep.add(f"Images: {len(inv)}   People: {len(c)}")
    rep.add(f"Images per person: min {c.min()}, median {c.median():.0f}, mean {c.mean():.1f}, max {c.max()}")
    rep.add(f"People with exactly 1 image: {(c == 1).sum()} ({100 * (c == 1).mean():.1f}%)")
    rep.add(f"People with >= 5 images:     {(c >= 5).sum()} ({100 * (c >= 5).mean():.1f}%)")
    rep.add(f"People with 5-10 images:     {((c >= 5) & (c <= 10)).sum()}")
    return c


def describe_quality(rep, df):
    rep.head("2. Image quality and lighting (analysed images)")
    ok = df[df["outcome"] != OUT_BAD]
    rep.add(f"Analysed images: {len(df)}   unreadable/corrupt: {(df['outcome'] == OUT_BAD).sum()}")
    sizes = ok[["width", "height"]].drop_duplicates()
    if len(sizes) == 1:
        rep.add(f"All images are {int(sizes.iloc[0]['width'])}x{int(sizes.iloc[0]['height'])} px")
    else:
        rep.add(f"Image sizes vary: width {int(ok['width'].min())}-{int(ok['width'].max())}, "
                f"height {int(ok['height'].min())}-{int(ok['height'].max())}")
    dup = df["file_hash"].notna() & df["file_hash"].duplicated(keep=False)
    rep.add(f"Exact duplicate images: {int(dup.sum())}")
    stats = ok[["brightness", "contrast", "sharpness"]].describe().loc[["mean", "std", "min", "50%", "max"]]
    rep.add(stats.round(1).to_string())
    n = len(ok)
    shares = {"dark": (ok["brightness"] < DARK_T).mean(), "bright": (ok["brightness"] > BRIGHT_T).mean(),
              "low contrast": (ok["contrast"] < LOW_CONTRAST_T).mean(), "blurry": (ok["sharpness"] < BLUR_T).mean()}
    for k, v in shares.items():
        rep.add(f"  {k:<13}: {100 * v:5.1f}% of images")
    cnt, lo, hi = iqr_outliers(ok["brightness"])
    rep.add(f"Brightness outliers by box-plot rule (1.5 x IQR): {cnt} images (outside {max(lo, 0):.0f}-{min(hi, 255):.0f})")
    within = ok.groupby("person")["brightness"].std().mean()
    rep.add(f"Average brightness spread inside one person: {within:.1f} (overall spread {ok['brightness'].std():.1f})")
    return n, shares


def describe_detection(rep, df):
    rep.head("3. Face detection")
    for k in OUTCOME_ORDER:
        c = int((df["outcome"] == k).sum())
        if c:
            rep.add(f"  {k:<28}: {c:5d}  ({100 * c / len(df):.1f}%)")
    rate = df["detected"].mean()
    rep.add(f"Overall detection rate: {100 * rate:.1f}%   (failed: {100 * (1 - rate):.1f}%)")
    by = (df.dropna(subset=["light_bin"]).groupby("light_bin", observed=True)["detected"]
          .agg(["mean", "size"]))
    for name, r in by.iterrows():
        if r["size"] > 0:
            rep.add(f"  detection rate when {name:<7}: {100 * r['mean']:.1f}%  (n={int(r['size'])})")
    d = df[df["outcome"] != OUT_BAD]
    if d["detected"].nunique() == 2:
        med = d.groupby("detected_label")[["brightness", "contrast", "sharpness"]].median().round(1)
        rep.add("Median quality, detected vs not detected:")
        rep.add(med.to_string())
    return rate, by


def describe_correlation(rep, corr):
    if corr is None:
        return
    rep.head("4. Strongest correlations between measurements")
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().dropna()
    for (a, b), r in pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).items():
        rep.add(f"  {a} vs {b}: r = {r:+.2f}")
    rep.add("  (Correlation shows a linear relationship only - it does not prove cause and effect.)")


def describe_recognition(rep, genuine, impostor, table, eer_t):
    rep.head("5. Recognition quality (same person vs different people)")
    rep.add(f"Same-person pairs: {len(genuine)}  mean similarity {genuine.mean():.3f}")
    rep.add(f"Different-person pairs: {len(impostor)}  mean similarity {impostor.mean():.3f}")
    row = table.loc[(table["threshold"] - APP_THRESHOLD).abs().idxmin()]
    rep.add(f"At the app threshold {APP_THRESHOLD}: FAR = {100 * row['FAR']:.2f}% , FRR = {100 * row['FRR']:.2f}% , "
            f"balanced accuracy = {100 * row['balanced_accuracy']:.1f}%")
    e = table.loc[table["threshold"] == eer_t].iloc[0]
    rep.add(f"Equal-error threshold: {eer_t:.2f}  (FAR = {100 * e['FAR']:.2f}%, FRR = {100 * e['FRR']:.2f}%)")
    ok = table[table["FAR"] <= 0.01]
    if len(ok):
        r = ok.iloc[0]
        rep.add(f"Lowest threshold with FAR <= 1%: {r['threshold']:.2f}  (then FRR = {100 * r['FRR']:.2f}%)")
    return row


def write_findings(rep, counts, df, shares, rate, by, row, eer_t):
    rep.head("6. Findings and what they mean for the project")
    n = 0

    def item(text):
        nonlocal n
        n += 1
        rep.add(f"{n}. {text}")

    if (counts >= 5).mean() < 0.5:
        item(f"Only {100 * (counts >= 5).mean():.0f}% of people have >= 5 images, and "
             f"{100 * (counts == 1).mean():.0f}% have just one. Same-person comparisons need several images, "
             "so use --min-images 5 for LFW, and collect 5-10 images per consenting student for your own data.")
    else:
        item("Most people have enough images (>= 5), so each person can be modelled and tested.")
    if shares["dark"] + shares["bright"] > 0.10:
        item(f"{100 * (shares['dark'] + shares['bright']):.0f}% of images are very dark or very bright. "
             "Lighting varies a lot, so test the app under bright, normal and dim light and consider "
             "brightness normalisation (for example CLAHE) before detection.")
    else:
        item("Lighting is fairly consistent in this data; still test the app in your real classroom light.")
    if shares["blurry"] > 0.10:
        item(f"{100 * shares['blurry']:.0f}% of images look blurry (low Laplacian variance). "
             "Ask students to register with a steady, in-focus photo.")
    if rate < 0.95:
        item(f"Detection fails on {100 * (1 - rate):.1f}% of images. See plots 06-08 to check whether the failures "
             "are darker, blurrier or smaller faces, and register only images that pass detection.")
    else:
        item(f"Detection succeeds on {100 * rate:.1f}% of images, so failed detections are a minor issue here.")
    multi = (df["outcome"] == OUT_MULTI).mean()
    if multi > 0.05:
        item(f"{100 * multi:.0f}% of images contain several faces. Registration in the app needs exactly one face, "
             "so those images would be rejected.")
    if (df["file_hash"].notna() & df["file_hash"].duplicated(keep=False)).any():
        item("Exact duplicate images exist. Remove them, because duplicates inflate the accuracy of same-person tests.")
    if row is not None:
        if row["FAR"] > 0.01:
            item(f"At threshold {APP_THRESHOLD} the false-accept rate is {100 * row['FAR']:.2f}%: strangers could be "
                 "marked present. Consider a higher threshold (see plot 13).")
        if row["FRR"] > 0.10:
            item(f"At threshold {APP_THRESHOLD} the false-reject rate is {100 * row['FRR']:.1f}%: real students would "
                 "often be sent to the unknown-faces folder for manual checking.")
        if abs(eer_t - APP_THRESHOLD) > 0.05:
            item(f"The equal-error threshold on this data is {eer_t:.2f}, which differs from the app's "
                 f"{APP_THRESHOLD}. Choose the threshold by weighing FAR against FRR (plot 13).")
        else:
            item(f"The app threshold {APP_THRESHOLD} is close to the equal-error threshold ({eer_t:.2f}).")
    rep.add("")
    rep.add("Note: heuristic limits (dark/bright/blurry/small-face) are starting points; adjust them in the "
            "settings at the top of EDA.py after looking at the histograms.")


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EDA for the face attendance project")
    ap.add_argument("--data", required=True, help="folder with images (see docstring for layouts)")
    ap.add_argument("--out", default="eda_output", help="output folder (default: eda_output)")
    ap.add_argument("--anonymize", action="store_true", help="replace names with P001, P002, ...")
    ap.add_argument("--no-detect", action="store_true", help="skip face detection (counts + lighting only)")
    ap.add_argument("--detector", choices=["insightface", "haar"], default="insightface")
    ap.add_argument("--min-images", type=int, default=2,
                    help="analyse only people with at least this many images (default 2; use 5 for LFW)")
    ap.add_argument("--max-images", type=int, default=1000,
                    help="max images to analyse (0 = all). Detection is slow on CPU (~0.3-0.5 s/image)")
    ap.add_argument("--per-person-cap", type=int, default=20, help="max images taken per person when sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="also open the plots in windows")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rep = Report()

    print("\nPRIVACY: use only consented or public images; do not upload photos; "
          "use --anonymize before sharing any output.\n")
    print(f"Scanning {args.data} ...")
    inv = build_inventory(args.data)
    if inv.empty:
        sys.exit(f"No images found in {args.data}")
    if args.anonymize:
        inv = anonymize_inventory(inv)

    rep.add("FACE ATTENDANCE - EXPLORATORY DATA ANALYSIS")
    rep.add(f"Data folder: {'<hidden>' if args.anonymize else args.data}   anonymized: {args.anonymize}")
    counts = describe_dataset(rep, inv)

    print("\nPlots: image counts (whole dataset)")
    safe("images per person", plot_counts, inv, args.out, args.show)
    safe("count distribution", plot_count_distribution, inv, args.out, args.show)

    subset = select_subset(inv, args.min_images, args.max_images, args.per_person_cap, args.seed)
    print(f"\nAnalysing {len(subset)} images from {subset['person'].nunique()} people ...")

    detector = None
    if not args.no_detect:
        detector = FaceDetector(args.detector)
    df, embs = measure_images(subset, args.data, detector, args.anonymize)

    print("\nPlots: quality and lighting")
    safe("image sizes", plot_image_sizes, df, args.out, args.show)
    safe("lighting", plot_lighting, df, args.out, args.show)
    safe("within-person variation", plot_variation, df, args.out, args.show)
    n_ok, shares = describe_quality(rep, df)

    rate, by, corr, row, eer_t = 1.0, None, None, None, APP_THRESHOLD
    if detector is not None:
        print("\nPlots: detection")
        safe("detection overview", plot_detection_overview, df, args.out, args.show)
        safe("detection heatmap", plot_detection_heatmap, df, args.out, args.show)
        safe("failure vs quality", plot_failure_vs_quality, df, args.out, args.show)
        safe("face geometry", plot_geometry, df, args.out, args.show)
        rate, by = describe_detection(rep, df)

    print("\nPlots: relationships")
    corr = safe("correlation", plot_correlation, df, args.out, args.show)
    safe("pairplot", plot_pairplot, df, args.out, args.show, seed=args.seed)
    describe_correlation(rep, corr)

    # Recognition analysis needs real embeddings (InsightFace) of single-face images
    if detector is not None and detector.kind == "insightface":
        idx = [i for i, e in enumerate(embs) if e is not None and df.loc[i, "outcome"] == OUT_OK]
        if len(idx) >= 10:
            emb = np.array([embs[i] for i in idx])
            persons = df.loc[idx, "person"].to_numpy()
            genuine, impostor = similarity_analysis(emb, persons, seed=args.seed)
            if len(genuine) >= 10 and len(impostor) >= 10:
                print("\nPlots: recognition (FAR / FRR)")
                table = threshold_table(genuine, impostor)
                table.to_csv(os.path.join(args.out, "threshold_table.csv"), index=False)
                eer_t = safe("similarity plots", plot_similarity, genuine, impostor, table,
                             args.out, args.show) or APP_THRESHOLD
                safe("person similarity", plot_person_similarity, emb, persons, args.out, args.show)
                row = describe_recognition(rep, genuine, impostor, table, eer_t)
            else:
                rep.head("5. Recognition quality")
                rep.add("Not enough same-person pairs. Use people with >= 2 images (--min-images 2 or higher).")
        else:
            rep.head("5. Recognition quality")
            rep.add("Too few single-face detections to compare people.")
    elif detector is not None:
        rep.head("5. Recognition quality")
        rep.add("Skipped: needs the InsightFace detector (embeddings).")

    if detector is not None:
        write_findings(rep, counts, df, shares, rate, by, row, eer_t)

    csv_path = os.path.join(args.out, "eda_image_features.csv")
    cols = [c for c in df.columns if c not in ("detected_label",)]
    df[cols].to_csv(csv_path, index=False)
    rep.write(os.path.join(args.out, "eda_summary.txt"))
    print(f"\nDone. Results are in '{args.out}/': plots (.png), eda_summary.txt, eda_image_features.csv"
          + (", threshold_table.csv" if row is not None else ""))
    if not args.anonymize:
        print("Reminder: outputs contain real names - do not publish them. Use --anonymize before sharing.")


if __name__ == "__main__":
    main()
