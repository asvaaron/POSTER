"""
Statistical comparison of Pyramid Cross-Fusion Transformer models.
Runs six tests:
  1. Overall McNemar's test             (accuracy, per sample)
  2. Class-level McNemar's test         (accuracy per emotion class)
  3. Bootstrap F1 test                  (macro-F1 and per-class F1)
  4. Cohen's Kappa                      (agreement beyond chance, per model)
  5. Confusion Matrix Chi-Square        (error pattern difference between pairs)
  6. Effect Size (Odds Ratio)           (magnitude of McNemar disagreement)

Usage:
  python mcnemar_test.py \
    --dataset rafdb \
    --gpu 0,1 \
    --checkpoints small_base.pth small_mod.pth base_base.pth base_mod.pth large_base.pth large_mod.pth \
    --head_types simple enhanced_v2 simple enhanced_v2 simple enhanced_v2

Model order: small_baseline, small_modified, base_baseline, base_modified, large_baseline, large_modified
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import argparse
import os
from torchvision import transforms
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)
from scipy.stats import chi2_contingency
from itertools import combinations
import pandas as pd

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from models.emotion_hyp import pyramid_trans_expr

# ── Class label maps ──────────────────────────────────────────────────────────
CLASS_NAMES = {
    'rafdb':           ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    'affectnet':       ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger'],
    'affectnet8class': ['Neutral', 'Happiness', 'Sadness', 'Surprise',
                        'Fear', 'Disgust', 'Anger', 'Contempt'],
}

# Cohen's Kappa interpretation thresholds (Landis & Koch 1977)
def kappa_label(k):
    if k < 0:      return 'Poor'
    if k < 0.20:   return 'Slight'
    if k < 0.40:   return 'Fair'
    if k < 0.60:   return 'Moderate'
    if k < 0.80:   return 'Substantial'
    return 'Almost Perfect'

# McNemar odds ratio interpretation
def or_label(odds_ratio):
    if odds_ratio < 1.5:  return 'Negligible'
    if odds_ratio < 2.0:  return 'Small'
    if odds_ratio < 3.0:  return 'Medium'
    return 'Large'


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='rafdb')
    parser.add_argument('--checkpoints',    type=str, nargs='+', required=True)
    parser.add_argument('--model_names',    type=str, nargs='+',
                        default=['small_baseline', 'small_modified',
                                 'base_baseline',  'base_modified',
                                 'large_baseline', 'large_modified'])
    parser.add_argument('--model_types',    type=str, nargs='+',
                        default=['small', 'small', 'base', 'base', 'large', 'large'])
    parser.add_argument('--head_types',     type=str, nargs='+',
                        default=['simple', 'simple', 'simple', 'simple', 'simple', 'simple'],
                        help='Valid: simple | enhanced_v2 | enhanced_v3 | enhanced_v4 | '
                             'enhanced_v5 | enhanced_v6 | enhanced_v7')
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--workers',        type=int, default=2)
    parser.add_argument('--gpu',            type=str, default='0,1')
    parser.add_argument('--alpha',          type=float, default=0.05)
    parser.add_argument('--bootstrap_n',    type=int, default=10000,
                        help='Bootstrap resamples for F1 test (default: 10000)')
    return parser.parse_args()


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, num_classes, model_type, head_type):
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes,
                               type=model_type, head_type=head_type)
    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded: {checkpoint_path}  |  head={head_type}  |  GPUs={device_ids}")
    return model


def get_predictions(model, val_loader):
    """Returns (preds, correct, accuracy, gt_labels) all as np.arrays."""
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            outputs, _ = model(imgs.cuda())
            _, predicts = torch.max(outputs, 1)
            all_preds.extend(predicts.cpu().tolist())
            all_targets.extend(targets.tolist())
    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    correct = (preds == targets).astype(int)
    return preds, correct, correct.mean(), targets


# ── McNemar helpers ───────────────────────────────────────────────────────────

def contingency_table(correct_a, correct_b):
    n00 = np.sum((correct_a == 1) & (correct_b == 1))
    n01 = np.sum((correct_a == 1) & (correct_b == 0))
    n10 = np.sum((correct_a == 0) & (correct_b == 1))
    n11 = np.sum((correct_a == 0) & (correct_b == 0))
    return np.array([[n00, n01], [n10, n11]])


def run_mcnemar(table, exact=False):
    try:
        res  = mcnemar(table, exact=exact, correction=(not exact))
        stat = res.statistic if not exact else float('nan')
        return stat, res.pvalue
    except Exception:
        return float('nan'), float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Overall McNemar
# ═══════════════════════════════════════════════════════════════════════════════

def overall_mcnemar(predictions, accuracies, model_names, alpha, dataset_name):
    pairs = list(combinations(model_names, 2))
    rows  = []
    print("\n" + "=" * 70)
    print("TEST 1 — Overall McNemar's Test")
    print("  H0: P(A correct, B wrong) = P(A wrong, B correct)  [n01 = n10]")
    print("  H1: Models make errors on systematically different samples")
    print("=" * 70)

    for (na, nb) in pairs:
        table = contingency_table(predictions[na], predictions[nb])
        n01, n10 = table[0, 1], table[1, 0]
        use_exact = (n01 + n10) < 25
        stat, pval = run_mcnemar(table, exact=use_exact)
        sig = pval < alpha

        rows.append({
            'Model A': na, 'Model B': nb,
            'Acc A': f"{accuracies[na]:.4f}", 'Acc B': f"{accuracies[nb]:.4f}",
            'n01 (A✓B✗)': n01, 'n10 (A✗B✓)': n10,
            'Statistic': f"{stat:.4f}" if not np.isnan(stat) else 'exact',
            'p-value': f"{pval:.4f}",
            f'Sig (α={alpha})': '✓ YES' if sig else '✗ NO',
        })
        print(f"\n  {na}  vs  {nb}")
        print(f"    Acc: {accuracies[na]:.4f} vs {accuracies[nb]:.4f} | "
              f"n01={n01} n10={n10} | p={pval:.4f} | "
              f"{'SIGNIFICANT ✓' if sig else 'not significant'}"
              + (' [exact]' if use_exact else ''))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(f'mcnemar_overall_{dataset_name}.csv', index=False)
    print(f"\n  Saved → mcnemar_overall_{dataset_name}.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Class-level McNemar
# ═══════════════════════════════════════════════════════════════════════════════

def class_mcnemar(predictions, gt_labels, model_names, class_names, alpha, dataset_name):
    pairs      = list(combinations(model_names, 2))
    alpha_bonf = alpha / len(class_names)
    rows       = []

    print("\n" + "=" * 80)
    print("TEST 2 — Class-Level McNemar's Test")
    print(f"  H0: No difference in per-class error rates between models")
    print(f"  H1: One model is significantly better/worse on a specific emotion")
    print(f"  Bonferroni α* = {alpha}/{len(class_names)} = {alpha_bonf:.4f}")
    print("=" * 80)

    for (na, nb) in pairs:
        ca_all, cb_all = predictions[na], predictions[nb]
        print(f"\n  ── {na}  vs  {nb} ──")
        header = (f"  {'Class':<12} {'N':>5}  {'AccA':>6}  {'AccB':>6}  "
                  f"{'n01':>5}  {'n10':>5}  {'p-value':>9}  {'Sig*':>6}  Note")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for c, cname in enumerate(class_names):
            mask  = (gt_labels == c)
            n_cls = mask.sum()
            if n_cls == 0:
                continue

            ca, cb = ca_all[mask], cb_all[mask]
            n00 = np.sum((ca == 1) & (cb == 1))
            n01 = np.sum((ca == 1) & (cb == 0))
            n10 = np.sum((ca == 0) & (cb == 1))
            n11 = np.sum((ca == 0) & (cb == 0))

            table      = np.array([[n00, n01], [n10, n11]])
            use_exact  = (n01 + n10) < 25
            _, pval    = run_mcnemar(table, exact=use_exact)
            sig        = pval < alpha_bonf
            note       = '[exact]' if use_exact else ''

            print(f"  {cname:<12} {n_cls:>5}  {ca.mean():>6.4f}  {cb.mean():>6.4f}  "
                  f"{n01:>5}  {n10:>5}  {pval:>9.4f}  "
                  f"{'✓ YES' if sig else '✗ NO':>6}  {note}")

            rows.append({
                'Model A': na, 'Model B': nb, 'Class': cname,
                'N (class)': n_cls,
                'Acc A': f"{ca.mean():.4f}", 'Acc B': f"{cb.mean():.4f}",
                'n01 (A✓B✗)': n01, 'n10 (A✗B✓)': n10,
                'p-value': f"{pval:.4f}",
                f'Sig (α*={alpha_bonf:.4f})': '✓ YES' if sig else '✗ NO',
                'Test type': 'exact' if use_exact else 'chi2',
            })

    df = pd.DataFrame(rows)
    df.to_csv(f'mcnemar_class_{dataset_name}.csv', index=False)
    print(f"\n  Saved → mcnemar_class_{dataset_name}.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Bootstrap F1
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_f1_test(preds_a, preds_b, gt_labels, class_names,
                      n_bootstrap=10000, seed=42):
    rng         = np.random.default_rng(seed)
    n           = len(gt_labels)
    num_classes = len(class_names)

    macro_f1_a     = f1_score(gt_labels, preds_a, average='macro', zero_division=0)
    macro_f1_b     = f1_score(gt_labels, preds_b, average='macro', zero_division=0)
    per_class_f1_a = f1_score(gt_labels, preds_a, average=None, zero_division=0)
    per_class_f1_b = f1_score(gt_labels, preds_b, average=None, zero_division=0)
    obs_macro_delta     = macro_f1_b - macro_f1_a
    obs_per_class_delta = per_class_f1_b - per_class_f1_a

    boot_macro     = np.zeros(n_bootstrap)
    boot_per_class = np.zeros((n_bootstrap, num_classes))

    for i in range(n_bootstrap):
        idx  = rng.integers(0, n, size=n)
        gt_b = gt_labels[idx]
        pa_b = preds_a[idx]
        pb_b = preds_b[idx]
        boot_macro[i] = (f1_score(gt_b, pb_b, average='macro', zero_division=0) -
                         f1_score(gt_b, pa_b, average='macro', zero_division=0))
        pc_a = f1_score(gt_b, pa_b, average=None,
                        labels=list(range(num_classes)), zero_division=0)
        pc_b = f1_score(gt_b, pb_b, average=None,
                        labels=list(range(num_classes)), zero_division=0)
        boot_per_class[i] = pc_b - pc_a

    macro_pval      = np.mean(np.abs(boot_macro) >= np.abs(obs_macro_delta))
    per_class_pvals = np.array([
        np.mean(np.abs(boot_per_class[:, c]) >= np.abs(obs_per_class_delta[c]))
        for c in range(num_classes)
    ])
    ci_lo = np.percentile(boot_macro, 2.5)
    ci_hi = np.percentile(boot_macro, 97.5)

    return {
        'macro_f1_a': macro_f1_a, 'macro_f1_b': macro_f1_b,
        'macro_delta': obs_macro_delta, 'macro_pval': macro_pval,
        'macro_ci': (ci_lo, ci_hi),
        'per_class_f1_a': per_class_f1_a, 'per_class_f1_b': per_class_f1_b,
        'per_class_delta': obs_per_class_delta, 'per_class_pvals': per_class_pvals,
    }


def run_bootstrap_f1(all_preds, gt_labels, model_names, class_names,
                     alpha, n_bootstrap, dataset_name):
    pairs      = list(combinations(model_names, 2))
    alpha_bonf = alpha / len(class_names)
    macro_rows, per_class_rows = [], []

    print("\n" + "=" * 80)
    print("TEST 3 — Bootstrap F1 Test")
    print("  H0: True macro-F1 difference Δ = F1(B) − F1(A) = 0")
    print("  H1: The F1 difference is real and not due to sampling variation")
    print(f"  Resamples={n_bootstrap} | Bonferroni α*={alpha_bonf:.4f} for per-class")
    print("=" * 80)

    for (na, nb) in pairs:
        res     = bootstrap_f1_test(all_preds[na], all_preds[nb], gt_labels,
                                    class_names, n_bootstrap=n_bootstrap)
        macro_sig       = res['macro_pval'] < alpha
        ci_lo, ci_hi    = res['macro_ci']
        ci_crosses_zero = ci_lo <= 0 <= ci_hi

        print(f"\n  ── {na}  vs  {nb} ──")
        print(f"    Macro-F1: A={res['macro_f1_a']:.4f}  B={res['macro_f1_b']:.4f}  "
              f"Δ={res['macro_delta']:+.4f}")
        print(f"    95% CI of Δ: [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"{'(CI crosses zero → unreliable)' if ci_crosses_zero else '(CI excludes zero ✓)'}")
        print(f"    p-value: {res['macro_pval']:.4f}  →  "
              f"{'SIGNIFICANT ✓' if macro_sig else 'not significant'}")

        macro_rows.append({
            'Model A': na, 'Model B': nb,
            'Macro-F1 A': f"{res['macro_f1_a']:.4f}",
            'Macro-F1 B': f"{res['macro_f1_b']:.4f}",
            'Delta (B-A)': f"{res['macro_delta']:+.4f}",
            '95% CI low': f"{ci_lo:+.4f}", '95% CI high': f"{ci_hi:+.4f}",
            'CI crosses zero': 'YES' if ci_crosses_zero else 'NO',
            'p-value': f"{res['macro_pval']:.4f}",
            f'Sig (α={alpha})': '✓ YES' if macro_sig else '✗ NO',
        })

        print(f"\n    {'Class':<12} {'F1_A':>6}  {'F1_B':>6}  {'Δ':>7}  "
              f"{'p-value':>9}  {'Sig*':>6}")
        print(f"    {'-'*55}")

        for c, cname in enumerate(class_names):
            f1a = res['per_class_f1_a'][c]
            f1b = res['per_class_f1_b'][c]
            d   = res['per_class_delta'][c]
            pv  = res['per_class_pvals'][c]
            sig = pv < alpha_bonf
            print(f"    {cname:<12} {f1a:>6.4f}  {f1b:>6.4f}  {d:>+7.4f}  "
                  f"{pv:>9.4f}  {'✓ YES' if sig else '✗ NO':>6}")
            per_class_rows.append({
                'Model A': na, 'Model B': nb, 'Class': cname,
                'F1 A': f"{f1a:.4f}", 'F1 B': f"{f1b:.4f}",
                'Delta (B-A)': f"{d:+.4f}", 'p-value': f"{pv:.4f}",
                f'Sig (α*={alpha_bonf:.4f})': '✓ YES' if sig else '✗ NO',
            })

    df_macro     = pd.DataFrame(macro_rows)
    df_per_class = pd.DataFrame(per_class_rows)
    df_macro.to_csv(f'bootstrap_f1_macro_{dataset_name}.csv', index=False)
    df_per_class.to_csv(f'bootstrap_f1_perclass_{dataset_name}.csv', index=False)
    print(f"\n  Saved → bootstrap_f1_macro_{dataset_name}.csv")
    print(f"  Saved → bootstrap_f1_perclass_{dataset_name}.csv")
    return df_macro, df_per_class


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Cohen's Kappa  (per model, not pairwise)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cohens_kappa(all_preds, gt_labels, model_names, dataset_name):
    """
    Measures agreement between each model and ground truth beyond chance.

    H0: The model's predictions agree with ground truth only by chance (κ = 0)
    H1: The model agrees with ground truth better than chance (κ > 0)

    Interpretation (Landis & Koch 1977):
      κ < 0.20  → Slight   | 0.20–0.40 → Fair      | 0.40–0.60 → Moderate
      0.60–0.80 → Substantial            | > 0.80   → Almost Perfect
    """
    rows = []
    print("\n" + "=" * 70)
    print("TEST 4 — Cohen's Kappa  (model vs ground truth)")
    print("  H0: Model agrees with ground truth only by chance  (κ = 0)")
    print("  H1: Model agreement is better than chance          (κ > 0)")
    print("  Interpretation scale: Landis & Koch (1977)")
    print("=" * 70)
    print(f"\n  {'Model':<25} {'κ':>7}  {'Interpretation':<20}  {'Better than chance?'}")
    print(f"  {'-'*70}")

    for name in model_names:
        kappa = cohen_kappa_score(gt_labels, all_preds[name])
        label = kappa_label(kappa)
        # Approximate z-test: κ > 0 is almost always true for a trained model,
        # what matters is the magnitude
        better = '✓ YES' if kappa > 0.60 else ('~ Moderate' if kappa > 0.40 else '✗ Weak')
        print(f"  {name:<25} {kappa:>7.4f}  {label:<20}  {better}")
        rows.append({
            'Model': name, 'Kappa': f"{kappa:.4f}",
            'Interpretation': label, 'Substantial+': better,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'cohens_kappa_{dataset_name}.csv', index=False)
    print(f"\n  Saved → cohens_kappa_{dataset_name}.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Confusion Matrix Chi-Square
# ═══════════════════════════════════════════════════════════════════════════════

def run_confusion_chisquare(all_preds, gt_labels, model_names, class_names,
                            alpha, dataset_name):
    """
    Tests whether the PATTERN of errors (which classes get confused with which)
    differs significantly between two models.

    Unlike McNemar (which only asks correct/wrong), this test uses the full
    N×N confusion matrix to detect if Model B confuses Sad→Neutral less often
    than Model A, even if overall accuracy is similar.

    H0: The confusion matrix structure is the same for both models
        (errors are distributed identically across classes)
    H1: The two models make errors on different class pairs
        (one model has a systematically different confusion pattern)

    Uses chi-square test of independence on the flattened confusion matrices.
    Cramér's V is reported as effect size:
      V < 0.10 → Negligible | 0.10–0.30 → Small | 0.30–0.50 → Medium | > 0.50 → Large
    """
    pairs = list(combinations(model_names, 2))
    rows  = []
    num_classes = len(class_names)

    print("\n" + "=" * 80)
    print("TEST 5 — Confusion Matrix Chi-Square")
    print("  H0: Both models produce the same confusion pattern across classes")
    print("  H1: The models differ in which classes they confuse with each other")
    print("  Effect size: Cramér's V  "
          "(< 0.10 Negligible | 0.10–0.30 Small | 0.30–0.50 Medium | > 0.50 Large)")
    print("=" * 80)

    for (na, nb) in pairs:
        cm_a = confusion_matrix(gt_labels, all_preds[na],
                                labels=list(range(num_classes)))
        cm_b = confusion_matrix(gt_labels, all_preds[nb],
                                labels=list(range(num_classes)))

        # Stack confusion matrices: shape (2*N, N) — one row block per model
        # Chi-square tests if the row distributions (error profiles) differ
        combined = np.vstack([cm_a, cm_b])

        try:
            chi2, pval, dof, _ = chi2_contingency(combined, correction=False)
            # Cramér's V = sqrt(χ² / (n * (min(r,c) - 1)))
            n_total = combined.sum()
            min_dim = min(combined.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0.0
        except Exception:
            chi2, pval, dof, cramers_v = float('nan'), float('nan'), 0, float('nan')

        sig = pval < alpha

        # Effect size label
        if np.isnan(cramers_v):    v_label = 'N/A'
        elif cramers_v < 0.10:     v_label = 'Negligible'
        elif cramers_v < 0.30:     v_label = 'Small'
        elif cramers_v < 0.50:     v_label = 'Medium'
        else:                      v_label = 'Large'

        print(f"\n  ── {na}  vs  {nb} ──")
        print(f"    χ²={chi2:.4f}  dof={dof}  p={pval:.4f}  "
              f"Cramér's V={cramers_v:.4f} ({v_label})  "
              f"→  {'SIGNIFICANT ✓' if sig else 'not significant'}")

        # Show which off-diagonal cells differ most between models
        diff = cm_b.astype(int) - cm_a.astype(int)
        print(f"    Top confusion shifts (B − A, off-diagonal only):")
        off_diag = [(diff[i, j], class_names[i], class_names[j])
                    for i in range(num_classes) for j in range(num_classes) if i != j]
        off_diag.sort(key=lambda x: abs(x[0]), reverse=True)
        for delta, true_cls, pred_cls in off_diag[:5]:
            direction = 'B fixes ✓' if delta < 0 else 'B regresses ✗'
            print(f"      True={true_cls:<10} Pred={pred_cls:<10} "
                  f"Δ={delta:+4d}  {direction}")

        rows.append({
            'Model A': na, 'Model B': nb,
            'Chi2': f"{chi2:.4f}", 'dof': dof,
            'p-value': f"{pval:.4f}",
            "Cramér's V": f"{cramers_v:.4f}",
            'Effect size': v_label,
            f'Sig (α={alpha})': '✓ YES' if sig else '✗ NO',
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'confusion_chisquare_{dataset_name}.csv', index=False)
    print(f"\n  Saved → confusion_chisquare_{dataset_name}.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Effect Size (Odds Ratio from McNemar)
# ═══════════════════════════════════════════════════════════════════════════════

def run_effect_size(predictions, gt_labels, model_names, class_names,
                    alpha, dataset_name):
    """
    Computes the Odds Ratio (OR) from the McNemar 2×2 table as effect size.

    OR = n10 / n01
      OR > 1  → Model B is better  (fixes more errors than it introduces)
      OR < 1  → Model A is better
      OR = 1  → No difference

    Magnitude interpretation:
      OR < 1.5  → Negligible | 1.5–2.0 → Small | 2.0–3.0 → Medium | > 3.0 → Large

    Also reports 95% CI for OR using the log method.
    Run at both OVERALL and PER-CLASS level.
    """
    pairs = list(combinations(model_names, 2))
    rows  = []

    print("\n" + "=" * 80)
    print("TEST 6 — Effect Size: Odds Ratio (from McNemar table)")
    print("  OR = n10 / n01  (how many times more errors B fixes vs introduces)")
    print("  OR > 1 → B better | OR < 1 → A better | OR = 1 → equal")
    print("  Magnitude: < 1.5 Negligible | 1.5–2.0 Small | 2.0–3.0 Medium | > 3.0 Large")
    print("=" * 80)

    for (na, nb) in pairs:
        print(f"\n  ── {na}  vs  {nb} ──")
        print(f"  {'Level':<14} {'n01':>5}  {'n10':>5}  {'OR':>7}  "
              f"{'95% CI':>18}  {'Magnitude':<12}  Direction")
        print(f"  {'-'*75}")

        # Overall
        table    = contingency_table(predictions[na], predictions[nb])
        n01, n10 = table[0, 1], table[1, 0]
        _print_or_row(na, nb, 'Overall', n01, n10, rows, print_row=True)

        # Per class
        for c, cname in enumerate(class_names):
            mask = (gt_labels == c)
            ca   = predictions[na][mask]
            cb   = predictions[nb][mask]
            n01c = int(np.sum((ca == 1) & (cb == 0)))
            n10c = int(np.sum((ca == 0) & (cb == 1)))
            _print_or_row(na, nb, cname, n01c, n10c, rows, print_row=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'effect_size_or_{dataset_name}.csv', index=False)
    print(f"\n  Saved → effect_size_or_{dataset_name}.csv")
    return df


def _print_or_row(na, nb, level, n01, n10, rows, print_row=True):
    """Compute OR + 95% CI and optionally print one row."""
    # Add 0.5 continuity correction when cells are zero
    n01c = n01 + 0.5 if n01 == 0 else n01
    n10c = n10 + 0.5 if n10 == 0 else n10

    or_val = n10c / n01c
    # 95% CI via log method: SE(log OR) = sqrt(1/n01 + 1/n10)
    se_log = np.sqrt(1 / n01c + 1 / n10c)
    log_or = np.log(or_val)
    ci_lo  = np.exp(log_or - 1.96 * se_log)
    ci_hi  = np.exp(log_or + 1.96 * se_log)

    mag   = or_label(max(or_val, 1 / or_val))  # symmetric magnitude
    direc = f"B better ✓" if or_val > 1 else ('A better ✓' if or_val < 1 else 'Equal')

    if print_row:
        print(f"  {level:<14} {n01:>5}  {n10:>5}  {or_val:>7.3f}  "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]{'':<4}  {mag:<12}  {direc}")

    rows.append({
        'Model A': na, 'Model B': nb, 'Level': level,
        'n01 (A✓B✗)': n01, 'n10 (A✗B✓)': n10,
        'Odds Ratio': f"{or_val:.4f}",
        'OR 95% CI low': f"{ci_lo:.4f}", 'OR 95% CI high': f"{ci_hi:.4f}",
        'Magnitude': mag, 'Direction': direc,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(args):
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == 'rafdb':
        datapath, num_classes = './data/raf-basic/', 7
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
    elif args.dataset == 'affectnet':
        datapath, num_classes = './data/AffectNet/', 7
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
    elif args.dataset == 'affectnet8class':
        datapath, num_classes = './data/AffectNet/', 8
        val_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    class_names = CLASS_NAMES[args.dataset]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    print(f"\nDataset : {args.dataset} | Val samples: {len(val_dataset)}")
    print(f"Classes : {class_names}\n")

    assert (len(args.checkpoints) == len(args.model_names) ==
            len(args.model_types) == len(args.head_types)), \
        "checkpoints / model_names / model_types / head_types must all have the same length"

    # ── Collect predictions ───────────────────────────────────────────────────
    all_preds   = {}   # raw predicted labels  (F1, kappa, confusion)
    predictions = {}   # binary correct/wrong   (McNemar, effect size)
    accuracies  = {}
    gt_labels   = None

    for ckpt, name, mtype, htype in zip(
            args.checkpoints, args.model_names, args.model_types, args.head_types):
        print(f"Evaluating [{name}] ...")
        model = load_model(ckpt, num_classes, mtype, htype)
        preds, correct, acc, labels = get_predictions(model, val_loader)
        all_preds[name]   = preds
        predictions[name] = correct
        accuracies[name]  = acc
        if gt_labels is None:
            gt_labels = labels
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        kappa    = cohen_kappa_score(labels, preds)
        print(f"  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}  κ={kappa:.4f}\n")
        del model
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"  {'Model':<25} {'Acc':>6}  {'Macro-F1':>9}  {'Kappa':>7}  {'Agreement'}")
    print(f"  {'-'*60}")
    for name in args.model_names:
        f1    = f1_score(gt_labels, all_preds[name], average='macro', zero_division=0)
        kappa = cohen_kappa_score(gt_labels, all_preds[name])
        print(f"  {name:<25} {accuracies[name]:>6.4f}  {f1:>9.4f}  "
              f"{kappa:>7.4f}  {kappa_label(kappa)}")

    # ── Run all 6 tests ───────────────────────────────────────────────────────
    overall_mcnemar(predictions, accuracies, args.model_names,
                    args.alpha, args.dataset)

    class_mcnemar(predictions, gt_labels, args.model_names,
                  class_names, args.alpha, args.dataset)

    run_bootstrap_f1(all_preds, gt_labels, args.model_names,
                     class_names, args.alpha, args.bootstrap_n, args.dataset)

    run_cohens_kappa(all_preds, gt_labels, args.model_names, args.dataset)

    run_confusion_chisquare(all_preds, gt_labels, args.model_names,
                            class_names, args.alpha, args.dataset)

    run_effect_size(predictions, gt_labels, args.model_names,
                    class_names, args.alpha, args.dataset)

    print("\n" + "=" * 60)
    print("All tests complete. CSVs saved:")
    for fname in [
        f'mcnemar_overall_{args.dataset}.csv',
        f'mcnemar_class_{args.dataset}.csv',
        f'bootstrap_f1_macro_{args.dataset}.csv',
        f'bootstrap_f1_perclass_{args.dataset}.csv',
        f'cohens_kappa_{args.dataset}.csv',
        f'confusion_chisquare_{args.dataset}.csv',
        f'effect_size_or_{args.dataset}.csv',
    ]:
        print(f"  → {fname}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPUs: {args.gpu}  ({torch.cuda.device_count()} devices visible)")
    run_all(args)
