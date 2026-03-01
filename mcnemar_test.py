"""
Statistical comparison of Pyramid Cross-Fusion Transformer models.
Runs three tests:
  1. Overall McNemar's test         (accuracy, per sample)
  2. Class-level McNemar's test     (accuracy per emotion class)
  3. Bootstrap F1 test              (macro-F1 and per-class F1)

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
from sklearn.metrics import f1_score
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
    'affectnet8class': ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
}


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str, default='rafdb')
    parser.add_argument('--checkpoints',   type=str, nargs='+', required=True)
    parser.add_argument('--model_names',   type=str, nargs='+',
                        default=['small_baseline', 'small_modified',
                                 'base_baseline',  'base_modified',
                                 'large_baseline', 'large_modified'])
    parser.add_argument('--model_types',   type=str, nargs='+',
                        default=['small', 'small', 'base', 'base', 'large', 'large'])
    parser.add_argument('--head_types',    type=str, nargs='+',
                        default=['simple', 'simple', 'simple', 'simple', 'simple', 'simple'],
                        help='Valid: simple | enhanced_v2 | enhanced_v3 | enhanced_v4 | enhanced_v5 | enhanced_v6 | enhanced_v7')
    parser.add_argument('--val_batch_size',type=int, default=64)
    parser.add_argument('--workers',       type=int, default=2)
    parser.add_argument('--gpu',           type=str, default='0,1')
    parser.add_argument('--alpha',         type=float, default=0.05)
    parser.add_argument('--bootstrap_n',   type=int, default=10000,
                        help='Number of bootstrap resamples for F1 test (default: 10000)')
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
        res = mcnemar(table, exact=exact, correction=(not exact))
        stat = res.statistic if not exact else float('nan')
        return stat, res.pvalue
    except Exception:
        return float('nan'), float('nan')


# ── Bootstrap F1 test ─────────────────────────────────────────────────────────

def bootstrap_f1_test(preds_a, preds_b, gt_labels, class_names,
                      n_bootstrap=10000, alpha=0.05, seed=42):
    """
    For each model pair, estimates whether the difference in macro-F1
    (and per-class F1) is statistically significant via paired bootstrap.

    How it works:
      1. Compute observed F1 difference:  delta = F1(B) - F1(A)
      2. Resample the test set N times with replacement
      3. Compute F1 difference on each resample → bootstrap distribution
      4. p-value = proportion of resamples where |delta*| >= |delta_obs|
         (two-tailed test)
      5. 95% CI = [2.5th, 97.5th] percentile of bootstrap deltas
    """
    rng = np.random.default_rng(seed)
    n   = len(gt_labels)
    num_classes = len(class_names)

    # Observed F1 scores
    macro_f1_a = f1_score(gt_labels, preds_a, average='macro', zero_division=0)
    macro_f1_b = f1_score(gt_labels, preds_b, average='macro', zero_division=0)
    per_class_f1_a = f1_score(gt_labels, preds_a, average=None, zero_division=0)
    per_class_f1_b = f1_score(gt_labels, preds_b, average=None, zero_division=0)

    obs_macro_delta      = macro_f1_b - macro_f1_a
    obs_per_class_delta  = per_class_f1_b - per_class_f1_a

    # Bootstrap
    boot_macro_deltas     = np.zeros(n_bootstrap)
    boot_per_class_deltas = np.zeros((n_bootstrap, num_classes))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gt_b  = gt_labels[idx]
        pa_b  = preds_a[idx]
        pb_b  = preds_b[idx]

        f1_a_b = f1_score(gt_b, pa_b, average='macro', zero_division=0)
        f1_b_b = f1_score(gt_b, pb_b, average='macro', zero_division=0)
        boot_macro_deltas[i] = f1_b_b - f1_a_b

        pc_a = f1_score(gt_b, pa_b, average=None,
                        labels=list(range(num_classes)), zero_division=0)
        pc_b = f1_score(gt_b, pb_b, average=None,
                        labels=list(range(num_classes)), zero_division=0)
        boot_per_class_deltas[i] = pc_b - pc_a

    # Two-tailed p-value: how often does |bootstrap delta| >= |observed delta|?
    macro_pval = np.mean(np.abs(boot_macro_deltas) >= np.abs(obs_macro_delta))

    per_class_pvals = np.array([
        np.mean(np.abs(boot_per_class_deltas[:, c]) >= np.abs(obs_per_class_delta[c]))
        for c in range(num_classes)
    ])

    # Confidence intervals
    ci_lo = np.percentile(boot_macro_deltas, 2.5)
    ci_hi = np.percentile(boot_macro_deltas, 97.5)

    return {
        'macro_f1_a':        macro_f1_a,
        'macro_f1_b':        macro_f1_b,
        'macro_delta':       obs_macro_delta,
        'macro_pval':        macro_pval,
        'macro_ci':          (ci_lo, ci_hi),
        'per_class_f1_a':    per_class_f1_a,
        'per_class_f1_b':    per_class_f1_b,
        'per_class_delta':   obs_per_class_delta,
        'per_class_pvals':   per_class_pvals,
    }


def run_bootstrap_f1(all_preds, gt_labels, model_names, class_names,
                     alpha, n_bootstrap, dataset_name):
    """Run bootstrap F1 test for all model pairs and print/save results."""
    pairs = list(combinations(model_names, 2))
    # Bonferroni correction across classes
    alpha_bonf = alpha / len(class_names)

    macro_rows     = []
    per_class_rows = []

    print("\n" + "=" * 80)
    print(f"Bootstrap F1 Test  (n={n_bootstrap} resamples | "
          f"Bonferroni α* = {alpha}/{len(class_names)} = {alpha_bonf:.4f} for per-class)")
    print("=" * 80)

    for (na, nb) in pairs:
        res = bootstrap_f1_test(
            all_preds[na], all_preds[nb], gt_labels,
            class_names, n_bootstrap=n_bootstrap, alpha=alpha
        )

        macro_sig = res['macro_pval'] < alpha
        ci_lo, ci_hi = res['macro_ci']

        print(f"\n── {na}  vs  {nb} ──")
        print(f"  Macro-F1:  A={res['macro_f1_a']:.4f}  B={res['macro_f1_b']:.4f}  "
              f"Δ={res['macro_delta']:+.4f}")
        print(f"  95% CI of Δ: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  p-value: {res['macro_pval']:.4f}  →  "
              f"{'SIGNIFICANT ✓' if macro_sig else 'not significant'}")

        macro_rows.append({
            'Model A':      na,
            'Model B':      nb,
            'Macro-F1 A':   f"{res['macro_f1_a']:.4f}",
            'Macro-F1 B':   f"{res['macro_f1_b']:.4f}",
            'Delta (B-A)':  f"{res['macro_delta']:+.4f}",
            '95% CI low':   f"{ci_lo:+.4f}",
            '95% CI high':  f"{ci_hi:+.4f}",
            'p-value':      f"{res['macro_pval']:.4f}",
            f'Sig (α={alpha})': '✓ YES' if macro_sig else '✗ NO',
        })

        # Per-class
        print(f"\n  {'Class':<12} {'F1_A':>6}  {'F1_B':>6}  {'Δ':>7}  "
              f"{'p-value':>9}  {'Sig*':>6}")
        print(f"  {'-'*60}")

        for c, cname in enumerate(class_names):
            f1a = res['per_class_f1_a'][c]
            f1b = res['per_class_f1_b'][c]
            d   = res['per_class_delta'][c]
            pv  = res['per_class_pvals'][c]
            sig = pv < alpha_bonf

            print(f"  {cname:<12} {f1a:>6.4f}  {f1b:>6.4f}  {d:>+7.4f}  "
                  f"{pv:>9.4f}  {'✓ YES' if sig else '✗ NO':>6}")

            per_class_rows.append({
                'Model A':    na,
                'Model B':    nb,
                'Class':      cname,
                'F1 A':       f"{f1a:.4f}",
                'F1 B':       f"{f1b:.4f}",
                'Delta (B-A)':f"{d:+.4f}",
                'p-value':    f"{pv:.4f}",
                f'Sig (α*={alpha_bonf:.4f})': '✓ YES' if sig else '✗ NO',
            })

    df_macro     = pd.DataFrame(macro_rows)
    df_per_class = pd.DataFrame(per_class_rows)

    df_macro.to_csv(f'bootstrap_f1_macro_{dataset_name}.csv', index=False)
    df_per_class.to_csv(f'bootstrap_f1_perclass_{dataset_name}.csv', index=False)
    print(f"\nSaved → bootstrap_f1_macro_{dataset_name}.csv")
    print(f"Saved → bootstrap_f1_perclass_{dataset_name}.csv")
    return df_macro, df_per_class


# ── Overall McNemar ───────────────────────────────────────────────────────────

def overall_mcnemar(predictions, accuracies, model_names, alpha, dataset_name):
    pairs = list(combinations(model_names, 2))
    rows  = []
    print("\n" + "=" * 70)
    print("Overall McNemar's Test")
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
        print(f"\n{na}  vs  {nb}")
        print(f"  Acc: {accuracies[na]:.4f} vs {accuracies[nb]:.4f} | "
              f"n01={n01} n10={n10} | p={pval:.4f} | "
              f"{'SIGNIFICANT ✓' if sig else 'not significant'}"
              + (' [exact]' if use_exact else ''))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(f'mcnemar_overall_{dataset_name}.csv', index=False)
    print(f"\nSaved → mcnemar_overall_{dataset_name}.csv")
    return df


# ── Class-level McNemar ───────────────────────────────────────────────────────

def class_mcnemar(predictions, gt_labels, model_names, class_names, alpha, dataset_name):
    pairs      = list(combinations(model_names, 2))
    alpha_bonf = alpha / len(class_names)
    rows       = []

    print("\n" + "=" * 80)
    print(f"Class-Level McNemar's Test  "
          f"(Bonferroni α* = {alpha}/{len(class_names)} = {alpha_bonf:.4f})")
    print("=" * 80)

    for (na, nb) in pairs:
        ca_all = predictions[na]
        cb_all = predictions[nb]

        print(f"\n── {na}  vs  {nb} ──")
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

            table     = np.array([[n00, n01], [n10, n11]])
            use_exact = (n01 + n10) < 25
            stat, pval = run_mcnemar(table, exact=use_exact)
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
    print(f"\nSaved → mcnemar_class_{dataset_name}.csv")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

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
    all_preds   = {}   # raw predicted labels  (for F1)
    predictions = {}   # binary correct/wrong   (for McNemar)
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
        print(f"  Accuracy: {acc:.4f}  |  "
              f"Macro-F1: {f1_score(labels, preds, average='macro', zero_division=0):.4f}\n")
        del model
        torch.cuda.empty_cache()

    # ── Accuracy summary ──────────────────────────────────────────────────────
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"  {'Model':<25} {'Acc':>6}  {'Macro-F1':>9}")
    print(f"  {'-'*45}")
    for name in args.model_names:
        f1 = f1_score(gt_labels, all_preds[name], average='macro', zero_division=0)
        print(f"  {name:<25} {accuracies[name]:>6.4f}  {f1:>9.4f}")

    # ── Run all tests ─────────────────────────────────────────────────────────
    overall_mcnemar(predictions, accuracies, args.model_names, args.alpha, args.dataset)
    class_mcnemar(predictions, gt_labels, args.model_names, class_names, args.alpha, args.dataset)
    run_bootstrap_f1(all_preds, gt_labels, args.model_names, class_names,
                     args.alpha, args.bootstrap_n, args.dataset)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPUs: {args.gpu}  ({torch.cuda.device_count()} devices visible)")
    run_all(args)