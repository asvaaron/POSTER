"""
McNemar's Test for comparing Pyramid Cross-Fusion Transformer models.
Compares 6 models: [small, base, large] x [baseline, modified]

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
from itertools import combinations
import pandas as pd

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from models.emotion_hyp import pyramid_trans_expr

# ── Class label maps ──────────────────────────────────────────────────────────
CLASS_NAMES = {
    'rafdb':          ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    'affectnet':      ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger'],
    'affectnet8class':['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+',
                        default=['small_baseline', 'small_modified',
                                 'base_baseline',  'base_modified',
                                 'large_baseline', 'large_modified'])
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['small', 'small', 'base', 'base', 'large', 'large'])
    parser.add_argument('--head_types', type=str, nargs='+',
                        default=['simple', 'simple', 'simple', 'simple', 'simple', 'simple'],
                        help='Valid: simple | enhanced_v2 | enhanced_v3 | enhanced_v4 | enhanced_v5 | enhanced_v6 | enhanced_v7')
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--workers',        type=int, default=2)
    parser.add_argument('--gpu',            type=str, default='0,1')
    parser.add_argument('--alpha',          type=float, default=0.05)
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
    """Returns (correct: np.array, accuracy: float, gt_labels: np.array)."""
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            outputs, _ = model(imgs.cuda())
            _, predicts = torch.max(outputs, 1)
            all_preds.extend(predicts.cpu().tolist())
            all_targets.extend(targets.tolist())
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    correct = (all_preds == all_targets).astype(int)
    return correct, correct.mean(), all_targets


# ── McNemar helpers ───────────────────────────────────────────────────────────

def contingency_table(correct_a, correct_b):
    n00 = np.sum((correct_a == 1) & (correct_b == 1))
    n01 = np.sum((correct_a == 1) & (correct_b == 0))  # A right, B wrong
    n10 = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B right
    n11 = np.sum((correct_a == 0) & (correct_b == 0))
    return np.array([[n00, n01], [n10, n11]])


def run_mcnemar(table, exact=False):
    try:
        res = mcnemar(table, exact=exact, correction=(not exact))
        stat = res.statistic if not exact else float('nan')
        return stat, res.pvalue
    except Exception:
        return float('nan'), float('nan')


# ── Overall test ──────────────────────────────────────────────────────────────

def overall_mcnemar(predictions, accuracies, model_names, alpha, dataset_name):
    pairs = list(combinations(model_names, 2))
    rows = []
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
              f"n01={n01} n10={n10} | p={pval:.4f} | {'SIGNIFICANT ✓' if sig else 'not significant'}"
              + (' [exact]' if use_exact else ''))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(f'mcnemar_overall_{dataset_name}.csv', index=False)
    print(f"\nSaved → mcnemar_overall_{dataset_name}.csv")
    return df


# ── Class-level test ──────────────────────────────────────────────────────────

def class_mcnemar(predictions, gt_labels, model_names, class_names, alpha, dataset_name):
    """
    For each model pair, run a separate McNemar's test restricted to samples
    of each true class.  Applies Bonferroni correction over the number of classes.

    Interpretation per cell:
      n01 = samples of this class where A was RIGHT and B was WRONG
      n10 = samples of this class where A was WRONG and B was RIGHT
    """
    pairs = list(combinations(model_names, 2))
    num_classes   = len(class_names)
    alpha_bonf    = alpha / num_classes   # Bonferroni-corrected threshold
    rows = []

    print("\n" + "=" * 80)
    print(f"Class-Level McNemar's Test  "
          f"(Bonferroni α* = {alpha}/{num_classes} = {alpha_bonf:.4f})")
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
            mask = (gt_labels == c)
            n_cls = mask.sum()
            if n_cls == 0:
                continue

            ca, cb = ca_all[mask], cb_all[mask]
            n00 = np.sum((ca == 1) & (cb == 1))
            n01 = np.sum((ca == 1) & (cb == 0))
            n10 = np.sum((ca == 0) & (cb == 1))
            n11 = np.sum((ca == 0) & (cb == 0))

            acc_a = ca.mean()
            acc_b = cb.mean()

            table     = np.array([[n00, n01], [n10, n11]])
            use_exact = (n01 + n10) < 25
            stat, pval = run_mcnemar(table, exact=use_exact)
            sig        = pval < alpha_bonf
            note       = '[exact]' if use_exact else ''

            print(f"  {cname:<12} {n_cls:>5}  {acc_a:>6.4f}  {acc_b:>6.4f}  "
                  f"{n01:>5}  {n10:>5}  {pval:>9.4f}  "
                  f"{'✓ YES' if sig else '✗ NO':>6}  {note}")

            rows.append({
                'Model A':          na,
                'Model B':          nb,
                'Class':            cname,
                'N (class)':        n_cls,
                'Acc A':            f"{acc_a:.4f}",
                'Acc B':            f"{acc_b:.4f}",
                'n01 (A✓B✗)':      n01,
                'n10 (A✗B✓)':      n10,
                'p-value':          f"{pval:.4f}",
                f'Sig (α*={alpha_bonf:.4f})': '✓ YES' if sig else '✗ NO',
                'Test type':        'exact' if use_exact else 'chi2',
            })

    df = pd.DataFrame(rows)
    out = f'mcnemar_class_{dataset_name}.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
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

    assert len(args.checkpoints) == len(args.model_names) == len(args.model_types) == len(args.head_types), \
        "checkpoints / model_names / model_types / head_types must all have the same length"

    # ── Collect predictions ───────────────────────────────────────────────────
    predictions = {}
    accuracies  = {}
    gt_labels   = None   # same for every model — captured once

    for ckpt, name, mtype, htype in zip(
            args.checkpoints, args.model_names, args.model_types, args.head_types):
        print(f"Evaluating [{name}] ...")
        model = load_model(ckpt, num_classes, mtype, htype)
        correct, acc, labels = get_predictions(model, val_loader)
        predictions[name] = correct
        accuracies[name]  = acc
        if gt_labels is None:
            gt_labels = labels          # store ground-truth labels once
        print(f"  Overall accuracy: {acc:.4f}\n")
        del model
        torch.cuda.empty_cache()

    # ── Accuracy summary ──────────────────────────────────────────────────────
    print("=" * 50)
    print("Model Accuracies")
    print("=" * 50)
    for name in args.model_names:
        print(f"  {name:<25} {accuracies[name]:.4f}")

    # ── Run tests ─────────────────────────────────────────────────────────────
    overall_mcnemar(predictions, accuracies, args.model_names, args.alpha, args.dataset)
    class_mcnemar(predictions, gt_labels, args.model_names, class_names, args.alpha, args.dataset)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPUs: {args.gpu}  ({torch.cuda.device_count()} devices visible)")
    run_all(args)