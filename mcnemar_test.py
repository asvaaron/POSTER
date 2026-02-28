"""
McNemar's Test for comparing Pyramid Cross-Fusion Transformer models.
Compares 6 models: [small, base, large] x [baseline, modified]

Usage:
  python mcnemar_test.py \
    --dataset rafdb \
    --gpu 0,1 \
    --checkpoints small_base.pth small_mod.pth base_base.pth base_mod.pth large_base.pth large_mod.pth

Model order is assumed: small_baseline, small_modified, base_baseline, base_modified, large_baseline, large_modified
Baseline models use head_type='simple'; modified models use head_type='enhanced' (auto-assigned by default).
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='List of checkpoint paths (e.g. small_base.pth base_base.pth ...)')
    parser.add_argument('--model_names', type=str, nargs='+',
                        default=['small_baseline', 'small_modified',
                                 'base_baseline', 'base_modified',
                                 'large_baseline', 'large_modified'],
                        help='Names for each model (must match --checkpoints order)')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['small', 'small', 'base', 'base', 'large', 'large'],
                        help='Model size for each checkpoint: small/base/large')
    parser.add_argument('--head_types', type=str, nargs='+',
                        default=['simple', 'enhanced', 'simple', 'enhanced', 'simple', 'enhanced'],
                        help='Head type per model. Default alternates simple/enhanced for baseline/modified pairs.')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='Batch size for validation. Default 64 (32 per GPU with 2 GPUs).')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    return parser.parse_args()


def load_model(checkpoint_path, num_classes, model_type, head_type='simple'):
    model = pyramid_trans_expr(
        img_size=224,
        num_classes=num_classes,
        type=model_type,
        head_type=head_type
    )
    # Use all GPUs set by CUDA_VISIBLE_DEVICES
    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded: {checkpoint_path}  |  head_type={head_type}  |  GPUs={device_ids}")
    return model


def get_predictions(model, val_loader):
    """Returns binary correct/incorrect array and predicted labels."""
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            outputs, _ = model(imgs.cuda())
            _, predicts = torch.max(outputs, 1)
            all_preds.extend(predicts.cpu().tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    correct = (all_preds == all_targets).astype(int)  # 1=correct, 0=wrong
    acc = correct.mean()
    return correct, acc


def build_contingency_table(correct_a, correct_b):
    """
    Build McNemar's 2x2 contingency table.

    Table layout:
                    Model B correct   Model B wrong
    Model A correct      n00               n01
    Model A wrong        n10               n11
    """
    n00 = np.sum((correct_a == 1) & (correct_b == 1))  # both correct
    n01 = np.sum((correct_a == 1) & (correct_b == 0))  # A correct, B wrong
    n10 = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B correct
    n11 = np.sum((correct_a == 0) & (correct_b == 0))  # both wrong
    return np.array([[n00, n01], [n10, n11]])


def run_mcnemar_tests(args):
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == 'rafdb':
        datapath = './data/raf-basic/'
        num_classes = 7
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
    elif args.dataset == 'affectnet':
        datapath = './data/AffectNet/'
        num_classes = 7
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
    elif args.dataset == 'affectnet8class':
        datapath = './data/AffectNet/'
        num_classes = 8
        val_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )
    print(f"\nDataset: {args.dataset} | Val samples: {len(val_dataset)}\n")

    # ── Get predictions for all models ───────────────────────────────────────
    assert len(args.checkpoints) == len(args.model_names) == len(args.model_types), \
        "checkpoints, model_names, and model_types must have the same length"

    predictions = {}
    accuracies = {}

    for ckpt, name, mtype, htype in zip(
            args.checkpoints, args.model_names, args.model_types, args.head_types):
        print(f"Evaluating [{name}]...")
        model = load_model(ckpt, num_classes, mtype, htype)
        correct, acc = get_predictions(model, val_loader)
        predictions[name] = correct
        accuracies[name] = acc
        print(f"  Accuracy: {acc:.4f}\n")
        del model
        torch.cuda.empty_cache()

    # ── McNemar's test for all pairs ─────────────────────────────────────────
    model_names = args.model_names
    pairs = list(combinations(model_names, 2))

    results = []
    print("=" * 70)
    print("McNemar's Test Results")
    print("=" * 70)

    for (name_a, name_b) in pairs:
        table = build_contingency_table(predictions[name_a], predictions[name_b])
        # exact=True uses the exact binomial test (recommended when n01+n10 < 25)
        result = mcnemar(table, exact=False, correction=True)  # chi-squared with continuity correction

        n01 = table[0, 1]  # A correct, B wrong
        n10 = table[1, 0]  # A wrong, B correct
        significant = result.pvalue < args.alpha

        results.append({
            'Model A': name_a,
            'Model B': name_b,
            'Acc A': f"{accuracies[name_a]:.4f}",
            'Acc B': f"{accuracies[name_b]:.4f}",
            'n01 (A✓B✗)': n01,
            'n10 (A✗B✓)': n10,
            'Statistic': f"{result.statistic:.4f}",
            'p-value': f"{result.pvalue:.4f}",
            f'Sig (α={args.alpha})': '✓ YES' if significant else '✗ NO',
        })

        print(f"\n{name_a}  vs  {name_b}")
        print(f"  Acc: {accuracies[name_a]:.4f} vs {accuracies[name_b]:.4f}")
        print(f"  Contingency table:\n"
              f"               B correct  B wrong\n"
              f"  A correct  [{table[0,0]:>9}  {table[0,1]:>7}]\n"
              f"  A wrong    [{table[1,0]:>9}  {table[1,1]:>7}]")
        print(f"  n01={n01}, n10={n10} | χ²={result.statistic:.4f} | p={result.pvalue:.4f} | "
              f"{'SIGNIFICANT ✓' if significant else 'not significant'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save to CSV
    out_path = f'mcnemar_results_{args.dataset}.csv'
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # ── Accuracy summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Model Accuracies")
    print("=" * 70)
    for name in model_names:
        print(f"  {name:<25} {accuracies[name]:.4f}")


if __name__ == "__main__":
    args = parse_args()
    # Must set before any CUDA calls
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPUs: {args.gpu}  ({torch.cuda.device_count()} devices visible)")
    run_mcnemar_tests(args)