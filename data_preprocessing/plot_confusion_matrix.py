import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(cm, labels_name, title, acc, output_path=None):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Calculate per-class metrics
    per_class_precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
    per_class_recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
    per_class_f1 = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-10)

    # Calculate macro averages
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)

    # Print confusion matrix
    print("\n" + "=" * 70)
    print(f"Confusion Matrix - {title}")
    print("=" * 70)
    print(cm)
    print("\n" + "-" * 70)
    print(f"Overall Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print("-" * 70)
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for i, label in enumerate(labels_name):
        print(f"{label:<15} {per_class_precision[i]:<12.4f} {per_class_recall[i]:<12.4f} {per_class_f1[i]:<12.4f}")
    print("=" * 70 + "\n")

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.0f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    # Add metrics to title
    title_text = f'{title}\nAcc: {acc:.4f} | F1: {macro_f1:.4f} | Prec: {macro_precision:.4f} | Rec: {macro_recall:.4f}'
    plt.title(title_text, fontsize=11)

    plt.colorbar()
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=90)
    plt.yticks(num_class, labels_name)
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join('./Confusion_matrix', title)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, "acc" + str(acc) + ".png"), format='png', dpi=300)
    plt.show()