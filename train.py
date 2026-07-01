import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms

from cnn_model import DentalCNN
from dataset import DentalDataset

CSV_FILE        = "labeled_data.csv"
MODEL_PATH      = "dental_cnn_trained.pth"
THRESHOLD_PATH  = "threshold.json"
BATCH_SIZE      = 16
EPOCHS          = 35
LR_BACKBONE     = 1e-4   # layer4: 낮은 LR (과적합 방지)
LR_HEAD         = 5e-4   # classifier: 높은 LR
VAL_RATIO       = 0.2
MIN_CONFIDENCE  = 0.80   # 낮은 확신도 레이블 제외

# 강화된 증강 (학습용)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_split(labels):
    crowded = [i for i, l in enumerate(labels) if l == 1]
    normal  = [i for i, l in enumerate(labels) if l == 0]
    random.shuffle(crowded); random.shuffle(normal)
    cut_c = max(1, int(len(crowded) * VAL_RATIO))
    cut_n = max(1, int(len(normal)  * VAL_RATIO))
    return crowded[cut_c:] + normal[cut_n:], crowded[:cut_c] + normal[:cut_n]


def find_best_threshold(model, val_loader, device):
    """검증셋에서 F1 최대화 threshold 탐색."""
    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            scores = torch.sigmoid(model(inputs).squeeze(1)).cpu()
            all_scores.extend(scores.tolist())
            all_targets.extend(targets.tolist())

    scores_np  = np.array(all_scores)
    targets_np = np.array(all_targets)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.91, 0.05):
        preds = (scores_np >= thr).astype(float)
        tp = ((preds == 1) & (targets_np == 1)).sum()
        fp = ((preds == 1) & (targets_np == 0)).sum()
        fn = ((preds == 0) & (targets_np == 1)).sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return round(best_thr, 2), round(best_f1, 3)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    full = DentalDataset(CSV_FILE, min_confidence=MIN_CONFIDENCE)
    labels = full.labels
    print(f"총 데이터: {len(labels)}장 (신뢰도 {MIN_CONFIDENCE}+ 필터 후)")

    train_idx, val_idx = make_split(labels)
    train_labels = [labels[i] for i in train_idx]
    n_normal  = train_labels.count(0)
    n_crowded = train_labels.count(1)
    print(f"학습: {len(train_idx)}장 (정상 {n_normal} / 삐뚤음 {n_crowded})")
    print(f"검증: {len(val_idx)}장\n")

    # WeightedRandomSampler
    w = [1.0/n_normal if l == 0 else 1.0/n_crowded for l in train_labels]
    sampler = WeightedRandomSampler(w, len(w))

    train_ds = Subset(DentalDataset(CSV_FILE, min_confidence=MIN_CONFIDENCE,
                                    transform=TRAIN_TRANSFORM), train_idx)
    val_ds   = Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)

    model = DentalCNN(pretrained=True).to(device)

    # 차등 학습률: layer4 vs classifier
    backbone_params = list(model.backbone[7].parameters())
    head_params     = list(model.classifier.parameters())
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=1e-4)

    pos_weight = torch.tensor([n_normal / n_crowded]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        # ── 학습 ──────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs).squeeze(1), targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # ── 검증 ──────────────────────────────────────────────
        model.eval()
        tp = fp = tn = fn = 0
        v_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs).squeeze(1)
                v_loss += criterion(out, targets).item()
                preds = (torch.sigmoid(out) >= 0.5).float()
                tp += ((preds==1)&(targets==1)).sum().item()
                fp += ((preds==1)&(targets==0)).sum().item()
                tn += ((preds==0)&(targets==0)).sum().item()
                fn += ((preds==0)&(targets==1)).sum().item()

        recall    = tp/(tp+fn+1e-8)
        precision = tp/(tp+fp+1e-8)
        f1        = 2*precision*recall/(precision+recall+1e-8)
        acc       = (tp+tn)/len(val_idx)*100

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"loss {t_loss/len(train_loader):.4f}→{v_loss/len(val_loader):.4f} | "
            f"Acc {acc:.1f}%  Recall {recall:.2f}  Prec {precision:.2f}  F1 {f1:.2f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           ✅ 저장 (F1 {f1:.2f})")

        scheduler.step()

    # ── 최적 threshold 탐색 ───────────────────────────────────
    print("\n최적 threshold 탐색 중...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    best_thr, best_f1_thr = find_best_threshold(model, val_loader, device)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_thr, "f1": best_f1_thr}, f)

    print(f"최적 threshold: {best_thr}  (F1 {best_f1_thr})")
    print(f"\n학습 완료 → {MODEL_PATH}  |  threshold → {THRESHOLD_PATH}")


if __name__ == "__main__":
    train()
