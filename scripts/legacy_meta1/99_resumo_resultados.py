import sys
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import OUTPUTS_DIR, PROCESSED_DIR, DATA_DIR  # noqa: F401


# -----------------------------
# Utils
# -----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def cosine_batch(E: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    E = E.astype(np.float32)
    q = q.astype(np.float32)
    qn = np.linalg.norm(q) + eps
    En = np.linalg.norm(E, axis=1) + eps
    return (E @ q) / (En * qn)


def pick_best_from_log(df: pd.DataFrame | None, model_name: str) -> dict:
    """
    Tenta achar colunas padronizadas: loss_val, iou_val, dice_val
    """
    out = {"model": model_name}

    if df is None or df.empty:
        out.update({"status": "missing_log"})
        return out

    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_iou = find_col(["iou_val", "IoU_val", "val_iou", "iouValid", "iou_validação"])
    col_dice = find_col(["dice_val", "Dice_val", "val_dice", "diceValid", "dice_validação"])
    col_loss = find_col(["loss_val", "perda_val", "val_loss", "val_perda"])

    if col_iou is None:
        out.update({"status": "log_without_iou"})
        return out

    epoch_col = "epoch" if "epoch" in df.columns else None
    best_idx = int(df[col_iou].astype(float).idxmax())
    best_epoch = int(df.loc[best_idx, epoch_col]) if epoch_col else (best_idx + 1)

    out.update(
        {
            "status": "ok",
            "best_epoch": best_epoch,
            "best_iou_val": float(df.loc[best_idx, col_iou]),
            "best_dice_val": float(df.loc[best_idx, col_dice]) if col_dice else np.nan,
            "best_loss_val": float(df.loc[best_idx, col_loss]) if col_loss else np.nan,
            "n_epochs_logged": int(len(df)),
        }
    )
    return out


# -----------------------------
# Face metrics
# -----------------------------
def summarize_pairs_cosine(csv_path: Path, name: str) -> dict:
    df = safe_read_csv(csv_path)
    out = {"name": name, "pairs_csv": str(csv_path)}

    if df is None or df.empty or "tipo" not in df.columns or "cosine_sim" not in df.columns:
        out["status"] = "missing_or_invalid"
        return out

    out["status"] = "ok"
    grp = df.groupby("tipo")["cosine_sim"].agg(["count", "mean", "min", "max"]).reset_index()

    def get_stats(tipo: str):
        r = grp[grp["tipo"].astype(str).str.lower() == tipo]
        if r.empty:
            return None
        row = r.iloc[0].to_dict()
        return {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }

    out["positivo"] = get_stats("positivo")
    out["negativo"] = get_stats("negativo")
    return out


def eval_topk_from_db(
    db_emb: Path,
    db_meta: Path,
    n_queries: int = 100,
    topk: int = 5,
    seed: int = 42,
) -> dict:
    """
    Avalia Top-1 / Top-5 usando embeddings já calculados.
    Só avalia queries que possuem >= 2 imagens na mesma identidade
    (senão é impossível acertar após remover self).
    """
    if not db_emb.exists() or not db_meta.exists():
        return {"status": "missing_db"}

    E = np.load(db_emb)
    meta = pd.read_csv(db_meta)

    if len(meta) != len(E):
        return {"status": "db_misaligned", "meta_len": int(len(meta)), "emb_len": int(len(E))}

    if "id" not in meta.columns:
        return {"status": "meta_without_id"}

    counts = meta["id"].astype(str).value_counts()
    valid_ids = set(counts[counts >= 2].index.astype(str))
    valid_idx = meta.index[meta["id"].astype(str).isin(valid_ids)].to_numpy()

    if len(valid_idx) < 2:
        return {"status": "not_enough_multi_image_ids"}

    rng = random.Random(seed)
    valid_idx = list(valid_idx)
    rng.shuffle(valid_idx)
    chosen = valid_idx[: min(n_queries, len(valid_idx))]

    top1_ok = 0
    top5_ok = 0
    used = 0

    for qi in chosen:
        q_id = str(meta.loc[qi, "id"])
        q = E[qi]

        sims = cosine_batch(E, q)
        sims[qi] = -1e9  # remove self

        idx = np.argsort(-sims)[: max(topk, 5)]
        pred_ids = [str(meta.loc[i, "id"]) for i in idx]

        used += 1
        if pred_ids and pred_ids[0] == q_id:
            top1_ok += 1
        if q_id in pred_ids[:topk]:
            top5_ok += 1

    return {
        "status": "ok",
        "n_queries_used": used,
        "top1_acc": top1_ok / used if used else 0.0,
        "top5_acc": top5_ok / used if used else 0.0,
        "topk": topk,
    }


# -----------------------------
# Optional segmentation test eval (best-effort)
# -----------------------------
def try_eval_segmentation_test(
    model_kind: str,
    model_path: Path,
    test_csv: Path,
    device: str = "cpu",
    tam_img: int = 256,
) -> dict:
    """
    Best-effort: se torch/torchvision estiverem disponíveis no ambiente em que você rodar,
    calcula IoU/Dice no test.csv.
    """
    out = {"status": "skipped", "reason": "not_requested_or_unavailable"}

    if not model_path.exists() or not test_csv.exists():
        out["status"] = "missing_files"
        return out

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        import cv2
        import pandas as pd
    except Exception as e:
        out["status"] = "no_torch_env"
        out["reason"] = repr(e)
        return out

    class BaseSeg(Dataset):
        def __init__(self, arq_csv: Path, tam_img: int):
            self.tabela = pd.read_csv(arq_csv)
            self.tam_img = tam_img

        def __len__(self):
            return len(self.tabela)

        def __getitem__(self, idx):
            linha = self.tabela.iloc[idx]
            img = cv2.imread(str(ROOT / linha["img"]), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(ROOT / linha["msk"]), cv2.IMREAD_GRAYSCALE)

            if img is None or msk is None:
                raise FileNotFoundError(f"Falha ao ler: {linha['img']} | {linha['msk']}")

            img = cv2.resize(img, (self.tam_img, self.tam_img), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (self.tam_img, self.tam_img), interpolation=cv2.INTER_NEAREST)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            msk = (msk > 0).astype(np.float32)

            img_t = torch.from_numpy(img).permute(2, 0, 1)
            msk_t = torch.from_numpy(msk).unsqueeze(0)
            return img_t, msk_t

    def medir_iou_dice(logits, alvo, limiar=0.5, eps=1e-6):
        prob = torch.sigmoid(logits)
        pred = (prob > limiar).float()
        inter = (pred * alvo).sum(dim=(2, 3))
        uniao = (pred + alvo - pred * alvo).sum(dim=(2, 3))
        iou = ((inter + eps) / (uniao + eps)).mean().item()
        dice = ((2 * inter + eps) / (pred.sum(dim=(2, 3)) + alvo.sum(dim=(2, 3)) + eps)).mean().item()
        return iou, dice

    # modelo
    if model_kind == "unet":

        def bloco_conv(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        class UnetPeq(nn.Module):
            def __init__(self, base=32):
                super().__init__()
                self.e1 = bloco_conv(3, base)
                self.p1 = nn.MaxPool2d(2)
                self.e2 = bloco_conv(base, base * 2)
                self.p2 = nn.MaxPool2d(2)
                self.e3 = bloco_conv(base * 2, base * 4)
                self.p3 = nn.MaxPool2d(2)
                self.meio = bloco_conv(base * 4, base * 8)
                self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
                self.d3 = bloco_conv(base * 8, base * 4)
                self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                self.d2 = bloco_conv(base * 4, base * 2)
                self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                self.d1 = bloco_conv(base * 2, base)
                self.saida = nn.Conv2d(base, 1, 1)

            def forward(self, x):
                a1 = self.e1(x)
                a2 = self.e2(self.p1(a1))
                a3 = self.e3(self.p2(a2))
                m = self.meio(self.p3(a3))
                b3 = self.u3(m)
                b3 = torch.cat([b3, a3], dim=1)
                b3 = self.d3(b3)
                b2 = self.u2(b3)
                b2 = torch.cat([b2, a2], dim=1)
                b2 = self.d2(b2)
                b1 = self.u1(b2)
                b1 = torch.cat([b1, a1], dim=1)
                b1 = self.d1(b1)
                return self.saida(b1)

        model = UnetPeq(base=32)

        def forward_logits(m, x):
            return m(x)

    elif model_kind == "deeplab":
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50
        except Exception as e:
            out["status"] = "no_torchvision"
            out["reason"] = repr(e)
            return out

        def criar_deeplab_binario():
            try:
                return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
            except TypeError:
                pass
            try:
                return deeplabv3_resnet50(pretrained=False, num_classes=1)
            except TypeError:
                pass
            m = deeplabv3_resnet50(pretrained=False)
            last = m.classifier[-1]
            m.classifier[-1] = nn.Conv2d(last.in_channels, 1, kernel_size=1)
            return m

        model = criar_deeplab_binario()

        def forward_logits(m, x):
            outd = m(x)
            logits = outd["out"] if isinstance(outd, dict) else outd
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return logits

    else:
        out["status"] = "unknown_model_kind"
        return out

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ds = BaseSeg(test_csv, tam_img)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    ious, dices = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = forward_logits(model, x)
            iou, dice = medir_iou_dice(logits, y)
            ious.append(iou)
            dices.append(dice)

    out["status"] = "ok"
    out["iou_test"] = float(np.mean(ious)) if ious else np.nan
    out["dice_test"] = float(np.mean(dices)) if dices else np.nan
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw_queries", type=int, default=200, help="número de queries para avaliar Top-1/Top-5 no LFW")
    ap.add_argument("--topk", type=int, default=5, help="Top-K para Top-5 (por padrão 5)")
    ap.add_argument("--do_seg_test_eval", action="store_true", help="tenta calcular IoU/Dice no test recarregando modelos")
    ap.add_argument("--device", type=str, default="cpu", help="cpu ou cuda (para --do_seg_test_eval)")
    args = ap.parse_args()

    out_dir = OUTPUTS_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Segmentação -----------------
    unet_log = safe_read_csv(OUTPUTS_DIR / "segmentacao" / "log_treino.csv")
    deeplab_log = safe_read_csv(OUTPUTS_DIR / "segmentacao_deeplab" / "log_treino.csv")

    unet_best = pick_best_from_log(unet_log, "unet")
    deeplab_best = pick_best_from_log(deeplab_log, "deeplab")

    # opcional: teste real seg
    seg_test = {}
    if args.do_seg_test_eval:
        test_csv = PROCESSED_DIR / "celebamask_hq_bin" / "splits" / "test.csv"
        seg_test["unet"] = try_eval_segmentation_test(
            "unet",
            OUTPUTS_DIR / "segmentacao" / "unet_melhor.pt",
            test_csv,
            device=args.device,
        )
        seg_test["deeplab"] = try_eval_segmentation_test(
            "deeplab",
            OUTPUTS_DIR / "segmentacao_deeplab" / "deeplab_melhor.pt",
            test_csv,
            device=args.device,
        )
    else:
        seg_test["unet"] = {"status": "skipped"}
        seg_test["deeplab"] = {"status": "skipped"}

    # ----------------- Face pessoal -----------------
    face_pairs = summarize_pairs_cosine(OUTPUTS_DIR / "face_module" / "pairs_cosine.csv", "face_pessoal")

    # ----------------- LFW -----------------
    lfw_eval = eval_topk_from_db(
        OUTPUTS_DIR / "lfw_face_module" / "embeddings.npy",
        OUTPUTS_DIR / "lfw_face_module" / "meta.csv",
        n_queries=args.lfw_queries,
        topk=args.topk,
        seed=42,
    )

    # ----------------- Montar tabela final -----------------
    rows = []

    # segmentação
    for best in [unet_best, deeplab_best]:
        kind = best["model"]
        t = seg_test.get(kind, {"status": "skipped"})
        rows.append(
            {
                "section": "segmentation",
                "name": kind,
                "best_epoch": best.get("best_epoch", np.nan),
                "best_iou_val": best.get("best_iou_val", np.nan),
                "best_dice_val": best.get("best_dice_val", np.nan),
                "best_loss_val": best.get("best_loss_val", np.nan),
                "test_eval_status": t.get("status", "skipped"),
                "iou_test": t.get("iou_test", np.nan),
                "dice_test": t.get("dice_test", np.nan),
            }
        )

    # face pessoal
    rows.append(
        {
            "section": "face_personal",
            "name": "pairs_cosine",
            "pos_count": face_pairs.get("positivo", {}).get("count") if face_pairs.get("status") == "ok" else np.nan,
            "pos_mean": face_pairs.get("positivo", {}).get("mean") if face_pairs.get("status") == "ok" else np.nan,
            "neg_count": face_pairs.get("negativo", {}).get("count") if face_pairs.get("status") == "ok" else np.nan,
            "neg_mean": face_pairs.get("negativo", {}).get("mean") if face_pairs.get("status") == "ok" else np.nan,
            "status": face_pairs.get("status", "missing"),
        }
    )

    # LFW eval
    rows.append(
        {
            "section": "face_lfw",
            "name": "retrieval_eval",
            "status": lfw_eval.get("status", "missing"),
            "n_queries_used": lfw_eval.get("n_queries_used", np.nan),
            "top1_acc": lfw_eval.get("top1_acc", np.nan),
            "top5_acc": lfw_eval.get("top5_acc", np.nan),
            "topk": lfw_eval.get("topk", args.topk),
        }
    )

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "summary_results.csv"
    df_out.to_csv(csv_path, index=False)

    # ----------------- Painel simples -----------------
    fig_path = out_dir / "painel_resultados.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 7))

        # 1) barras de IoU val
        plt.subplot(2, 2, 1)
        seg_df = df_out[df_out["section"] == "segmentation"].copy()
        plt.bar(seg_df["name"], seg_df["best_iou_val"])
        plt.title("Segmentação: melhor IoU (val)")
        plt.ylim(0, 1)

        # 2) barras Dice val
        plt.subplot(2, 2, 2)
        plt.bar(seg_df["name"], seg_df["best_dice_val"])
        plt.title("Segmentação: melhor Dice (val)")
        plt.ylim(0, 1)

        # 3) face pessoal (pos vs neg mean)
        plt.subplot(2, 2, 3)
        fp = df_out[df_out["section"] == "face_personal"].iloc[0]
        vals = [fp.get("pos_mean", np.nan), fp.get("neg_mean", np.nan)]
        plt.bar(["positivo_mean", "negativo_mean"], vals)
        plt.title("Face pessoal: cosine média (pares)")
        plt.ylim(-1, 1)

        # 4) LFW Top1/Top5
        plt.subplot(2, 2, 4)
        lf = df_out[df_out["section"] == "face_lfw"].iloc[0]
        vals = [lf.get("top1_acc", np.nan), lf.get("top5_acc", np.nan)]
        plt.bar(["Top-1", f"Top-{args.topk}"], vals)
        plt.title("LFW: acurácia de recuperação")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        panel_status = "ok"
    except Exception as e:
        panel_status = f"failed: {repr(e)}"

    print("[OK] summary_results.csv:", csv_path)
    print("[OK] painel_resultados.png:", fig_path, f"({panel_status})")


if __name__ == "__main__":
    main()