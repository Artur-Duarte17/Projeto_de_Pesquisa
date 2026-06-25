import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import RAW_DIR, PROCESSED_DIR

# -----------------------------
# Entradas (raw)
# -----------------------------
pasta_base = RAW_DIR / "celebamask_hq"
pasta_imagens_raw = pasta_base / "CelebA-HQ-img"
pasta_mascaras_raw = pasta_base / "CelebAMask-HQ-mask-anno"

# -----------------------------
# Saída (processed)
# -----------------------------
pasta_saida = PROCESSED_DIR / "celebamask_hq_bin"
pasta_imagens_out = pasta_saida / "images"
pasta_mascaras_out = pasta_saida / "masks"

pasta_imagens_out.mkdir(parents=True, exist_ok=True)
pasta_mascaras_out.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Agrupar máscaras por ID "00000"
# -----------------------------
mascaras_por_id = defaultdict(list)
for subpasta in pasta_mascaras_raw.iterdir():
    if not subpasta.is_dir():
        continue
    for arq in subpasta.glob("*.png"):
        # Ex: 00000_eyebrow.png -> id = 00000
        id_img = arq.stem.split("_")[0]
        mascaras_por_id[id_img].append(arq)

ids = sorted(mascaras_por_id.keys())
print(f"[OK] IDs com máscaras: {len(ids)}")

# -----------------------------
# 2) Criar máscara binária por ID e copiar imagem
# -----------------------------
for id_img in tqdm(ids, desc="Preprocess CelebAMask-HQ (bin)"):
    arq_img = pasta_imagens_raw / f"{id_img}.jpg"
    if not arq_img.exists():
        continue

    img = cv2.imread(str(arq_img), cv2.IMREAD_COLOR)
    if img is None:
        continue

    h, w = img.shape[:2]
    mask_bin = np.zeros((h, w), dtype=np.uint8)

    for arq_m in mascaras_por_id[id_img]:
        m = cv2.imread(str(arq_m), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = np.maximum(mask_bin, (m > 0).astype(np.uint8) * 255)

    # Salvar
    cv2.imwrite(str(pasta_imagens_out / f"{id_img}.jpg"), img)
    cv2.imwrite(str(pasta_mascaras_out / f"{id_img}.png"), mask_bin)

print(f"[OK] Saída em: {pasta_saida}")
