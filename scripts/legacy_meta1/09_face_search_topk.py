import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.model_zoo import get_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import OUTPUTS_DIR, DATA_DIR

DB_EMB  = OUTPUTS_DIR / "face_module" / "embeddings.npy"
DB_META = OUTPUTS_DIR / "face_module" / "meta.csv"

QUERY_DIR = DATA_DIR / "query"
OUT_CSV   = OUTPUTS_DIR / "face_module" / "topk_results.csv"
TOPK = 10


def find_rec_model():
    base = Path.home() / ".insightface" / "models"
    cand = [p for p in base.rglob("w600k_r50.onnx") if ".ipynb_checkpoints" not in str(p)]
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Não achei w600k_r50.onnx em: {base}")


def cosine_batch(E, q, eps=1e-9):
    # E: N x D, q: D
    E = E.astype(np.float32)
    q = q.astype(np.float32)

    qn = np.linalg.norm(q) + eps
    En = np.linalg.norm(E, axis=1) + eps
    return (E @ q) / (En * qn)


def main():
    QUERY_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_EMB.exists() or not DB_META.exists():
        print("[ERRO] Banco não encontrado. Rode antes o script 08.")
        print("Esperado:", DB_EMB)
        print("Esperado:", DB_META)
        return

    # 1) Pegar 1 imagem de query
    q_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        q_imgs.extend(QUERY_DIR.glob(ext))

    # ignora lixo do Jupyter
    q_imgs = [p for p in q_imgs if ".ipynb_checkpoints" not in str(p)]

    if not q_imgs:
        print(f"[ERRO] Coloque uma imagem em: {QUERY_DIR}")
        return

    query_path = q_imgs[0]
    img = cv2.imread(str(query_path))
    if img is None:
        print(f"[ERRO] Não consegui ler a imagem: {query_path}")
        return

    # 2) Modelo de reconhecimento (sem detecção)
    model_path = find_rec_model()
    print("[OK] Modelo rec:", model_path)
    rec = get_model(str(model_path))
    rec.prepare(ctx_id=-1)  # CPU

    # garante 112x112
    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

    q = rec.get_feat(img).flatten().astype(np.float32)

    # 3) Carregar banco
    E = np.load(DB_EMB)              # N x D
    meta = pd.read_csv(DB_META)      # colunas: id, path

    if len(meta) != len(E):
        print("[ERRO] meta.csv e embeddings.npy estão desalinhados!")
        print("Linhas meta:", len(meta))
        print("Embeddings:", len(E))
        return

    # 4) Tentar remover auto-match (se a query for uma imagem do próprio banco)
    # meta["path"] é relativo ao ROOT (do 08)
    q_abs = query_path.resolve()
    keep_idx = []
    for i, row in meta.iterrows():
        p_rel = Path(str(row["path"]))
        p_abs = (ROOT / p_rel).resolve()
        if p_abs != q_abs:
            keep_idx.append(i)

    if len(keep_idx) < len(meta):
        meta = meta.iloc[keep_idx].reset_index(drop=True)
        E = E[keep_idx]
        print("[OK] Auto-match removido do banco (query estava no meta).")

    # 5) Similaridade e Top-K
    sims = cosine_batch(E, q)
    k = min(TOPK, len(sims))
    idx = np.argsort(-sims)[:k]

    res = meta.iloc[idx].copy()
    res["cosine_sim"] = sims[idx]
    res.insert(0, "query", str(query_path.relative_to(ROOT)) if query_path.is_relative_to(ROOT) else str(query_path))

    res.to_csv(OUT_CSV, index=False)
    print(f"[OK] Top-{k} salvo em: {OUT_CSV}")
    print(res[["id", "cosine_sim", "path"]])


if __name__ == "__main__":
    main()