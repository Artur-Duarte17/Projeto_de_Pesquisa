import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import RAW_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "lfw_face_module"
EMB_NPY = OUT_DIR / "embeddings.npy"
META_CSV = OUT_DIR / "meta.csv"

# Comece com um limite para testar (depois você aumenta)
MAX_IMAGES = None  # ex: 2000


def find_lfw_funneled():
    """
    Tenta achar a pasta lfw_funneled dentro de data/raw/lfw,
    independente da estrutura exata (sklearn costuma criar lfw_home/lfw_funneled).
    """
    base = RAW_DIR / "lfw"
    # caminho mais comum do sklearn
    cand1 = base / "lfw_home" / "lfw_funneled"
    if cand1.exists():
        return cand1

    # fallback: busca recursiva
    cands = list(base.rglob("lfw_funneled"))
    if cands:
        return cands[0]

    raise FileNotFoundError(f"Não achei lfw_funneled dentro de: {base}")


def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(folder.rglob(e))

    # ignora lixo do Jupyter
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    return sorted(paths)


def pick_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return float((x2 - x1) * (y2 - y1))

    return sorted(faces, key=area, reverse=True)[0]


def get_embedding(face):
    # compatibilidade entre versões
    if hasattr(face, "embedding") and face.embedding is not None:
        return face.embedding
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return face.normed_embedding
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        LFW_DIR = find_lfw_funneled()
    except FileNotFoundError as e:
        print("[ERRO]", e)
        return

    print("[OK] LFW_DIR:", LFW_DIR)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

    img_paths = list_images(LFW_DIR)
    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"[OK] Vou processar {len(img_paths)} imagens do LFW")

    rows = []
    embs = []

    n_read_fail = 0
    n_no_face = 0
    n_no_emb = 0

    for p in tqdm(img_paths, desc="LFW embeddings", unit="img"):
        img = cv2.imread(str(p))
        if img is None:
            n_read_fail += 1
            continue

        faces = app.get(img)
        face = pick_largest_face(faces)
        if face is None:
            n_no_face += 1
            continue

        emb = get_embedding(face)
        if emb is None:
            n_no_emb += 1
            continue

        person_id = p.parent.name  # nome da pasta = identidade
        rows.append({"id": person_id, "path": str(p.relative_to(ROOT))})
        # testar troca str(p) para relative_to(ROOT).
        embs.append(emb.astype(np.float32))

    if not embs:
        print("[ERRO] Não consegui extrair embeddings do LFW.")
        return

    E = np.vstack(embs)
    np.save(EMB_NPY, E)
    pd.DataFrame(rows).to_csv(META_CSV, index=False)

    print(f"[OK] Banco LFW criado: {len(rows)} embeddings")
    print(f"[OK] Falhas leitura: {n_read_fail} | Sem rosto: {n_no_face} | Sem emb: {n_no_emb}")
    print(f"[OK] {EMB_NPY}")
    print(f"[OK] {META_CSV}")


if __name__ == "__main__":
    main()