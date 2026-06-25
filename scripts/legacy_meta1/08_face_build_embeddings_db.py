import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.model_zoo import get_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "face_module"
IN_DIR  = OUT_DIR / "aligned"
EMB_NPY = OUT_DIR / "embeddings.npy"
META_CSV = OUT_DIR / "meta.csv"

def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(folder.rglob(e))
    # ignora lixo do Jupyter
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    return paths

def pick_largest_face(faces):
    if not faces:
        return None
    def area(face):
        x1, y1, x2, y2 = face.bbox
        return float((x2 - x1) * (y2 - y1))
    return sorted(faces, key=area, reverse=True)[0]

def get_id(path: Path):
    # Se tiver subpastas em aligned/<ID>/img.jpg
    if path.parent.name not in ("aligned", "annotated", "input"):
        return path.parent.name

    stem = path.stem
    for suf in ("_aligned", "-aligned", " aligned"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep)[0]
    return stem

def get_embedding(face):
    if hasattr(face, "embedding") and face.embedding is not None:
        return face.embedding
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return face.normed_embedding
    return None

def find_rec_model():
    base = Path.home() / ".insightface" / "models"
    cand = list(base.rglob("w600k_r50.onnx"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Não achei w600k_r50.onnx em: {base}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Carregar modelo de reconhecimento (sem detecção)
    model_path = find_rec_model()
    print("[OK] Modelo rec:", model_path)
    rec = get_model(str(model_path))
    rec.prepare(ctx_id=-1)  # CPU

    # 2) Listar imagens aligned
    img_paths = list_images(IN_DIR)
    if not img_paths:
        print(f"[ERRO] Nenhuma imagem em {IN_DIR}")
        return

    rows = []
    embs = []

    # 3) Extrair embeddings direto das imagens 112x112
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Falha ao ler: {p}")
            continue

        # garante 112x112
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

        emb = rec.get_feat(img).flatten().astype(np.float32)
        rows.append({"id": get_id(p), "path": str(p.relative_to(ROOT))})
        embs.append(emb)

    if len(embs) < 1:
        print("[ERRO] Não consegui extrair embeddings.")
        return

    # 4) Salvar banco
    E = np.vstack(embs)  # N x D
    np.save(EMB_NPY, E)
    pd.DataFrame(rows).to_csv(META_CSV, index=False)

    print(f"[OK] Banco criado: {len(rows)} embeddings")
    print(f"[OK] {EMB_NPY}")
    print(f"[OK] {META_CSV}")

if __name__ == "__main__":
    main()