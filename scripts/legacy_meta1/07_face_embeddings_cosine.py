import sys
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from insightface.model_zoo import get_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "face_module"
IN_DIR  = OUT_DIR / "aligned"          
CSV_OUT = OUT_DIR / "pairs_cosine.csv"

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
    if path.parent.name not in ("aligned", "annotated", "input"):
        return path.parent.name
    # Caso esteja tudo solto em aligned/, pega o "prefixo" do arquivo: como por exemplo: "kaio 01_aligned.jpg" -> "kaio" "A_01_aligned.jpg"     -> "A" "B-02_aligned.jpg"     -> "B"
    stem = path.stem

    # remove sufixos comuns
    for suf in ("_aligned", "-aligned", " aligned"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]

    # separadores mais comuns
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep)[0]

    return stem

def get_embedding(face):
    # compatibilidade entre versões
    if hasattr(face, "embedding") and face.embedding is not None:
        return face.embedding
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return face.normed_embedding
    return None

def cosine(a, b, eps=1e-9):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def find_rec_model():
    base = Path.home() / ".insightface" / "models"
    cand = list(base.rglob("w600k_r50.onnx"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Não achei w600k_r50.onnx em: {base}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = find_rec_model()
    print("[OK] Modelo rec:", model_path)
    rec = get_model(str(model_path))  
    rec.prepare(ctx_id=-1)             # CPU se for usarr GPU depois, a tem que ajusta

    img_paths = list_images(IN_DIR)
    if len(img_paths) < 2:
        print(f"[ERRO] Poucas imagens em: {IN_DIR}")
        return

    items = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

        emb = rec.get_feat(img).flatten()  # embedding 512-d
        items.append({"path": str(p.relative_to(ROOT)), "id": get_id(p), "emb": emb})

    if len(items) < 2:
        print("[ERRO] Não consegui embeddings suficientes (rostos não detectados?).")
        return

    by_id = {}
    for it in items:
        by_id.setdefault(it["id"], []).append(it)

    ids = list(by_id.keys())
    random.seed(42)

    pairs = []

    # positivos: mesma identidade (se tiver 2+ imagens)
    for pid, lst in by_id.items():
        if len(lst) >= 2:
            for _ in range(min(6, len(lst))):
                a, b = random.sample(lst, 2)
                pairs.append((a, b, "positivo"))

    # negativos: identidades diferentes
    if len(ids) >= 2:
        for _ in range(max(12, len(pairs))):
            ida, idb = random.sample(ids, 2)
            a = random.choice(by_id[ida])
            b = random.choice(by_id[idb])
            pairs.append((a, b, "negativo"))
    else:
        print("[AVISO] Só 1 identidade encontrada. Negativos não serão gerados.")

    rows = []
    for a, b, tipo in pairs:
        sim = cosine(a["emb"], b["emb"])
        rows.append({
            "img_a": a["path"],
            "img_b": b["path"],
            "id_a": a["id"],
            "id_b": b["id"],
            "tipo": tipo,
            "cosine_sim": sim
        })

    df = pd.DataFrame(rows).sort_values("cosine_sim", ascending=False)
    df.to_csv(CSV_OUT, index=False)

    print(f"[OK] CSV gerado em: {CSV_OUT}")
    print(df.groupby("tipo")["cosine_sim"].agg(["count", "mean", "min", "max"]))

if __name__ == "__main__":
    main()