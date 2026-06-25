import sys
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR, OUTPUTS_DIR

IN_DIR = DATA_DIR / "private_faces" / "input"

OUT_DIR = OUTPUTS_DIR / "face_module"
OUT_ANN = OUT_DIR / "annotated"
OUT_ALI = OUT_DIR / "aligned"

def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(folder.rglob(e))
    return paths

def pick_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return float((x2 - x1) * (y2 - y1))

    return sorted(faces, key=area, reverse=True)[0]


def draw_face_info(img_bgr, face):
    out = img_bgr.copy()
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if hasattr(face, "kps") and face.kps is not None:
        for (x, y) in face.kps.astype(int):
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)

    return out

def align_face(img_bgr, face, out_size=112):
    """
    1) tenta alinhar com landmarks (melhor)
    2) se falhar, faz crop pelo bbox (fallback)
    """
    if hasattr(face, "kps") and face.kps is not None:
        try:
            from insightface.utils.face_align import norm_crop
            return norm_crop(img_bgr, landmark=face.kps, image_size=out_size)
        except Exception:
            pass

    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = max(0, x2); y2 = max(0, y2)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size > 0:
        return cv2.resize(crop, (out_size, out_size))
    return None

def main():
    OUT_ANN.mkdir(parents=True, exist_ok=True)
    OUT_ALI.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

    print("[OK] 06 iniciou.")
    print("IN_DIR:", IN_DIR)
    print("OUT_DIR:", OUT_DIR)
    img_paths = list_images(IN_DIR)
    if not img_paths:
        print(f"[ERRO] Nenhuma imagem em: {IN_DIR}")
        return
    print(f"[OK] {len(img_paths)} imagens encontradas.")

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[AVISO] Falha ao ler: {p}")
            continue

        faces = app.get(img)
        face = pick_largest_face(faces)
        if face is None:
            print(f"[INFO] Sem rosto: {p.name}")
            continue

        annotated = draw_face_info(img, face)
        out_ann = OUT_ANN / f"{p.stem}_annotated.jpg"
        cv2.imwrite(str(out_ann), annotated)
        print(f"[OK] {p.name} -> annotated")
        aligned = align_face(img, face, out_size=112)
        if aligned is not None:
            out_ali = OUT_ALI / f"{p.stem}_aligned.jpg"
            cv2.imwrite(str(out_ali), aligned)
            print(f"[OK] {p.name} -> aligned")
        else:
            print(f"[AVISO] {p.name} sem aligned (fallback falhou)")

if __name__ == "__main__":
    main()