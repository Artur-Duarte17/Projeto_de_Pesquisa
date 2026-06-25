import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import RAW_DIR
from sklearn.datasets import fetch_lfw_people

pasta_lfw = RAW_DIR / "lfw"
pasta_lfw.mkdir(parents=True, exist_ok=True)
lfw = fetch_lfw_people(data_home=str(pasta_lfw), resize=1.0)

print("Aquisicao OK")
print("Pasta:", pasta_lfw)
print("Imagens:", lfw.images.shape[0])
print("Tamanho (H x W):", lfw.images.shape[1], "x", lfw.images.shape[2])
print("Pessoas (classes):", len(lfw.target_names))