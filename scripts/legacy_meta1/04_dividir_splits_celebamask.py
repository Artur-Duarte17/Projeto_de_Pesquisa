import sys
from pathlib import Path
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import PROCESSED_DIR
# -----------------------------
# Pastas (dataset processado)
# -----------------------------
pasta_base = PROCESSED_DIR / "celebamask_hq_bin"
pasta_img = pasta_base / "images"
pasta_msk = pasta_base / "masks"

# Saída
pasta_split = pasta_base / "splits"
pasta_split.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Configuração do split
# -----------------------------
semente = 42
random.seed(semente)

pct_treino = 0.70
pct_val = 0.15
# pct_teste = restante

# -----------------------------
# 1) Listar IDs válidos
# -----------------------------
ids_img = {arq.stem for arq in pasta_img.glob("*.jpg")}
ids_msk = {arq.stem for arq in pasta_msk.glob("*.png")}
ids = sorted(list(ids_img.intersection(ids_msk)))

print("IDs válidos:", len(ids))

# -----------------------------
# 2) Embaralhar e dividir
# -----------------------------
random.shuffle(ids)

total = len(ids)
qt_treino = int(pct_treino * total)
qt_val = int(pct_val * total)
qt_teste = total - qt_treino - qt_val

ids_treino = ids[:qt_treino]
ids_val = ids[qt_treino:qt_treino + qt_val]
ids_teste = ids[qt_treino + qt_val:]

# -----------------------------
# 3) Salvar CSV
# -----------------------------
def salvar_csv(nome, lista_ids):
    img_paths = [pasta_img / f"{i}.jpg" for i in lista_ids]
    msk_paths = [pasta_msk / f"{i}.png" for i in lista_ids]

    tabela = pd.DataFrame({
        "id": lista_ids,
        "img": [str(p.relative_to(ROOT)) for p in img_paths],
        "msk": [str(p.relative_to(ROOT)) for p in msk_paths],
    })

    arq_saida = pasta_split / f"{nome}.csv"
    tabela.to_csv(arq_saida, index=False)
    print(f"{nome}: {len(tabela)} -> {arq_saida}")

salvar_csv("train", ids_treino)
salvar_csv("val", ids_val)
salvar_csv("test", ids_teste)

print("OK - splits prontos")
print("Treino:", len(ids_treino), "| Val:", len(ids_val), "| Teste:", len(ids_teste))