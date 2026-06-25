import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import RAW_DIR, OUTPUTS_DIR

pasta_base = RAW_DIR / "celebamask_hq"
pasta_imagens = pasta_base / "CelebA-HQ-img"
pasta_mascaras = pasta_base / "CelebAMask-HQ-mask-anno"

pasta_saida = OUTPUTS_DIR / "validacao_celebamask_hq"
pasta_saida.mkdir(parents=True, exist_ok=True)
arquivo_figura = pasta_saida / "celebamask_hq_exemplos.png"

# 1) Contagens (resultado)
imagens = sorted(pasta_imagens.glob("*.jpg"))
print("Imagens HQ (.jpg):", len(imagens))

arquivos_mascara = []
for subpasta in pasta_mascaras.iterdir():
    if subpasta.is_dir():
        arquivos_mascara.extend(list(subpasta.glob("*.png")))
print("Arquivos de máscara (partes):", len(arquivos_mascara))

ids_disponiveis = sorted({arq.name.split("_")[0] for arq in arquivos_mascara})
print("IDs com máscaras:", len(ids_disponiveis))
print("Exemplo de ID:", ids_disponiveis[0] if ids_disponiveis else "N/A")

# 2) Figura de exemplos (5 IDs) - determinístico
ids_para_exemplo = ids_disponiveis[:5]

linhas = []
for id_str in ids_para_exemplo:
    id_num = int(id_str)  # "00000" -> 0
    caminho_imagem = pasta_imagens / f"{id_num}.jpg"
    imagem = cv2.imread(str(caminho_imagem), cv2.IMREAD_COLOR)

    if imagem is None:
        print("Aviso: não consegui ler a imagem:", caminho_imagem)
        continue

    h, w = imagem.shape[:2]
    mascara_final = np.zeros((h, w), dtype=np.uint8)

    # Junta todas as partes daquele ID em uma máscara binária
    for subpasta in pasta_mascaras.iterdir():
        if not subpasta.is_dir():
            continue

        for caminho_mascara in subpasta.glob(f"{id_str}_*.png"):
            m = cv2.imread(str(caminho_mascara), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            # Máscaras são 512x512; imagens podem ser 1024x1024 -> ajusta para (w,h)
            m_bin = (m > 0).astype(np.uint8)
            if m_bin.shape[0] != h or m_bin.shape[1] != w:
                m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)

            mascara_final = np.maximum(mascara_final, m_bin)

    mascara_vis = (mascara_final * 255).astype(np.uint8)
    mascara_vis = cv2.cvtColor(mascara_vis, cv2.COLOR_GRAY2BGR)

    par = np.concatenate([imagem, mascara_vis], axis=1)  # imagem | máscara
    linhas.append(par)

if linhas:
    painel = np.concatenate(linhas, axis=0)
    cv2.imwrite(str(arquivo_figura), painel)
    print("Figura salva em:", arquivo_figura)
else:
    print("Não foi possível gerar a figura.")