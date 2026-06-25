import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path
import random
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import PROCESSED_DIR, OUTPUTS_DIR

# -----------------
# Configuração
# -----------------
semente = 123
random.seed(semente)
np.random.seed(semente)
torch.manual_seed(semente)
torch.cuda.manual_seed_all(semente)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = (dispositivo == "cuda")

tam_img = 256        # se der falta de memória, reduza p/ 224
tam_lote = 10         # se der falta de memória, reduza p/ 4 ou 2
epocas = 100          # agora é LIMITE máximo (early stopping pode parar antes)
taxa = 1e-3

# Early Stopping
paciencia = 10       # épocas sem melhorar antes de parar
min_delta = 0.001    # melhora mínima de IoU p/ contar como melhora

# -----------------
# Pastas e arquivos
# -----------------
base_massa = PROCESSED_DIR / "celebamask_hq_bin"

# Ajuste aqui se sua pasta chamar diferente:
pasta_split = base_massa / "splits"      # ou "divisoes" / "divisões" (conforme sua estrutura)

pasta_saida = OUTPUTS_DIR / "segmentacao"
pasta_saida.mkdir(parents=True, exist_ok=True)

arq_modelo  = pasta_saida / "unet_melhor.pt"
arq_curvas  = pasta_saida / "curvas_loss_iou.png"
arq_exemplos = pasta_saida / "exemplos_predicao.png"
arq_log     = pasta_saida / "log_treino.csv"
arq_ckpt    = pasta_saida / "unet_ultimo_ckpt.pt"

# -----------------
# Dataset
# -----------------
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

        img = cv2.resize(img, (self.tam_img, self.tam_img), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.tam_img, self.tam_img), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk = (msk > 0).astype(np.float32)

        img_t = torch.from_numpy(img).permute(2, 0, 1)     # 3,H,W
        msk_t = torch.from_numpy(msk).unsqueeze(0)         # 1,H,W
        return img_t, msk_t

# -----------------
# U-Net pequena
# -----------------
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
        self.e2 = bloco_conv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = bloco_conv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)

        self.meio = bloco_conv(base*4, base*8)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = bloco_conv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = bloco_conv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = bloco_conv(base*2, base)

        self.saida = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a1 = self.e1(x)
        a2 = self.e2(self.p1(a1))
        a3 = self.e3(self.p2(a2))
        m  = self.meio(self.p3(a3))

        b3 = self.u3(m)
        b3 = torch.cat([b3, a3], dim=1)
        b3 = self.d3(b3)

        b2 = self.u2(b3)
        b2 = torch.cat([b2, a2], dim=1)
        b2 = self.d2(b2)

        b1 = self.u1(b2)
        b1 = torch.cat([b1, a1], dim=1)
        b1 = self.d1(b1)

        return self.saida(b1)  # logits

# -----------------
# Loss e métricas
# -----------------
loss_bce = nn.BCEWithLogitsLoss()

def loss_dice(logits, alvo, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * alvo).sum(dim=(2, 3)) + eps
    den = (prob + alvo).sum(dim=(2, 3)) + eps
    dice = num / den
    return 1 - dice.mean()

def medir_iou_dice(logits, alvo, limiar=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > limiar).float()

    inter = (pred * alvo).sum(dim=(2, 3))
    uniao = (pred + alvo - pred * alvo).sum(dim=(2, 3))
    iou = ((inter + eps) / (uniao + eps)).mean().item()

    dice = ((2*inter + eps) / (pred.sum(dim=(2,3)) + alvo.sum(dim=(2,3)) + eps)).mean().item()
    return iou, dice

# -----------------
# Rodar 1 época
# -----------------
def rodar_epoca(modelo, carregador, otimizador=None):
    treino = otimizador is not None
    modelo.train(treino)

    soma_loss = 0.0
    soma_iou = 0.0
    soma_dice = 0.0
    n = 0

    for img_lote, msk_lote in tqdm(carregador, leave=False):
        img_lote = img_lote.to(dispositivo)
        msk_lote = msk_lote.to(dispositivo)

        with torch.set_grad_enabled(treino):
            logits = modelo(img_lote)
            loss = loss_bce(logits, msk_lote) + loss_dice(logits, msk_lote)

            if treino:
                otimizador.zero_grad()
                loss.backward()
                otimizador.step()

        iou, dice = medir_iou_dice(logits.detach(), msk_lote)
        soma_loss += loss.item()
        soma_iou += iou
        soma_dice += dice
        n += 1

    return soma_loss / n, soma_iou / n, soma_dice / n

# -----------------
# Figuras
# -----------------
def salvar_curvas(hist):
    plt.figure()
    plt.plot(hist["loss_treino"], label="loss_treino")
    plt.plot(hist["loss_val"], label="loss_val")
    plt.plot(hist["iou_treino"], label="iou_treino")
    plt.plot(hist["iou_val"], label="iou_val")
    plt.legend()
    plt.title("Curvas - Loss e IoU")
    plt.xlabel("Época")
    plt.savefig(arq_curvas, dpi=200, bbox_inches="tight")
    plt.close()

def salvar_exemplos(modelo, arq_csv, qtd=3):
    modelo.eval()
    tabela = pd.read_csv(arq_csv)
    tabela = tabela.sample(n=min(qtd, len(tabela)), random_state=semente)

    itens = []
    for _, linha in tabela.iterrows():
        img = cv2.imread(str(ROOT / linha["img"]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(ROOT / linha["msk"]), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (tam_img, tam_img), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (tam_img, tam_img), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(dispositivo)

        with torch.no_grad():
            logits = modelo(x)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

        gt = ((msk > 0).astype(np.uint8) * 255)
        itens.append((img_rgb, gt, pred))

    linhas = len(itens)
    plt.figure(figsize=(9, 3*linhas))
    for i, (img_rgb, gt, pred) in enumerate(itens):
        plt.subplot(linhas, 3, i*3+1)
        plt.imshow(img_rgb); plt.axis("off"); plt.title("Imagem")

        plt.subplot(linhas, 3, i*3+2)
        plt.imshow(gt, cmap="gray"); plt.axis("off"); plt.title("Máscara real")

        plt.subplot(linhas, 3, i*3+3)
        plt.imshow(pred, cmap="gray"); plt.axis("off"); plt.title("Predição")

    plt.tight_layout()
    plt.savefig(arq_exemplos, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------
# Checkpoint
# -----------------
def salvar_ckpt(ep, modelo, otimizador, melhor_iou, hist, epocas_sem_melhora):
    torch.save({
        "epoca": ep,
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "melhor_iou": melhor_iou,
        "hist": hist,
        "epocas_sem_melhora": epocas_sem_melhora,
        "config": {
            "tam_img": tam_img,
            "tam_lote": tam_lote,
            "epocas": epocas,
            "taxa": taxa,
            "semente": semente,
            "paciencia": paciencia,
            "min_delta": min_delta,
        }
    }, arq_ckpt)

def carregar_ckpt(modelo, otimizador):
    if not arq_ckpt.exists():
        return 1, -1.0, None, 0

    ckpt = torch.load(arq_ckpt, map_location=dispositivo)
    modelo.load_state_dict(ckpt["modelo"])
    otimizador.load_state_dict(ckpt["otimizador"])
    melhor_iou = float(ckpt.get("melhor_iou", -1.0))
    hist = ckpt.get("hist", None)
    ep_ini = int(ckpt.get("epoca", 0)) + 1
    epocas_sem_melhora = int(ckpt.get("epocas_sem_melhora", 0))
    return ep_ini, melhor_iou, hist, epocas_sem_melhora

# -----------------
# Main
# -----------------
def main():
    print("Dispositivo:", dispositivo)

    arq_treino = pasta_split / "train.csv"
    arq_val    = pasta_split / "val.csv"
    arq_teste  = pasta_split / "test.csv"

    if not arq_treino.exists() or not arq_val.exists() or not arq_teste.exists():
        raise FileNotFoundError(
            f"Não achei treino/val/test em: {pasta_split}\n"
            f"Esperado: treino.csv, val.csv, teste.csv"
        )

    base_treino = BaseSeg(arq_treino, tam_img)
    base_val    = BaseSeg(arq_val, tam_img)
    base_teste  = BaseSeg(arq_teste, tam_img)

    pin = (dispositivo == "cuda")
    car_treino = DataLoader(base_treino, batch_size=tam_lote, shuffle=True,  num_workers=0, pin_memory=pin)
    car_val    = DataLoader(base_val,    batch_size=tam_lote, shuffle=False, num_workers=0, pin_memory=pin)
    car_teste  = DataLoader(base_teste,  batch_size=tam_lote, shuffle=False, num_workers=0, pin_memory=pin)

    modelo = UnetPeq(base=32).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)

    hist_padrao = {
        "loss_treino": [], "loss_val": [],
        "iou_treino": [], "iou_val": [],
        "dice_treino": [], "dice_val": []
    }

    ep_ini, melhor_iou, hist_ckpt, epocas_sem_melhora = carregar_ckpt(modelo, otimizador)
    hist = hist_padrao if hist_ckpt is None else hist_ckpt
    if arq_ckpt.exists():
       print(f">> Retomando ckpt: começando em epoca={ep_ini} (max={epocas}) | melhor_iou={melhor_iou:.4f} | sem_melhora={epocas_sem_melhora}")
        
    try:
        for ep in range(ep_ini, epocas + 1):
            print(f"\nÉpoca {ep}/{epocas}")

            loss_t, iou_t, dice_t = rodar_epoca(modelo, car_treino, otimizador=otimizador)
            loss_v, iou_v, dice_v = rodar_epoca(modelo, car_val, otimizador=None)

            hist["loss_treino"].append(loss_t)
            hist["loss_val"].append(loss_v)
            hist["iou_treino"].append(iou_t)
            hist["iou_val"].append(iou_v)
            hist["dice_treino"].append(dice_t)
            hist["dice_val"].append(dice_v)

            print(f"treino: loss={loss_t:.4f} iou={iou_t:.4f} dice={dice_t:.4f}")
            print(f"val:    loss={loss_v:.4f} iou={iou_v:.4f} dice={dice_v:.4f}")

            # ----- Melhor modelo + early stopping -----
            if iou_v > melhor_iou + min_delta:
                melhor_iou = iou_v
                epocas_sem_melhora = 0
                torch.save(modelo.state_dict(), arq_modelo)
                print(">> Salvou melhor modelo:", arq_modelo)
            else:
                epocas_sem_melhora += 1
                print(f">> Sem melhora relevante (patience {epocas_sem_melhora}/{paciencia})")

            # Salvar progresso
            pd.DataFrame(hist).to_csv(arq_log, index=False)
            salvar_curvas(hist)
            salvar_ckpt(ep, modelo, otimizador, melhor_iou, hist, epocas_sem_melhora)

            # Parada
            if epocas_sem_melhora >= paciencia:
                print(f">> Early stopping: sem melhora em {paciencia} épocas.")
                break

    finally:
        pd.DataFrame(hist).to_csv(arq_log, index=False)
        salvar_curvas(hist)

        if arq_modelo.exists():
            try:
                modelo.load_state_dict(torch.load(arq_modelo, map_location=dispositivo))
                salvar_exemplos(modelo, arq_teste, qtd=3)
            except Exception as e:
                print("Aviso: falhou ao gerar exemplos:", repr(e))

        print("\nArquivos gerados:")
        print("Modelo melhor:", arq_modelo if arq_modelo.exists() else "(não encontrado)")
        print("Checkpoint:", arq_ckpt if arq_ckpt.exists() else "(não encontrado)")
        print("Curvas:", arq_curvas)
        print("Exemplos:", arq_exemplos if arq_exemplos.exists() else "(não encontrado)")
        print("Log:", arq_log)

    if arq_modelo.exists():
        modelo.load_state_dict(torch.load(arq_modelo, map_location=dispositivo))
        loss_te, iou_te, dice_te = rodar_epoca(modelo, car_teste, otimizador=None)
        print(f"\nTESTE: loss={loss_te:.4f} iou={iou_te:.4f} dice={dice_te:.4f}")

if __name__ == "__main__":
    main()