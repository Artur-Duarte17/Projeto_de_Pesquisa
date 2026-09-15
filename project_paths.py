from pathlib import Path

# Raiz do projeto (onde esta este arquivo)
ROOT = Path(__file__).resolve().parent

# Dados
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Saidas padronizadas
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
