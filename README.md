# Projeto de Pesquisa - CBIR e Reconhecimento Facial

Protótipo de pesquisa para recuperação de imagens fotográficas por conteúdo e reconhecimento facial em coleções de grande escala.

## Estado atual

O repositório contém os scripts e notebooks do pipeline experimental. Datasets, ambientes Conda, checkpoints, embeddings, imagens geradas e outros artefatos pesados ficam fora do Git.

Componentes principais:

- segmentação binária com CelebAMask-HQ;
- treinamento U-Net e DeepLabV3;
- detecção/alinhamento facial com InsightFace;
- geração de embeddings e busca Top-K;
- avaliação em LFW;
- resumo de métricas em `scripts/99_resumo_resultados.py`.

## Estrutura versionada

- `scripts/`: etapas do pipeline.
- `00_run_pipeline.ipynb`: notebook orquestrador.
- `project_paths.py`: caminhos padronizados do projeto.
- `requirements/`: dependências mínimas por ambiente.

## Arquivos fora do Git

Estas pastas são locais e devem ser mantidas fora do GitHub:

- `data/`
- `outputs/`
- `OUTPUTS/`
- `saidas/`
- `laboratorio/`
- `snapshots/`
- checkpoints como `*.pt`, `*.pth`, `*.ckpt`
- embeddings como `*.npy`, `*.npz`

Para compartilhar datasets ou modelos, use armazenamento externo, GitHub Releases, DVC, Hugging Face Hub, Google Drive ou outro repositório de artefatos.

## Ambientes

O projeto foi usado com ambientes Conda locais em `laboratorio/`, mas essa pasta nao deve ser versionada. Para recriar, use os arquivos em `requirements/` como ponto de partida.

Exemplo:

```powershell
conda create -n cibir_face python=3.10 -y
conda activate cibir_face
pip install -r requirements/face.txt
```
