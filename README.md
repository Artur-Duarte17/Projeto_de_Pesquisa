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

## V1 atual: face + CBIR global + fusao

Os scripts novos ficam separados dos scripts historicos:

- `scripts/face/`: indexacao multi-rosto, busca por pessoa e avaliacao facial.
- `scripts/global/`: indexacao ResNet50, busca por imagem inteira e avaliacao CBIR.
- `scripts/fusion/`: combinacao ponderada dos scores facial e global.
- `scripts/app/`: interface Streamlit minima.

Os scripts antigos continuam na raiz de `scripts/` como referencia.

### 1. Indexar todos os rostos de uma pasta

```powershell
python scripts/face/01_index_faces.py `
  --input-dir data/raw/lfw/lfw_home/lfw_funneled `
  --output-dir outputs/face_index `
  --device cpu
```

Saidas:

- `outputs/face_index/face_embeddings.npy`
- `outputs/face_index/face_metadata.csv`

### 2. Buscar uma pessoa

```powershell
python scripts/face/02_search_person.py `
  --query data/query/Aaron_Peirsol_0004.jpg `
  --index-dir outputs/face_index `
  --output-dir outputs/face_index `
  --topk 10 `
  --threshold 0.35
```

O resultado retorna a foto inteira, mesmo que o indice seja por rosto.

### 3. Avaliar busca facial

Crie `evaluation_queries.csv` com:

```csv
query_id,query_path,target_label,query_type
q1,data/query/Aaron_Peirsol_0004.jpg,Aaron_Peirsol,face
```

Opcionalmente crie `relevance.csv`:

```csv
query_id,image_id,relevant
q1,abc123,1
```

Se `relevance.csv` nao for informado, a avaliacao facial usa `target_label` contra a coluna `identity` do indice.

```powershell
python scripts/face/03_evaluate_face.py `
  --queries-csv data/evaluation/face_queries.csv `
  --index-dir outputs/face_index `
  --output-dir outputs/reports
```

### 4. Indexar imagem inteira com ResNet50

```powershell
python scripts/global/01_index_global_resnet.py `
  --input-dir data/raw/holidays `
  --output-dir outputs/global_index `
  --device cpu
```

Saidas:

- `outputs/global_index/global_embeddings.npy`
- `outputs/global_index/global_metadata.csv`

### 5. Buscar imagem semelhante

```powershell
python scripts/global/02_search_global.py `
  --query data/query/example.jpg `
  --index-dir outputs/global_index `
  --output-dir outputs/global_index `
  --topk 10
```

### 6. Avaliar CBIR global

Use `evaluation_queries.csv` e `relevance.csv`. Se `relevance.csv` nao for informado, o script usa `target_label` contra `label_or_group` do indice.

```powershell
python scripts/global/03_evaluate_global.py `
  --queries-csv data/evaluation/global_queries.csv `
  --index-dir outputs/global_index `
  --output-dir outputs/reports
```

### 7. Fusao face + global

```powershell
python scripts/fusion/01_fusion_search.py `
  --query data/query/example.jpg `
  --face-index-dir outputs/face_index `
  --global-index-dir outputs/global_index `
  --output-dir outputs/fusion `
  --face-weight 0.7 `
  --global-weight 0.3
```

```powershell
python scripts/fusion/02_evaluate_fusion.py `
  --queries-csv data/evaluation/fusion_queries.csv `
  --relevance-csv data/evaluation/relevance.csv `
  --face-index-dir outputs/face_index `
  --global-index-dir outputs/global_index `
  --output-dir outputs/reports
```

### 8. Interface minima

```powershell
streamlit run scripts/app/streamlit_app.py
```

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

Para o modulo global:

```powershell
conda create -n cibir_global python=3.10 -y
conda activate cibir_global
pip install -r requirements/global.txt
```

Para a interface:

```powershell
pip install -r requirements/app.txt
```

Para fusao e app, use um ambiente que tenha ao mesmo tempo as dependencias de face e global:

```powershell
conda create -n cibir_fusion python=3.10 -y
conda activate cibir_fusion
pip install -r requirements/fusion.txt
```
