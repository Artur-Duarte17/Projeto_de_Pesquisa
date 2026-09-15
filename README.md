# Projeto de Pesquisa - CBIR e Reconhecimento Facial

Protótipo de pesquisa para recuperação de imagens fotográficas por conteúdo e reconhecimento facial em coleções de grande escala.

> **Antes de usar resultados ou alterar o escopo, leia [`LEIA_PRIMEIRO.md`](LEIA_PRIMEIRO.md).** A auditoria de 15/09/2026 encerrou a segmentação como linha ativa e identificou correções obrigatórias no protocolo de avaliação. As métricas antigas são históricas, não resultados oficiais atuais.

## Estado atual

O repositorio contem a versao atual do sistema de recuperacao fotografica por face + CBIR global + fusao. Datasets, ambientes Conda, checkpoints, embeddings, imagens geradas e outros artefatos pesados ficam fora do Git.

Componentes principais da versao atual:

- indexacao facial multi-rosto com InsightFace;
- busca por pessoa retornando fotos completas;
- extracao de embeddings globais com ResNet50;
- busca por similaridade visual da imagem inteira;
- avaliacao com Precision@K, Recall@K, mAP e tempo medio;
- fusao tardia de scores facial/global;
- interface minima em Streamlit.

## Estrutura versionada

- `scripts/face/`: indexacao multi-rosto, busca por pessoa e avaliacao facial.
- `scripts/global/`: indexacao ResNet50, busca por imagem inteira e avaliacao CBIR.
- `scripts/fusion/`: combinacao ponderada dos scores facial e global.
- `scripts/app/`: interface Streamlit minima.
- `scripts/data/`: download/preparo de bases publicas usadas nos experimentos.
- `scripts/docs/`: geracao reprodutivel dos documentos finais.
- `project_paths.py`: caminhos padronizados do projeto.
- `requirements/`: dependencias minimas por ambiente.

## V1 atual: face + CBIR global + fusao

Os scripts da versao atual ficam separados dos scripts historicos:

- `scripts/face/`: indexacao multi-rosto, busca por pessoa e avaliacao facial.
- `scripts/global/`: indexacao ResNet50, busca por imagem inteira e avaliacao CBIR.
- `scripts/fusion/`: combinacao ponderada dos scores facial e global.
- `scripts/app/`: interface Streamlit minima.

Os scripts da Meta 1 foram preservados em arquivo privado e no histórico Git, mas removidos da árvore ativa. Eles não fazem parte do fluxo atual.

### 1. Indexar todos os rostos de uma pasta

```powershell
python scripts/face/01_index_faces.py `
  --input-dir data/raw/lfw/lfw_home/lfw_funneled `
  --output-dir outputs/face_index `
  --device cuda
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

### Base facial multi-rosto: Gallagher Collection

A base Gallagher e util para validar o caso real de album/evento: varias pessoas podem aparecer na mesma foto e cada rosto anotado possui uma identidade.

Baixar e preparar localmente:

```powershell
python scripts/data/download_gallagher.py `
  --output-dir data/raw/gallagher `
  --workers 8
```

Saidas locais:

- `data/raw/gallagher/images/`
- `data/raw/gallagher/metadata/image_urls.csv`
- `data/raw/gallagher/metadata/face_annotations.csv`

Observacoes:

- a base e apenas para pesquisa academica nao comercial;
- as imagens nao devem ser redistribuidas;
- `data/raw/` fica fora do Git.

Indexar a Gallagher em GPU:

```powershell
python scripts/face/01_index_faces.py `
  --input-dir data/raw/gallagher/images `
  --output-dir outputs/face_index_gallagher `
  --device cuda `
  --no-identity-from-parent
```

Preparar consultas e relevancia a partir das anotacoes oficiais:

```powershell
python scripts/face/04_prepare_gallagher_eval.py `
  --index-dir outputs/face_index_gallagher `
  --device cuda `
  --overwrite
```

Avaliar busca facial na Gallagher:

```powershell
python scripts/face/03_evaluate_face.py `
  --queries-csv data/evaluation/gallagher_face_queries.csv `
  --relevance-csv data/evaluation/gallagher_relevance.csv `
  --index-dir outputs/face_index_gallagher `
  --output-dir outputs/reports/gallagher_face `
  --topk 10 `
  --threshold 0.35 `
  --device cuda `
  --save-visual-examples
```

Analisar falsos positivos e consultas dificeis:

```powershell
python scripts/face/05_analyze_face_errors.py `
  --queries-csv data/evaluation/gallagher_face_queries.csv `
  --relevance-csv data/evaluation/gallagher_relevance.csv `
  --output-dir outputs/reports/gallagher_face/error_analysis
```

Essa avaliacao usa `source_image_id` para remover a foto original da consulta quando a consulta e um recorte gerado a partir de uma imagem ja indexada. Isso evita um acerto trivial por auto-comparacao.

### 4. Indexar imagem inteira com ResNet50

```powershell
python scripts/global/01_index_global_resnet.py `
  --input-dir data/raw/holidays `
  --output-dir outputs/global_index `
  --device cuda
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

### Base global: INRIA Holidays

A INRIA Holidays e usada para avaliar CBIR global por imagem inteira. Ela possui 1491 imagens organizadas em 500 grupos. A primeira imagem de cada grupo e a consulta; as demais imagens do mesmo grupo sao relevantes.

Baixar e preparar localmente pelo espelho Kaggle:

```powershell
python scripts/data/download_holidays.py `
  --output-dir data/raw/holidays `
  --source kaggle
```

Saidas locais:

- `data/raw/holidays/images/`
- `data/raw/holidays/metadata/holidays_metadata.csv`
- `data/evaluation/holidays_global_queries.csv`
- `data/evaluation/holidays_relevance.csv`

Indexar com ResNet50:

```powershell
python scripts/global/01_index_global_resnet.py `
  --input-dir data/raw/holidays/images `
  --output-dir outputs/global_index_holidays `
  --device cuda
```

Avaliar:

```powershell
python scripts/global/03_evaluate_global.py `
  --queries-csv data/evaluation/holidays_global_queries.csv `
  --relevance-csv data/evaluation/holidays_relevance.csv `
  --index-dir outputs/global_index_holidays `
  --output-dir outputs/reports/holidays_global `
  --topk 10 `
  --device cuda `
  --save-visual-examples
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

- `data/raw/`
- `data/processed/`
- `outputs/`
- `OUTPUTS/`
- `saidas/`
- `laboratorio/`
- `snapshots/`
- `tmp/`
- `docs/fontes_bibliograficas_brutas/`
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

O ambiente facial usa `onnxruntime-gpu` para acelerar o InsightFace com NVIDIA/CUDA. Em maquina sem GPU NVIDIA, troque por `onnxruntime==1.23.2` e execute os scripts com `--device cpu`.

Para o modulo global:

```powershell
conda create -n cibir_global python=3.10 -y
conda activate cibir_global
pip install -r requirements/global.txt
```

O `requirements/global.txt` usa wheels CUDA 12.8 do PyTorch para acelerar ResNet50 em GPU NVIDIA. Em maquina sem GPU NVIDIA, instale a versao CPU do PyTorch e execute os scripts globais com `--device cpu`.

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
