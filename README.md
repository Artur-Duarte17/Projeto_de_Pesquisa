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
- `tests/`: testes sinteticos das regras de ranking, exclusao e metricas.
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
- `outputs/face_index/face_index_manifest.json`

O indice registra o SHA-256 de cada fotografia. Isso permite reconhecer e excluir uma copia da imagem-fonte mesmo quando ela foi enviada pela interface com outro caminho.

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

As colunas opcionais `source_image_id` e `exclude_image_ids` informam imagens que nunca podem participar do ranking. `exclude_image_ids` aceita IDs separados por virgula, ponto e virgula ou barra vertical.

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
  --output-dir outputs/reports `
  --save-topk 10
```

A avaliacao calcula AP sobre o ranking integral elegivel. `--save-topk` limita apenas as linhas gravadas no CSV de exemplos. Precision@K sempre usa K como denominador, inclusive quando o metodo retorna menos de K imagens.

### 4. Protocolo facial LFW

A avaliação LFW usa uma consulta para cada identidade com pelo menos duas fotografias efetivamente indexadas. A consulta é escolhida de forma determinística por SHA-256 com a semente registrada, e todas as outras fotografias indexadas da mesma identidade são relevantes. A fotografia-fonte é explicitamente excluída.

Gerar ou conferir o protocolo congelado:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/face/06_prepare_lfw_eval.py `
  --index-dir outputs/experiments/ex-008_lfw_face_index `
  --output-dir data/evaluation `
  --seed 20260915 `
  --min-images 2 `
  --overwrite
```

Avaliar o índice recriado:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/face/07_evaluate_lfw.py `
  --queries-csv data/evaluation/lfw_face_queries.csv `
  --relevance-csv data/evaluation/lfw_face_relevance.csv `
  --index-dir outputs/experiments/ex-008_lfw_face_index `
  --output-dir outputs/experiments/ex-009_lfw_face_evaluation `
  --save-topk 10 `
  --batch-size 128 `
  --device cuda
```

O avaliador LFW reutiliza o embedding da maior face da fotografia-consulta já calculado na EX-008. Isso é equivalente a extrair novamente o mesmo vetor com o mesmo modelo, evita processamento duplicado e permite calcular similaridade, agregação por fotografia e ranking integral em lotes na GPU. O protocolo atual contém 1.672 consultas e 7.449 relações de relevância. O LFW é usado como baseline técnico de recuperação facial; ele não substitui a avaliação em fotografias de eventos com múltiplas pessoas.

A execução validada obteve P@5 de 0,457177, P@10 de 0,284629, Recall@5 de 0,901149, Recall@10 de 0,945308 e mAP de 0,965135. Como muitas identidades possuem menos de cinco imagens relevantes, posições ausentes reduzem Precision@K pelo denominador fixo definido no protocolo.

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
  --output-dir outputs/experiments/ex-010_gallagher_face_index `
  --device cuda `
  --no-identity-from-parent
```

Preparar consultas e relevancia a partir das anotacoes oficiais:

```powershell
python scripts/face/04_prepare_gallagher_eval.py `
  --index-dir outputs/experiments/ex-010_gallagher_face_index `
  --output-dir data/evaluation `
  --query-crop-dir data/query/gallagher_ex011 `
  --manifest-dir outputs/experiments/ex-011_gallagher_protocol `
  --device cuda `
  --overwrite
```

Avaliar busca facial na Gallagher:

```powershell
python scripts/face/03_evaluate_face.py `
  --queries-csv data/evaluation/gallagher_face_queries.csv `
  --relevance-csv data/evaluation/gallagher_relevance.csv `
  --index-dir outputs/experiments/ex-010_gallagher_face_index `
  --output-dir outputs/experiments/ex-012_gallagher_face_evaluation `
  --save-topk 10 `
  --threshold -1 `
  --device cuda `
  --save-visual-examples
```

Na avaliação, `--threshold -1` preserva o ranking integral necessário para AP/mAP. O limiar facial positivo é apropriado para a busca interativa, mas não para truncar o ranking usado pelas métricas.

A execução validada de 20 consultas Gallagher obteve P@5 de 0,870000, P@10 de 0,645000, Recall@5 de 0,566409, Recall@10 de 0,662866 e mAP de 0,913949. As consultas representam identidades entre as mais frequentes do acervo; essa seleção deve ser considerada ao interpretar os números.

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
  --input-dir data/raw/holidays/images `
  --output-dir outputs/experiments/ex-013_holidays_global_index `
  --device cuda `
  --batch-size 32 `
  --workers 4 `
  --no-label-from-parent
```

Saidas:

- `outputs/experiments/ex-013_holidays_global_index/global_embeddings.npy`
- `outputs/experiments/ex-013_holidays_global_index/global_metadata.csv`
- `outputs/experiments/ex-013_holidays_global_index/global_index_manifest.json`

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
  --output-dir outputs/experiments/ex-013_holidays_global_index `
  --device cuda `
  --weights imagenet `
  --batch-size 32 `
  --workers 4 `
  --no-label-from-parent
```

A execução auditada da EX-013 produziu 1.491 descritores globais de 2.048 dimensões, sem falhas, em uma matriz `float32` normalizada por L2. O manifesto correspondente está em `outputs/experiments/ex-013_holidays_global_index/global_index_manifest.json`.

Avaliar:

```powershell
python scripts/global/04_evaluate_holidays.py `
  --queries-csv data/evaluation/holidays_global_queries.csv `
  --relevance-csv data/evaluation/holidays_relevance.csv `
  --index-dir outputs/experiments/ex-013_holidays_global_index `
  --output-dir outputs/experiments/ex-014_holidays_global_evaluation `
  --save-topk 10 `
  --batch-size 128 `
  --device cuda `
  --save-visual-examples
```

O avaliador Holidays reutiliza o descritor da fotografia-consulta já congelado no índice da EX-013, exclui a própria fotografia e calcula na GPU o ranking integral das 1.490 candidatas. O Top-10 limita somente o que é salvo, não a profundidade usada no cálculo das métricas.

A EX-014 avaliou as 500 consultas oficiais sem falhas: `P@5 = 0,324000`, `P@10 = 0,177600`, `Recall@5 = 0,870215`, `Recall@10 = 0,917563` e `mAP = 0,842612`. Como a mediana é de uma imagem relevante por consulta, Precision@K deve ser interpretada em conjunto com Recall e mAP.

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
python scripts/fusion/03_prepare_paired_gallagher.py

python scripts/fusion/04_evaluate_paired_gallagher.py `
  --queries-csv data/evaluation/gallagher_fusion_queries.csv `
  --relevance-csv data/evaluation/gallagher_relevance.csv `
  --face-index-dir outputs/experiments/ex-010_gallagher_face_index `
  --global-index-dir outputs/experiments/ex-015_gallagher_global_index `
  --output-dir outputs/experiments/ex-017_gallagher_paired_fusion `
  --save-topk 10 `
  --device cuda `
  --save-visual-examples
```

Na avaliação pareada, o recorte representa a pessoa-alvo e a fotografia-fonte completa representa o contexto global da mesma consulta lógica. A fonte é excluída de todos os rankings. Os pesos somente face, somente contexto, `0,9/0,1`, `0,7/0,3` e `0,5/0,5` são comparados como análise de sensibilidade predefinida.

Cada indexacao, busca e avaliacao grava um manifesto JSON com configuracao, hashes dos arquivos de entrada e saida, versoes do ambiente e estado do Git.

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
- `docs/Referencias/`
- `docs/entregas/`
- checkpoints como `*.pt`, `*.pth`, `*.ckpt`
- embeddings como `*.npy`, `*.npz`

Para compartilhar datasets ou modelos, use armazenamento externo, GitHub Releases, DVC, Hugging Face Hub, Google Drive ou outro repositório de artefatos.

## Ambiente reproduzivel

Face, CBIR global, fusao e Streamlit devem usar o mesmo ambiente Python 3.10. As dependencias diretas ficam em `requirements/experiment-gpu.in`; o arquivo congelado e instalavel e `requirements/experiment-gpu.lock.txt`. Os arquivos `face.txt`, `global.txt`, `fusion.txt` e `app.txt` sao atalhos para esse mesmo conjunto, evitando combinacoes CPU/GPU incompatíveis.

Na maquina de referencia, crie um ambiente novo sem modificar o ambiente historico preservado:

```powershell
uv venv laboratorio/cibir_gpu --python 3.10
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe -r requirements/experiment-gpu.lock.txt
laboratorio/cibir_gpu/Scripts/python.exe scripts/validate_environment.py --require-cuda
```

O ambiente usa PyTorch CUDA 12.8 e `onnxruntime-gpu`. O ultimo comando grava `outputs/environment/EX-007_environment_report.json` e exige que pequenos calculos reais funcionem nas duas bibliotecas com CUDA antes de liberar os experimentos.

Em maquina sem GPU NVIDIA, deve-se criar uma especificacao CPU separada; nao altere o arquivo congelado usado para os resultados oficiais.

## Testes metodologicos rapidos

Os testes nao carregam modelos nem percorrem datasets:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

As definicoes formais e as condicoes para a proxima execucao estao em [`docs/protocolo_avaliacao.md`](docs/protocolo_avaliacao.md).
