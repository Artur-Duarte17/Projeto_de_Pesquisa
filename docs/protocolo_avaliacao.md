# Protocolo de avaliação do sistema de recuperação fotográfica

Data de referência: 15 de setembro de 2026.

## 1. Situação

Este documento define como as próximas avaliações devem ser executadas. Os resultados anteriores à correção metodológica permanecem históricos e não podem ser usados como números finais do artigo.

O código foi preparado para busca facial, busca global e fusão. Os testes sintéticos validam as regras isoladas, mas os índices e experimentos completos ainda precisam ser refeitos no ambiente definitivo.

## 2. Unidade de recuperação

A unidade retornada e avaliada é a fotografia completa, identificada por `image_id`. Na busca facial, diferentes rostos da mesma fotografia são comparados, mas somente o rosto com maior similaridade representa aquela fotografia no ranking.

Nenhuma fotografia pode aparecer duas vezes no mesmo ranking.

## 3. Exclusão da imagem-fonte

Uma imagem é excluída quando pelo menos uma destas condições é satisfeita:

1. seu caminho resolvido é igual ao caminho da consulta;
2. seu `image_id` consta em `source_image_id`;
3. seu `image_id` consta em `exclude_image_ids`;
4. seu SHA-256 é igual ao SHA-256 da consulta.

A comparação por hash é necessária para a interface, pois o arquivo enviado é copiado para outra pasta. Ela só funciona com índices recriados após a inclusão da coluna `sha256`.

As mesmas exclusões são aplicadas ao conjunto de relevância. Assim, a fotografia-fonte não conta como acerto possível nem permanece no denominador de relevantes.

## 4. Esquema das consultas

Colunas obrigatórias:

| Coluna | Significado |
|---|---|
| `query_id` | identificador único da consulta; |
| `query_path` | caminho da imagem usada na consulta. |

Colunas opcionais:

| Coluna | Significado |
|---|---|
| `target_label` | identidade ou grupo usado quando não existe CSV explícito de relevância; |
| `source_image_id` | fotografia da qual a consulta foi extraída; |
| `exclude_image_ids` | lista adicional separada por vírgula, ponto e vírgula ou barra vertical; |
| `query_type` | descrição do tipo de consulta. |

## 5. Definições das métricas

### 5.1 Precision@K

`Precision@K = relevantes encontrados nas primeiras K posições / K`.

Se forem retornadas menos de K imagens, as posições ausentes contam como não relevantes. O denominador não muda para o número de resultados retornados.

### 5.2 Recall@K

`Recall@K = relevantes encontrados nas primeiras K posições / total de relevantes elegíveis`.

O total de relevantes é calculado depois da exclusão da fotografia-fonte e de outras exclusões declaradas.

### 5.3 Average Precision

Para cada posição relevante do ranking integral, calcula-se a precisão acumulada naquela posição. A soma dessas precisões é dividida pelo total de relevantes elegíveis, incluindo relevantes que não foram recuperados.

### 5.4 mAP

`mAP` é a média da Average Precision de todas as consultas previstas no protocolo, inclusive consultas que falharam e receberam ranking vazio. Ela não é sinônimo de AP calculada apenas no Top-10.

## 6. Ranking integral e Top-K salvo

Os avaliadores solicitam o ranking integral de todas as imagens elegíveis. Esse ranking é usado para AP/mAP, Precision@K e Recall@K.

O parâmetro `--save-topk` controla somente quantas linhas de cada consulta são gravadas nos arquivos de exemplos. Alterar `--save-topk` não pode alterar as métricas.

O parâmetro antigo `--topk` permanece temporariamente como alias de compatibilidade, mas novos comandos devem usar `--save-topk`.

## 7. Manifestos

Cada execução de indexação, busca ou avaliação grava um manifesto JSON. O manifesto registra:

- script e método;
- data e horário UTC;
- configuração utilizada;
- caminhos, tamanhos e SHA-256 dos arquivos;
- versões das dependências principais;
- commit e estado da árvore Git;
- contagens de falhas relevantes.

Resultados usados no artigo devem ter manifesto com `dirty` igual a `false` e hashes compatíveis com os arquivos arquivados.

## 8. Testes sintéticos

O arquivo `tests/test_retrieval_methodology.py` verifica:

- denominador fixo de Precision@K;
- denominador completo de AP;
- preservação do ranking integral;
- limitação independente do Top-K salvo;
- exclusões explícitas e na relevância;
- exclusão por caminho e por SHA-256;
- propagação da exclusão para a fusão;
- criação do manifesto com hashes.

Comando rápido:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## 9. Condições antes dos experimentos completos

1. ambiente CUDA único validado pelo relatório `EX-007_environment_report.json`;
2. versões congeladas em `requirements/experiment-gpu.lock.txt`;
3. árvore Git limpa;
4. consultas e relevância congeladas;
5. índices recriados com a coluna `sha256`;
6. testes sintéticos aprovados;
7. diretórios de saída novos, sem mistura com resultados históricos;
8. execução de todos os métodos no mesmo equipamento;
9. conferência dos manifestos e CSVs antes de atualizar o manuscrito.

## 10. Execuções pesadas reservadas

### 10.1 Configuração aprovada na EX-007

Inventário confirmado em 15 de setembro de 2026:

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, com 4 GB de memória;
- driver NVIDIA: 610.62;
- Python: 3.10;
- PyTorch: 2.11.0 com CUDA 12.8;
- torchvision: 0.26.0 com CUDA 12.8;
- ONNX Runtime GPU: 1.23.2.

O ambiente `laboratorio/cibir_gpu` foi criado e aprovado com testes reais de PyTorch e ONNX Runtime em CUDA. O ambiente anterior `laboratorio/cibir_face_new`, que continha PyTorch CPU e não oferecia `CUDAExecutionProvider`, foi removido depois dessa aprovação.

### 10.2 Comandos

O ambiente definitivo foi criado fora da árvore versionada. Na máquina de referência, `uv` substituiu o `conda`, que não estava disponível no `PATH`. Os comandos executados foram:

```powershell
uv venv laboratorio/cibir_gpu --python 3.10
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe -r requirements/experiment-gpu.lock.txt
laboratorio/cibir_gpu/Scripts/python.exe scripts/validate_environment.py --require-cuda
```

Os dois primeiros comandos baixaram e instalaram as dependências congeladas. O terceiro gravou `outputs/environment/EX-007_environment_report.json` e aprovou o ambiente somente depois de cálculos reais no PyTorch e no ONNX Runtime com CUDA.

A sequência reservada após a aprovação da EX-007 é:

| ID | Execução | Situação |
|---|---|---|
| EX-007 | criar, validar e congelar o ambiente CUDA | concluída e aprovada; |
| EX-008 | recriar o índice facial LFW | concluída e aprovada; 13.233 imagens e 16.058 faces; |
| EX-009 | avaliar a busca facial LFW | concluída e aprovada; 1.672 consultas; |
| EX-010 | recriar o índice facial Gallagher | concluída e aprovada; 589 fotografias e 1.303 faces; |
| EX-011 | regenerar consultas e relevância Gallagher | concluída e aprovada; 20 consultas e 882 relações de relevância; |
| EX-012 | avaliar a busca facial Gallagher | concluída e aprovada; 20 consultas; |
| EX-013 | recriar o índice global Holidays | protocolo e executor preparados; execução completa pendente; |
| EX-014 | avaliar a busca global Holidays | pendente. |

### 10.3 Protocolo congelado da EX-009

O LFW é tratado como baseline técnico de recuperação por identidade, não como substituto da avaliação central em fotografias de eventos. O protocolo foi derivado exclusivamente do índice aprovado na EX-008:

- 13.185 fotografias possuem ao menos uma face indexada;
- 5.736 identidades aparecem no índice;
- 1.672 identidades possuem ao menos duas fotografias elegíveis;
- cada identidade elegível contribui com exatamente uma consulta;
- a consulta é a fotografia com menor `SHA-256(semente, identidade, image_id)`, usando a semente `20260915`;
- as demais fotografias indexadas da mesma identidade formam o conjunto de relevância;
- a fotografia-fonte é registrada em `source_image_id` e excluída do ranking e da relevância;
- o protocolo contém 1.672 consultas e 7.449 relações de relevância.

Como todas as consultas são fotografias do próprio índice, a EX-009 reutiliza o embedding da maior face já calculado e congelado na EX-008. A fotografia-fonte continua excluída. Similaridade, redução do melhor rosto por fotografia e ordenação integral são executadas em lotes na GPU. Em uma verificação de equivalência com 20 consultas, esse caminho produziu as mesmas métricas e o mesmo Top-10 do avaliador genérico; a diferença máxima entre scores foi `3,6 × 10^-7`, compatível com arredondamento de ponto flutuante.

Os arquivos canônicos são `data/evaluation/lfw_face_queries.csv` e `data/evaluation/lfw_face_relevance.csv`. O antigo `data/evaluation/face_queries.csv`, limitado a 20 identidades sem regra de seleção documentada, foi retirado da árvore ativa e permanece recuperável pelo histórico Git.

### 10.4 Resultado auditado da EX-009

A execução no commit `ac13b6f` avaliou 1.672 consultas sem falhas. Cada ranking contém as 13.184 fotografias elegíveis, depois da exclusão da fonte. Não houve fotografia-fonte no Top-10 nem repetição de `image_id` dentro de uma consulta. Os manifestos do protocolo e da avaliação registraram árvore Git limpa e hashes compatíveis com os arquivos produzidos.

| Métrica | Resultado |
|---|---:|
| Precision@5 | 0,457177 |
| Precision@10 | 0,284629 |
| Recall@5 | 0,901149 |
| Recall@10 | 0,945308 |
| mAP | 0,965135 |

Precision@K utiliza denominador fixo K. Como parte considerável das identidades elegíveis possui menos de cinco ou dez fotografias relevantes, P@5 e P@10 não devem ser interpretadas isoladamente como taxa de identificação. O mAP elevado também deve ser contextualizado: a mediana da AP foi 1,0 e muitas identidades possuem apenas uma fotografia relevante após a exclusão da consulta. Esse baseline mede recuperação por identidade no LFW, não generalização para álbuns de eventos.

### 10.5 Protocolo congelado da EX-011

As anotações oficiais Gallagher foram cruzadas com as 587 fotografias efetivamente presentes no índice da EX-010. As identidades foram ordenadas pela quantidade decrescente de fotografias anotadas e, em caso de empate, de forma estável. Dentro de cada identidade, os candidatos foram ordenados por nome da imagem e índice da face.

Foram geradas 20 consultas de 20 identidades e 882 relações de relevância. Cada consulta é um recorte construído a partir das coordenadas dos olhos nas anotações oficiais, validado novamente pelo detector facial e associado a `source_image_id`. Três candidatos foram rejeitados por ausência de face detectável no recorte; a seleção avançou para os candidatos seguintes sem reduzir o total de consultas. Todos os 20 recortes possuem tamanho e SHA-256 registrados em `data/evaluation/gallagher_query_crops_manifest.csv`.

### 10.6 Resultado auditado da EX-012

A avaliação no commit `66d41bc` executou 20 consultas sem falhas. Cada ranking contém 586 fotografias elegíveis depois da exclusão da fonte. Não houve fotografia-fonte no Top-10 nem repetição de `image_id` dentro de uma consulta. O manifesto registrou árvore Git limpa e hashes compatíveis com os CSVs produzidos.

| Métrica | Resultado |
|---|---:|
| Precision@5 | 0,870000 |
| Precision@10 | 0,645000 |
| Recall@5 | 0,566409 |
| Recall@10 | 0,662866 |
| mAP | 0,913949 |
| Tempo médio por consulta | 615,704 ms |

A AP mínima foi 0,180849, a mediana foi 0,962851 e a máxima foi 1,0. O resultado sustenta a busca facial multi-rosto no Gallagher, mas não representa todas as identidades: o protocolo usa 20 pessoas selecionadas entre as mais frequentes e uma consulta por identidade.

### 10.7 Protocolo congelado da EX-013

A EX-013 indexará as 1.491 imagens da INRIA Holidays com uma ResNet50 pré-treinada na ImageNet. A camada de classificação é removida e a saída da penúltima camada, com 2.048 dimensões, é normalizada pela norma L2 e armazenada em `float32`. Como as imagens estão em uma pasta plana e a relevância vem dos arquivos oficiais do protocolo Holidays, o nome da pasta não é usado como rótulo.

Para aproveitar os recursos do equipamento sem alterar a representação adotada, a leitura, conversão, transformação e geração de SHA-256 usam quatro trabalhadores de CPU, enquanto a inferência da ResNet50 é executada na GPU em lotes de 32 imagens. O manifesto registra esses parâmetros, o commit Git, os arquivos de entrada e os hashes dos resultados.

Antes do congelamento, 16 imagens foram processadas tanto individualmente quanto em lote. O cosseno mínimo entre descritores correspondentes foi 0,9999987, o Top-10 permaneceu idêntico e a diferença máxima entre scores foi 0,0001963. Essa diferença é compatível com arredondamento numérico de operações CUDA em formatos de lote distintos e não alterou a ordenação verificada. Um lote de 32 imagens também foi executado sem falta de memória na RTX 3050 Ti.

As condições de aprovação da execução completa são: 1.491 imagens examinadas, 1.491 descritores, nenhuma falha, matriz `float32` finita com dimensão 2.048, vetores normalizados, hashes válidos e manifesto associado a uma árvore Git limpa.

A fusão não recebe ainda um identificador de execução final. Primeiro deve ser decidido se ela usará uma fotografia completa como consulta para as duas modalidades ou caminhos distintos para o rosto e para o contexto global. Executar a fusão antes dessa decisão produziria um resultado difícil de interpretar.
