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

### 10.1 Configuração de referência da EX-007

Inventário confirmado em 15 de setembro de 2026:

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, com 4 GB de memória;
- driver NVIDIA: 610.62;
- Python: 3.10;
- PyTorch: 2.11.0 com CUDA 12.8;
- torchvision: 0.26.0 com CUDA 12.8;
- ONNX Runtime GPU: 1.23.2.

O ambiente preservado `laboratorio/cibir_face_new` contém PyTorch CPU e não oferece `CUDAExecutionProvider`; por isso ele não pode produzir os resultados oficiais novos. Ele deve permanecer intacto até a aprovação do ambiente substituto.

### 10.2 Comandos

O ambiente definitivo deve ser criado fora da árvore versionada. Na máquina de referência, `uv` substitui o `conda`, que não está disponível no `PATH`:

```powershell
uv venv laboratorio/cibir_gpu --python 3.10
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe -r requirements/experiment-gpu.lock.txt
laboratorio/cibir_gpu/Scripts/python.exe scripts/validate_environment.py --require-cuda
```

Os dois primeiros comandos podem demorar e baixar vários gigabytes. Eles integram a EX-007 e devem ser executados por Artur. O terceiro comando grava `outputs/environment/EX-007_environment_report.json` e somente aprova o ambiente se um cálculo real funcionar tanto no PyTorch quanto no ONNX Runtime com CUDA.

Não se deve apagar nem modificar `laboratorio/cibir_face_new` até a aprovação da EX-007. Depois disso, a sequência reservada é:

| ID | Execução | Situação |
|---|---|---|
| EX-007 | criar, validar e congelar o ambiente CUDA | preparada; aguarda execução por Artur; |
| EX-008 | recriar o índice facial LFW | pendente; |
| EX-009 | avaliar a busca facial LFW | pendente; |
| EX-010 | recriar o índice facial Gallagher | pendente; |
| EX-011 | regenerar consultas e relevância Gallagher | pendente; |
| EX-012 | avaliar a busca facial Gallagher | pendente; |
| EX-013 | recriar o índice global Holidays | pendente; |
| EX-014 | avaliar a busca global Holidays | pendente. |

A fusão não recebe ainda um identificador de execução final. Primeiro deve ser decidido se ela usará uma fotografia completa como consulta para as duas modalidades ou caminhos distintos para o rosto e para o contexto global. Executar a fusão antes dessa decisão produziria um resultado difícil de interpretar.
