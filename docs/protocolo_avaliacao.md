# Protocolo de avaliação do sistema de recuperação fotográfica

Data de referência: 22 de setembro de 2026.

## 1. Situação

Este documento define o protocolo vigente. Os resultados Gallagher anteriores à correção A01/A02 permanecem históricos e não podem ser usados como números finais. A evidência Gallagher atual é a sequência EX-033/EX-034/EX-035, congelada em 22/09/2026.

O código está preparado para busca facial, busca global e fusão. Os testes sintéticos validam regras isoladas; os manifests, hashes e CSVs da EX-033/034/035 registram a execução completa corrigida. Isso não substitui revisão humana, decisão ética ou alinhamento institucional.

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

Na EX-035 Gallagher, a exclusão efetiva é por `source_image_id` e por eventuais `exclude_image_ids` explícitos; esse avaliador não aplica exclusão adicional automática por SHA-256. A passagem de 589 para 588 candidatas corresponde à fonte excluída em cada consulta.

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

O arquivo `tests/test_gallagher_protocol.py` acrescenta regressões para:

- manter no gabarito uma fotografia anotada mesmo quando o detector não produz face;
- selecionar a detecção que contém o ponto médio dos olhos anotados, sem preferir um vizinho maior;
- falhar quando nenhuma caixa contém o alvo, sem fallback para outro rosto.

O terminal preservado da rodada corretiva registra 3/3 regressões Gallagher, 19/19 testes metodológicos e 35/35 testes na suíte completa. Esse registro é evidência de execução anterior; o fechamento documental de 22/09/2026 fez somente revisão estática e não reexecutou os testes.

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
| EX-013 | recriar o índice global Holidays | concluída e aprovada; 1.491 imagens e 1.491 descritores; |
| EX-014 | avaliar a busca global Holidays | concluída e aprovada; 500 consultas; |
| EX-015 | recriar o índice global Gallagher | concluída e aprovada; 589 imagens e 589 descritores; |
| EX-016 | gerar o protocolo pareado Gallagher | concluída e aprovada; 20 consultas; |
| EX-017 | comparar face, global contextual e fusão | concluída e aprovada; cinco configurações; |
| EX-018 | analisar erros e casos da fusão pareada | concluída e aprovada; três casos selecionados; |

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

**Histórico, substituído pela EX-033.**

As anotações oficiais Gallagher foram cruzadas com as 587 fotografias efetivamente presentes no índice da EX-010. As identidades foram ordenadas pela quantidade decrescente de fotografias anotadas e, em caso de empate, de forma estável. Dentro de cada identidade, os candidatos foram ordenados por nome da imagem e índice da face.

Foram geradas 20 consultas de 20 identidades e 882 relações de relevância. Cada consulta é um recorte construído a partir das coordenadas dos olhos nas anotações oficiais, validado novamente pelo detector facial e associado a `source_image_id`. Três candidatos foram rejeitados por ausência de face detectável no recorte; a seleção avançou para os candidatos seguintes sem reduzir o total de consultas. Todos os 20 recortes possuem tamanho e SHA-256 registrados em `data/evaluation/gallagher_query_crops_manifest.csv`.

### 10.6 Resultado auditado da EX-012

**Histórico, substituído pela comparação comum da EX-035.**

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

### 10.8 Resultado auditado da EX-013

A execução no commit `4400961` examinou as 1.491 imagens em 2 min 03 s, com taxa observada de 12,09 imagens/s. Foram produzidos 1.491 descritores globais sem falha. A matriz possui forma `(1491, 2048)`, tipo `float32`, somente valores finitos e erro máximo de norma L2 igual a `1,192092896 × 10^-7`.

Os metadados contêm 1.491 `image_id` e caminhos únicos, mapeamento contínuo entre linhas e descritores e SHA-256 válido para cada imagem. O manifesto registrou árvore Git limpa, os parâmetros CUDA, lote 32 e quatro trabalhadores de CPU. Uma segunda validação independente confirmou os hashes dos arquivos de descritores e metadados.

### 10.9 Protocolo congelado da EX-014

O protocolo local contém 500 consultas e 991 relações de relevância. A primeira imagem de cada grupo Holidays é a consulta e as demais imagens do mesmo grupo são relevantes. Como todas as consultas já pertencem ao índice aprovado na EX-013, o avaliador reutiliza o descritor global congelado correspondente, exclui explicitamente a fotografia-fonte e ordena integralmente as 1.490 imagens elegíveis. Somente os dez primeiros resultados são armazenados.

A multiplicação entre consultas e galeria é executada na GPU em lotes de 128. O `query_time_ms` mede somente esse backend de similaridade, exclusão e ordenação com descritores já calculados; portanto, não representa latência ponta a ponta de uma nova imagem enviada pelo usuário. Precision@5, Precision@10, Recall@5, Recall@10 e AP usam o ranking integral e a relevância oficial.

Em uma verificação com as primeiras 20 consultas, o avaliador otimizado e o avaliador genérico produziram métricas idênticas e o mesmo Top-10 em todas as consultas. A diferença máxima entre scores foi `0,001366`, decorrente da extração CUDA em lotes distintos, sem alteração da ordenação observada. A fotografia-fonte não apareceu nos resultados.

### 10.10 Resultado auxiliar da EX-014

A avaliação no commit `c6af569` executou as 500 consultas sem falhas. Cada ranking contém as 1.490 imagens elegíveis depois da exclusão da fonte; foram salvos dez resultados por consulta. Não houve fotografia-fonte, repetição de `image_id` ou valor não finito. O manifesto registrou árvore Git limpa e hashes compatíveis com todos os CSVs e com a figura qualitativa, que também passou por inspeção visual.

Esta é uma avaliação auxiliar do componente global. O projeto calcula AP como média das precisões nas posições relevantes, enquanto a integração oficial Holidays usa outra forma de integração da curva precisão-recall. Portanto, `mAP = 0,842612` é uma métrica local adaptada e não deve ser comparada diretamente ao valor oficial do benchmark. Nenhuma nova avaliação Holidays foi executada no fechamento A01/A02.

| Métrica | Resultado |
|---|---:|
| Precision@5 | 0,324000 |
| Precision@10 | 0,177600 |
| Recall@5 | 0,870215 |
| Recall@10 | 0,917563 |
| mAP | 0,842612 |
| Tempo médio do backend por consulta | 0,704 ms |

A AP mínima foi 0,004215, a mediana foi 1,0 e a máxima foi 1,0; nenhuma consulta teve AP zero. Cada consulta possui entre uma e doze imagens relevantes, com mediana igual a uma. Isso explica por que Precision@5 e Precision@10 são numericamente baixas apesar do Recall e do mAP elevados: depois de recuperar a única imagem relevante, as posições restantes até K contam como não relevantes. O tempo informado mede somente o ranking com descritores pré-calculados, não a extração ponta a ponta de uma imagem nova.

### 10.11 Protocolo definido para a fusão Gallagher

**Histórico, implementado originalmente nas EX-016/EX-017 e substituído pela sequência corrigida EX-033/EX-034/EX-035 abaixo.**

Cada item será uma única consulta lógica com dois caminhos explicitamente relacionados:

- o recorte validado na EX-011 representa a pessoa-alvo e produz o descritor facial;
- a fotografia-fonte completa representa o contexto visual e produz o descritor global;
- `source_image_id` liga os dois caminhos e exclui a fotografia-fonte de todas as modalidades;
- o conjunto de relevância continua sendo a presença da pessoa-alvo segundo as anotações oficiais.

A EX-015 recriou os 589 descritores globais Gallagher com a mesma ResNet50 e o mesmo ambiente da EX-013. A EX-016 produzirá um CSV pareado contendo `face_query_path`, `global_query_path`, identidade-alvo e fonte, com hashes e manifesto.

A EX-017 comparará, no mesmo protocolo e universo de candidatas, os baselines `1,0/0,0` (somente face) e `0,0/1,0` (somente contexto global) e as combinações `0,9/0,1`, `0,7/0,3` e `0,5/0,5`. Os cossenos serão convertidos para `[0,1]` antes da soma ponderada. As cinco configurações serão relatadas como análise de sensibilidade; nenhuma será escolhida retrospectivamente como única configuração vencedora.

Esse desenho responde a uma pergunta específica: o contexto da fotografia-fonte ajuda a recuperar outras fotografias que contêm a pessoa selecionada? Ele não equivale a buscar identidade somente pela cena e não constitui validação agro.

### 10.12 Resultado auditado da EX-015

A execução no commit `d300a6e` examinou as 589 fotografias Gallagher em 44 s, com taxa observada de 13,22 imagens/s. Foram produzidos 589 descritores globais, sem falhas, em uma matriz `(589, 2048)` de tipo `float32`. Todos os vetores são finitos e normalizados, com erro máximo de norma L2 igual a `1,192092896 × 10^-7`.

Os metadados possuem 589 identificadores únicos, mapeamento contínuo para a matriz e SHA-256 válido. Os identificadores são compatíveis com o índice facial: as 587 fotografias com face detectada formam um subconjunto do índice global, que também inclui as duas fotografias sem face detectável. O manifesto registrou árvore Git limpa, CUDA, lote 32, quatro trabalhadores de CPU e hashes válidos.

### 10.13 Resultado auditado da EX-016

**Histórico, substituído pela EX-034.**

O protocolo pareado foi gerado no commit `baf5f83` a partir das 20 consultas congeladas na EX-011. Ele contém 20 identidades-alvo, 20 recortes faciais e 20 fotografias completas, ligadas por `source_image_id`. Existem 19 fotografias-fonte únicas porque uma fotografia contém duas pessoas selecionadas como alvos distintos; isso é esperado em um acervo multi-rosto.

Cada recorte e cada fotografia completa possui SHA-256 conferido contra o arquivo local. Todas as fontes existem nos índices facial e global, todas as consultas possuem relevância e nenhuma fonte permanece no gabarito. O manifesto registrou árvore Git limpa e hashes válidos para o CSV pareado e seu inventário de evidências.

### 10.14 Resultado auditado da EX-017

**Histórico, substituído pela EX-035.**

A execução no commit `532bb84` avaliou as 20 consultas nas cinco configurações predefinidas, sempre com 588 fotografias elegíveis após excluir a fonte. Foram produzidos 100 rankings integrais e salvos 1.000 resultados. Não houve falha, fotografia-fonte, duplicidade ou valor não finito. O baseline somente facial reproduziu exatamente o Top-10 e todas as métricas da EX-012.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,913949 |
| 0,0 / 1,0 | 0,260000 | 0,195000 | 0,203513 | 0,234716 | 0,280833 |
| 0,9 / 0,1 | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,908721 |
| 0,7 / 0,3 | 0,820000 | 0,640000 | 0,533433 | 0,657866 | 0,865268 |
| 0,5 / 0,5 | 0,750000 | 0,595000 | 0,473894 | 0,609914 | 0,778968 |

O contexto global isolado ficou acima do experimento histórico que usava incorretamente o recorte facial como consulta global, mas permaneceu muito abaixo do reconhecimento facial na tarefa de recuperar uma pessoa. A fusão `0,9/0,1` manteve as métricas Top-K do baseline, porém reduziu o mAP em `0,005228`. Em AP por consulta, ela melhorou uma consulta, empatou nove e piorou dez. As fusões com maior peso global degradaram também as métricas Top-K.

Portanto, este conjunto não fornece evidência de que a soma linear com contexto global melhore, em média, a recuperação por identidade. Ele mostra que o contexto pode ajudar casos isolados, mas seu efeito não é consistente. A figura qualitativa de `0,7/0,3` foi inspecionada e apresentou coerência visual com pessoa e cenário; isso não altera a conclusão quantitativa. O tempo médio de extração facial foi `966,050 ms`; os tempos inferiores a 4 ms registrados para o backend medem apenas fusão e ordenação com descritores pré-calculados.

### 10.15 Resultado auditado da EX-018

**Histórico e derivado da EX-017; não fornece os números Gallagher vigentes.**

A análise no commit `8cb1e12` rotulou os 1.000 resultados Top-10 da EX-017 e comparou a AP de cada método com o baseline facial. Para a fusão principal `0,9/0,1`, uma consulta melhorou, nove permaneceram inalteradas e dez pioraram.

Os casos foram selecionados por regra determinística antes da inspeção visual:

- maior melhora: `gallagher_id27_q1`, AP de 0,950000 para 1,000000;
- exemplo inalterado: `gallagher_id12_q1`, AP igual a 1,000000;
- maior degradação: `gallagher_id19_q1`, AP de 0,950000 para 0,887500.

Na maior melhora, o contexto reorganizou fotografias do mesmo ambiente sem alterar quais dez imagens apareciam no Top-10. Na maior degradação, a semelhança do cenário e da composição de grupo promoveu uma imagem incorreta, ilustrando como contexto pode competir com identidade. As quatro figuras foram inspecionadas, mas permanecem somente em `outputs/` porque contêm pessoas identificáveis. O manifesto registrou árvore Git limpa, hashes válidos e os totais esperados para todos os métodos.

### 10.15-A Protocolo Gallagher corrigido — EX-033 e EX-034

A EX-033 constrói a relevância a partir das anotações cruzadas com as 589 fotografias da galeria global da EX-015, sem usar o sucesso do detector como filtro. As 587 fotografias do índice facial são subconjunto dessa galeria. As duas fotografias anotadas sem face detectada permanecem no gabarito e recebem contribuição facial zero durante a fusão.

Foram congeladas 20 consultas de 20 identidades e 884 relações de relevância. Cada consulta usa um recorte baseado nos olhos anotados e seleciona exclusivamente uma caixa que contém o ponto médio desses olhos. Se nenhuma detecção contém o ponto, o candidato é rejeitado; não há fallback para o maior rosto. Quando mais de uma caixa contém o ponto, o desempate usa distância do centro, menor área e coordenadas da caixa.

A EX-034 emparelha os 20 recortes com as fotografias-fonte completas. Existem 19 fontes únicas porque uma fotografia origina consultas de duas identidades. `source_image_id` é excluído tanto do ranking quanto da relevância. Todas as consultas usam os mesmos CSVs congelados e possuem hashes dos arquivos consultados.

Manifests:

- EX-033: commit `b0abf4c0777cd0aa6fce34b96709e43043c3eb7c`, `dirty=false`, SHA-256 `8066afb36f95545c4bea1a0b996a0d40acaef6d801fd8ed75bcf4b8aca8b4222`;
- EX-034: commit `938911b5ffa117a4341bf35d7e8665e74a2ac49b`, `dirty=false`, SHA-256 `d448900a1300aaf6eed730d66f3c90bec5ebab08e96126c75f34a962cabc513d`.

### 10.15-B Resultado Gallagher vigente — EX-035

A EX-035 executa, na mesma matriz de consultas, galeria e relevância, os baselines `1,0/0,0` e `0,0/1,0` e as fusões `0,9/0,1`, `0,7/0,3` e `0,5/0,5`. Os cossenos são convertidos para `[0,1]`; a contribuição facial é zero quando a fotografia não possui face; a ordenação usa score decrescente e `image_id` crescente como desempate.

Foram verificadas 20 consultas por método, 588 candidatas após excluir a fonte, 100 linhas de métricas por consulta e 1.000 linhas Top-10, sem fonte, duplicidade ou valor não finito. Todos os pontos anotados ficaram dentro das caixas selecionadas, e as caixas da avaliação coincidem com as congeladas na EX-033.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,951446 |
| 0,0 / 1,0 | 0,260000 | 0,195000 | 0,203512 | 0,234712 | 0,280864 |
| 0,9 / 0,1 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,946297 |
| 0,7 / 0,3 | 0,840000 | 0,675000 | 0,540571 | 0,682856 | 0,901525 |
| 0,5 / 0,5 | 0,770000 | 0,630000 | 0,481033 | 0,634905 | 0,813358 |

O manifest EX-035 registra o commit `938911b5ffa117a4341bf35d7e8665e74a2ac49b`, `dirty=false` e SHA-256 `a7ee9c119246b2c9b926376c4435ea24cabf72f7496095f74fdc729f60dbb1c8`. Nenhuma fusão superou o baseline facial no mAP agregado. Essa conclusão se limita às 20 consultas selecionadas e não constitui validação agro.

### 10.16 Protocolo Agrishow congelado

**Histórico/complementar privado, fora do núcleo e da conclusão principal do artigo delimitado.** Sua eventual divulgação depende de decisão ética, institucional e de direitos pessoais. Os resultados são preservados abaixo sem serem usados para sustentar generalização rural ou agrícola.

A Agrishow 2022 foi tratada como estudo de caso aplicado, não como benchmark geral. Foram preservados 124 originais em CC BY 2.0 depois da remoção de três derivados recortados. Antes de qualquer busca, 91 imagens foram marcadas como presença da pessoa-alvo, 33 como ausência e nenhuma permaneceu incerta. Uma única fonte foi sorteada entre as 91 imagens presentes por regra SHA-256 com semente registrada. O recorte facial foi confirmado visualmente e a fotografia completa correspondente foi congelada como consulta global.

A mesma fonte é excluída de todos os métodos. Os cinco pesos são os já definidos para o Gallagher: somente face, somente contexto e fusões 0,9/0,1, 0,7/0,3 e 0,5/0,5. Nenhum peso ou rótulo foi alterado depois da observação dos rankings.

### 10.17 Resultado auditado da EX-022

A indexação no commit `6aa8dad` examinou os 124 originais. O índice facial produziu 1.590 embeddings de 512 dimensões em 123 fotografias; houve uma imagem sem face, nenhuma falha de leitura e nenhuma face sem embedding. Essa imagem estava anotada como ausente, então todas as 91 fotografias relevantes possuíam ao menos uma face detectada. O índice global produziu 124 descritores de 2.048 dimensões, sem falhas.

As duas matrizes são `float32`, finitas e normalizadas, com erro máximo de norma L2 igual a `1,192092896 × 10^-7`. Identificadores, caminhos e hashes coincidem com o inventário; a fonte sorteada está nos dois índices. Os manifestos registraram CUDA, árvore Git limpa e o mesmo commit.

### 10.18 Resultado auditado da EX-023

A avaliação no commit `ecc37f7` usou uma consulta, 123 candidatas depois da exclusão da fonte e 90 imagens relevantes. Todos os cinco rankings possuem 123 itens, e os dez primeiros resultados armazenados não contêm a fonte, duplicidades ou valores não finitos.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| 0,0 / 1,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,797298 |
| 0,9 / 0,1 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| 0,7 / 0,3 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997898 |
| 0,5 / 0,5 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,993485 |

Todos os métodos acertaram as dez primeiras posições. A diferença de Recall decorre apenas do denominador de 90 relevantes. Face e fusão 0,9/0,1 empataram em mAP; as demais fusões ficaram ligeiramente abaixo, e contexto isolado obteve 0,797298. O resultado sustenta o funcionamento no evento, na pessoa e na consulta avaliados, mas não autoriza generalização para outras pessoas, eventos, máquinas ou condições do agronegócio.

### 10.19 Resultado auditado da EX-024

A análise no commit `1bf4a95` recalculou 615 posições — 123 para cada um dos cinco métodos — e reproduziu exatamente os Top-10 e as APs congeladas na EX-023. A fonte permaneceu ausente, cada ranking contém 90 relevantes e não houve duplicidade.

| Método | Primeira irrelevante | Última relevante | Irrelevantes antes da última relevante | Sobreposição Top-10 com face |
|---|---:|---:|---:|---:|
| Face | 90 | 107 | 17 | 10 |
| Contexto global | 17 | 123 | 33 | 4 |
| Fusão 0,9/0,1 | 90 | 107 | 17 | 9 |
| Fusão 0,7/0,3 | 90 | 111 | 21 | 7 |
| Fusão 0,5/0,5 | 82 | 112 | 22 | 5 |

O contexto global promoveu mais cedo fotografias sem a pessoa-alvo, apesar de manter os dez primeiros acertos. A fusão 0,9/0,1 alterou uma posição do Top-10, mas preservou a AP facial; pesos globais maiores anteciparam erros e empurraram as últimas imagens relevantes para posições posteriores.

A figura qualitativa contém o recorte facial e as cinco primeiras fotografias da busca facial, todas relevantes. As caixas dos rostos foram verificadas visualmente e apontam para a pessoa-alvo nos cinco painéis. O arquivo tem 3.680 × 1.660 pixels e é acompanhado de seis atribuições individuais CC BY 2.0, incluindo fotógrafo, fonte, página original e modificação aplicada.

### 10.20 Controle aleatório da EX-025

A EX-025 quantificou o efeito da alta prevalência de relevância na consulta inicial da Agrishow. Com 90 imagens relevantes entre 123 candidatas, a esperança aleatória exata de AP, Precision@5 e Precision@10 é `90/123 = 0,731707` para Precision e `0,741369` para AP. Foram geradas 200.000 permutações uniformes com a semente `12031681346522691727`, sem usar escores dos modelos para produzir os rankings aleatórios.

A AP simulada teve média 0,741524, desvio-padrão 0,037454 e percentis 2,5% e 97,5% iguais a 0,670118 e 0,815768. A AP facial de 0,998235 não foi atingida em nenhuma permutação; o valor unilateral suavizado foi `4,999975 × 10^-6`. A AP do contexto global, 0,797298, foi igualada ou superada em 14.348 permutações, com valor unilateral suavizado de 0,071745.

A probabilidade hipergeométrica exata de Precision@5 igual a 1 em um ranking aleatório é 0,203402, e a de Precision@10 igual a 1 é 0,038133. Portanto, resultados perfeitos no Top-K precisam ser acompanhados pela AP integral e não devem ser usados isoladamente como prova de desempenho.

### 10.21 Robustez em dez consultas — EX-026

A consulta original foi mantida e nove fontes adicionais foram selecionadas entre as fotografias relevantes pela menor chave SHA-256 formada com a semente `1161755629553718132`, o alvo e o identificador da imagem. A seleção ocorreu antes da recuperação. Na revisão facial prévia, a fonte inicialmente destinada à consulta 5, `agrishow2022_52029759375`, foi recusada porque a identidade não podia ser confirmada na revisão numerada. Sem inspecionar resultados de recuperação, foi usada a próxima fonte ainda não selecionada na ordem SHA-256 congelada, `agrishow2022_52029029111`. A decisão, a justificativa e a ausência de consulta aos rankings estão registradas no manifesto de emenda e no CSV de elegibilidade.

O protocolo final possui dez fontes únicas, 124 rótulos por consulta e 90 relevantes elegíveis depois da exclusão da própria fonte. Os recortes-alvo foram confirmados visualmente e congelados antes da avaliação. A execução reutilizou os índices aprovados na EX-022, avaliou 50 rankings integrais e salvou 500 itens Top-10. Não houve vazamento da fonte, duplicidade, falha ou valor não finito.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,987261 |
| 0,0 / 1,0 | 0,940000 | 0,920000 | 0,052222 | 0,102222 | 0,850717 |
| 0,9 / 0,1 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,986167 |
| 0,7 / 0,3 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,980514 |
| 0,5 / 0,5 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,960858 |

A AP facial média foi 0,987261, com desvio-padrão amostral 0,029403, mínimo 0,903834 e mediana 0,997053. O pior caso foi a consulta 6, que também obteve Precision@5 e Precision@10 iguais a 0,8. Em comparação com a face, o contexto global degradou as dez consultas; a fusão 0,9/0,1 melhorou uma, empatou duas e degradou sete; as fusões 0,7/0,3 e 0,5/0,5 melhoraram uma e degradaram nove. Nenhuma fusão superou a face em mAP médio.

### 10.22 Caso dirigido com chapéu e sombra — EX-027

A EX-027 avaliou separadamente a fotografia `agrishow2022_52030041698`, já anotada na EX-020 como um caso em que o chapéu projeta sombra sobre o rosto. Essa fonte não foi sorteada e, por isso, o resultado não foi misturado às dez consultas de robustez. A seleção ocorreu por interesse metodológico antes da recuperação. Entre 18 rostos detectados, a pessoa-alvo foi confirmada visualmente no índice 1; o recorte congelado foi redetectado com IoU de 0,709009.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997102 |
| 0,0 / 1,0 | 1,000000 | 0,800000 | 0,055556 | 0,088889 | 0,826134 |
| 0,9 / 0,1 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997506 |
| 0,7 / 0,3 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997749 |
| 0,5 / 0,5 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,988609 |

Nesse caso específico, a fusão 0,7/0,3 superou a face em 0,000647 de AP. O ganho não deve ser generalizado nem usado para escolher retrospectivamente um peso, pois a avaliação agregada da EX-026 continua favorecendo o baseline facial.

## 11. Avaliação do detector na WIDER FACE — EX-029 a EX-032

### 11.1 Escopo e fonte congelados — EX-029

A avaliação usa exclusivamente as 3.226 imagens da divisão oficial de validação da WIDER FACE. Treino e teste são proibidos. A página oficial do Multimedia Laboratory da Chinese University of Hong Kong registra a licença Creative Commons BY-NC-ND, os links separados para as imagens de validação, as anotações de caixas e o pacote de avaliação. URLs, tamanhos e hashes foram congelados em `data/evaluation/widerface_sources.json` antes do download integral.

O arquivo `WIDER_val.zip` deve possuir 362.752.168 bytes e SHA-256 `f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a`. O arquivo `wider_face_split.zip` deve possuir 3.591.642 bytes, MD5 `0e3767bcf0e326556d407bf5bff5d27c` e SHA-256 `c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c`. O pacote oficial `eval_tools.zip` deve possuir 8.447.003 bytes, MD5 `358576548629ca5dd6fc4b2de15b10ae` e SHA-256 `1cf49c8243fa8a1632efe7f6aa09fd7a3994ac838c9008bdbc9dd1d547bb234a`.

Antes da extração, o executor rejeita hash, tamanho, membro inseguro e qualquer marcador de treino ou teste. Depois da extração, exige exatamente 3.226 imagens, 3.226 blocos de anotação, correspondência integral de nomes e os quatro arquivos oficiais de ground truth: geral, Easy, Medium e Hard.

### 11.2 Detector e inferência congelados — EX-030

O modelo é exclusivamente `buffalo_l/det_10g.onnx`, com SHA-256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`. Embora o fluxo anterior o descrevesse genericamente como RetinaFace, o InsightFace 0.2.1 identifica tecnicamente esse arquivo como `insightface.model_zoo.scrfd.SCRFD`. O experimento registra essa classe explicitamente para não confundir detecção com reconhecimento.

Somente o grafo do detector é carregado. ArcFace não é carregado, nenhuma rotina de reconhecimento é executada e nenhum embedding é produzido. A sessão ONNX Runtime deve ter `CUDAExecutionProvider` como primeiro provedor ativo; uma sessão apenas em CPU reprova a execução.

As decisões congeladas antes dos resultados são:

- imagens em resolução original;
- entrada do detector de 640 × 640;
- `max_num=0`;
- piso de score 0,02;
- limiar operacional 0,50;
- IoU mínimo 0,50;
- uma passagem integral salva caixas e scores a partir de 0,02;
- as métricas no limiar 0,50 são derivadas dessa mesma saída, sem repetir a inferência integral.

Antes da passagem completa, uma amostra fixa de 20 imagens compara as caixas e scores obtidos diretamente em 0,50 com a filtragem, em 0,50, da saída produzida com piso 0,02. A tolerância é `1e-4` para caixas e `1e-6` para scores. Há ainda um único smoke test técnico de uma imagem; ele não constitui piloto experimental e nenhum resultado é usado para ajustar modelo, resolução ou limiar.

### 11.3 Métricas e auditoria congeladas — EX-031

AP Easy, Medium e Hard seguem o protocolo oficial WIDER de 1.000 limiares, normalização global de score, ground truths ignorados e integração VOC da curva precisão-recall. No limiar operacional 0,50 são calculados precisão, recall e F1 sobre todas as faces válidas. O tempo médio e a mediana por imagem medem apenas a chamada de inferência do detector.

O recall descritivo é estratificado pelos atributos oficiais de desfoque, iluminação, oclusão e pose. O tamanho usa a definição do artigo WIDER pela altura da face: pequena entre 10 e 50 pixels, média entre 50 e 300 pixels e grande acima de 300 pixels; faces abaixo de 10 pixels são relatadas separadamente.

Os testes sintéticos cobrem resultado perfeito, vazio, duplicado, caixa inválida e ground truth ignorado. Uma amostra fixa de 20 imagens repete as associações com uma implementação independente de IoU em `torchvision`; as caixas inclusivas do protocolo oficial são convertidas para a convenção exclusiva do `torchvision` antes da comparação. A execução também exige repetição determinística, valores finitos, caixas válidas e ausência de imagens ou anotações faltantes.

### 11.4 Comando completo reservado

| Campo | Valor |
|---|---|
| Identificador | EX-029/EX-030/EX-031 |
| Motivo | baixar somente a validação oficial, validar fonte e hashes, executar uma passagem do detector na GPU e produzir as métricas auditadas |
| Risco | download de aproximadamente 375 MB, uso temporário estimado de até 2 GB e execução prolongada da GPU; qualquer ausência de CUDA, divergência de hash ou presença de treino/teste interrompe o processo |
| Duração estimada | 30 a 60 minutos, dependente da rede, disco e RTX 3050 Ti |
| Resultado esperado | exatamente 3.226 imagens processadas, zero embeddings, AP Easy/Medium/Hard, métricas em 0,50, recalls por categoria, curva agregada, predições completas temporárias e manifesto auditável |

Executar uma única vez, a partir de `C:\Projeto_de_Pesquisa`, depois do commit de preparação:

```powershell
laboratorio\cibir_gpu\Scripts\python.exe scripts\face\08_run_widerface_validation.py
```

Não repetir com parâmetros diferentes depois de observar os resultados. Qualquer alteração de modelo, resolução, piso, limiar ou IoU deve receber outro identificador experimental.

### 11.5 Preservação e limpeza planejadas — EX-032

Após a auditoria, os resultados agregados serão incorporados aos documentos canônicos e ao dossiê privado. A WIDER mede robustez geral de detecção; não mede reconhecimento da mesma pessoa e não constitui validação agro. Nenhuma imagem, recorte ou caixa desenhada da WIDER será publicada. Agrishow permanece histórico/complementar privado; nenhuma figura com pessoas deve ser publicada sem resolução ética, institucional e de direitos pessoais.

Antes da limpeza, o dossiê será renderizado, inspecionado e sincronizado nas cópias local e OneDrive. A lista literal de exclusão será simulada e arquivada. Somente então poderão ser removidos imagens, arquivos compactados, anotações reproduzíveis, predições completas e caches WIDER, com registro do espaço recuperado e confirmação de preservação de LFW, Gallagher, Holidays, ambiente GPU e evidências pequenas.
