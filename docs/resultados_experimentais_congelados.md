# Resultados experimentais oficiais

Data de congelamento vigente: **22 de setembro de 2026**.

Este documento é a tabela canônica dos resultados reproduzidos no ambiente definitivo. Para Gallagher, a evidência vigente é a EX-035, baseada no protocolo corrigido EX-033/EX-034. EX-011, EX-012, EX-016, EX-017 e EX-018 permanecem históricos e não fornecem números finais.

## Ambiente comum

- Python 3.10.20;
- PyTorch 2.11.0 com CUDA 12.8;
- ONNX Runtime GPU 1.23.2;
- NVIDIA GeForce RTX 3050 Ti Laptop GPU;
- métricas calculadas sobre o ranking integral;
- Precision@K com denominador fixo K;
- fotografia-fonte excluída do ranking e da relevância;
- Top-10 usado somente como limite de armazenamento e apresentação.

## Tabela principal vigente

| Experimento | Base | Consultas | Candidatas por consulta | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Face por identidade | LFW | 1.672 | 13.184 | 0,457177 | 0,284629 | 0,901149 | 0,945308 | 0,965135 |
| Face somente | Gallagher corrigido | 20 | 588 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,951446 |
| Contexto global somente | Gallagher corrigido | 20 | 588 | 0,260000 | 0,195000 | 0,203512 | 0,234712 | 0,280864 |
| Face/contexto 0,9/0,1 | Gallagher corrigido | 20 | 588 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,946297 |
| Face/contexto 0,7/0,3 | Gallagher corrigido | 20 | 588 | 0,840000 | 0,675000 | 0,540571 | 0,682856 | 0,901525 |
| Face/contexto 0,5/0,5 | Gallagher corrigido | 20 | 588 | 0,770000 | 0,630000 | 0,481033 | 0,634905 | 0,813358 |
| CBIR global auxiliar, AP adaptada | Holidays | 500 | 1.490 | 0,324000 | 0,177600 | 0,870215 | 0,917563 | 0,842612 |

As cinco configurações Gallagher compartilham as mesmas 20 consultas, 884 relações de relevância, 589 fotografias no universo da galeria e 588 candidatas após excluir a fonte. As duas fotografias sem face detectada permanecem no gabarito; nelas, a contribuição facial é zero e o componente global continua disponível.

O valor Holidays usa a definição de AP implementada pelo projeto, diferente da integração do avaliador oficial. Ele permanece como avaliação auxiliar do componente global e não deve ser apresentado como diretamente comparável ao mAP oficial do benchmark. Nenhuma nova avaliação Holidays foi executada durante a correção A01/A02.

## O que cada linha responde

- **LFW:** o sistema recupera outras fotografias da mesma identidade em um baseline facial amplo?
- **Gallagher face:** o sistema encontra uma pessoa em fotografias com múltiplos rostos quando o alvo é ligado geometricamente à anotação?
- **Gallagher contexto:** a aparência da fotografia-fonte, sem usar o rosto, localiza outras fotografias que contêm a pessoa?
- **Gallagher fusão:** acrescentar contexto global ao sinal facial melhora a recuperação da pessoa sob o mesmo protocolo?
- **Holidays auxiliar:** a ResNet50 recupera fotografias visualmente semelhantes segundo a métrica adaptada do projeto?

As linhas não representam a mesma tarefa. Os valores de LFW, Gallagher e Holidays não devem ser comparados como se fossem métodos concorrentes no mesmo benchmark.

## Conclusões sustentadas

1. A busca facial apresentou desempenho elevado nos protocolos de identidade avaliados.
2. No Gallagher corrigido, face somente obteve o maior mAP, `0,951446`.
3. Contexto visual isolado não substitui reconhecimento facial para localizar uma pessoa.
4. Nenhum dos três pesos de fusão superou o baseline facial no mAP agregado.
5. A fusão 0,9/0,1 ficou próxima do baseline facial, `0,946297` contra `0,951446`, mas não o superou.
6. A conclusão Gallagher se limita a 20 identidades selecionadas, uma consulta por identidade e fotografias correlacionadas; não prova generalização entre álbuns ou eventos.
7. Holidays fornece somente evidência auxiliar do componente global com AP adaptada, não reprodução diretamente comparável ao benchmark oficial.
8. Gallagher e Holidays não constituem validação rural ou agrícola.

## Correção A01/A02

Na EX-033, a relevância passou a ser construída sobre todas as fotografias anotadas presentes na galeria global, sem depender de detecção facial. `100_2023.JPG` e `100_2024.JPG` permanecem relevantes para `gallagher_id2_q1`, elevando seu denominador de 299 para 301.

As consultas passaram a selecionar a caixa que contém o ponto médio dos olhos anotados, sem fallback para o maior rosto. Em `gallagher_id6_q1`, o alvo `(411,5; 370,0)` está na caixa `[299,3; 250,2; 542,2; 579,3]`; a AP facial passou do valor histórico `0,180849` para `0,9338169767`.

Essa verificação geométrica comprova consistência com a anotação. Ela não constitui reconhecimento humano independente nem amplia a população avaliada.

## Resultados Gallagher históricos substituídos

Os números abaixo são preservados para rastreabilidade, mas não devem ser reutilizados como resultados atuais.

| Evidência histórica | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| EX-012 — face, gabarito filtrado pelo detector | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,913949 |
| EX-017 — contexto global | 0,260000 | 0,195000 | 0,203513 | 0,234716 | 0,280833 |
| EX-017 — fusão 0,9/0,1 | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,908721 |
| EX-017 — fusão 0,7/0,3 | 0,820000 | 0,640000 | 0,533433 | 0,657866 | 0,865268 |
| EX-017 — fusão 0,5/0,5 | 0,750000 | 0,595000 | 0,473894 | 0,609914 | 0,778968 |

Essas execuções reutilizavam o gabarito com A01 e, para `gallagher_id6_q1`, o rosto vizinho de A02. EX-018 também permanece histórica porque deriva da EX-017.

Na análise histórica da EX-017/EX-018, a fusão 0,9/0,1 melhorou a AP de uma consulta, manteve nove e piorou dez. Esse perfil permanece útil para compreender o resultado antigo, mas não deve ser transferido para a EX-035 sem nova análise específica.

## Agrishow — histórico/complementar privado

Agrishow fica fora do núcleo do artigo e não sustenta a conclusão principal. Os resultados são preservados como estudo complementar privado de uma única pessoa e um único evento, pendente de resolução ética, institucional e de direitos pessoais. Eles não demonstram validação geral no domínio agro.

| Configuração | Consultas | Candidatas | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Face | 10 | 123 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,987261 |
| Contexto global | 10 | 123 | 0,940000 | 0,920000 | 0,052222 | 0,102222 | 0,850717 |
| Face/contexto 0,9/0,1 | 10 | 123 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,986167 |
| Face/contexto 0,7/0,3 | 10 | 123 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,980514 |
| Face/contexto 0,5/0,5 | 10 | 123 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,960858 |

### Controle aleatório histórico — EX-025

Com 90 relevantes em 123 candidatas, a esperança aleatória exata é 0,741369 para AP e 0,731707 para Precision@K. Em 200.000 permutações uniformes, a AP apresentou média 0,741524, desvio-padrão 0,037454 e intervalo empírico entre os percentis 2,5% e 97,5% de 0,670118 a 0,815768. A probabilidade exata de Precision@5 igual a 1 é 0,203402; para Precision@10 igual a 1, é 0,038133. A AP facial de 0,998235 não foi atingida nas permutações, enquanto a AP global de 0,797298 foi igualada ou superada em 14.348 delas.

### Caso dirigido de chapéu e sombra — EX-027

| Configuração face/global | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997102 |
| 0,0 / 1,0 | 1,000000 | 0,800000 | 0,055556 | 0,088889 | 0,826134 |
| 0,9 / 0,1 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997506 |
| 0,7 / 0,3 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997749 |
| 0,5 / 0,5 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,988609 |

O ganho de `0,000647` da fusão 0,7/0,3 sobre a face é específico desse caso e não altera a conclusão média histórica.

Nove fontes adicionais haviam sido escolhidas por regra pseudoaleatória determinística. Uma candidata foi recusada antes da recuperação porque a identidade-alvo não pôde ser confirmada; a próxima fonte elegível congelada foi usada sem inspeção dos rankings. Fotografias produzidas em sequência podem compartilhar cenário, enquadramento e participantes, portanto não são observações independentes. Figuras com pessoas permanecem apenas nos resultados locais.

## Limitações obrigatórias

- Gallagher usa 20 identidades entre as mais frequentes, uma consulta por identidade.
- As consultas e fotografias do mesmo acervo não equivalem a 20 álbuns independentes.
- LFW contém muitas identidades com somente uma fotografia relevante após excluir a consulta; isso eleva a frequência de AP igual a 1.
- Holidays possui mediana de uma imagem relevante por consulta e usa métrica adaptada local.
- Os tempos de backend com descritores pré-calculados não representam latência completa de uma imagem nova.
- Fotografias, recortes, embeddings e figuras com pessoas não integram o pacote documental seguro.
- As dez consultas Agrishow representam uma pessoa e um evento, com 90 relevantes entre 123 candidatas; alta prevalência e correlação limitam a interpretação.

## Evidência reprodutível

| Execução | Situação | Evidência principal | Commit de execução |
|---|---|---|---|
| EX-009 | vigente para LFW | `outputs/experiments/ex-009_lfw_face_evaluation` | `ac13b6f` |
| EX-012 | histórico Gallagher | `outputs/experiments/ex-012_gallagher_face_evaluation` | `66d41bc` |
| EX-014 | auxiliar Holidays | `outputs/experiments/ex-014_holidays_global_evaluation` | `c6af569` |
| EX-017 | histórico Gallagher | `outputs/experiments/ex-017_gallagher_paired_fusion` | `532bb84` |
| EX-018 | histórico Gallagher | `outputs/experiments/ex-018_paired_fusion_error_analysis` | `8cb1e12` |
| EX-033 | protocolo Gallagher vigente | `outputs/experiments/ex-033_gallagher_protocol_a01_a02_20260922_v2` | `b0abf4c` |
| EX-034 | protocolo pareado vigente | `outputs/experiments/ex-034_gallagher_paired_protocol_a01_a02_20260922_v2` | `938911b` |
| EX-035 | métricas Gallagher vigentes | `outputs/experiments/ex-035_gallagher_paired_fusion_a01_a02_20260922_v2` | `938911b` |
| EX-022 | histórico/complementar privado Agrishow | índices `ex-022_agrishow_face_index` e `ex-022_agrishow_global_index` | `6aa8dad` |
| EX-023 | histórico/complementar privado Agrishow | `outputs/experiments/ex-023_agrishow_paired_fusion` | `ecc37f7` |
| EX-024 | histórico/complementar privado Agrishow | `outputs/experiments/ex-024_agrishow_analysis` | `1bf4a95` |
| EX-025 | histórico/complementar privado Agrishow | `outputs/experiments/ex-025_agrishow_random_baseline` | `a466b03` |
| EX-026 | histórico/complementar privado Agrishow | protocolo, avaliação e análise em `outputs/experiments/ex-026_agrishow_*` | `c43d611`, `58fd076` |
| EX-027 | histórico/complementar privado Agrishow | protocolo e avaliação em `outputs/experiments/ex-027_agrishow_*` | `3d7c16c` |

Os outputs completos são locais e ignorados pelo Git. O pacote seguro e rastreável está em `docs/evidencias/gallagher_a01_a02_20260922_v2/`; ele preserva manifests e métricas, mas não fotografias, recortes, embeddings, pesos, datasets ou rankings Top-10 vinculáveis.

## Regra para alterações futuras

Estes números somente podem ser substituídos por nova execução identificada, protocolo documentado, manifesto de árvore limpa, hashes válidos e auditoria equivalente. Novas avaliações agro devem permanecer em seção separada e não podem reclassificar Gallagher como conjunto agro.
