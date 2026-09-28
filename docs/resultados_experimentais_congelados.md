# Resultados experimentais vigentes

Data de referência: **28 de setembro de 2026**.

## Execução que fornece os números atuais

A execução atual é `outputs/validation_runs/fechamento_20260928_cuda/`, no commit
limpo `91a2a28360ded41fd86abf3371c989d571bf95bd`. Terminou com código de saída
zero e 13 manifestos consistentes. Foram conferidos tamanhos e hashes das 92
referências a arquivos e os SHA-256 de 15.265 fotografias-fonte indexadas.

Os resultados EX-035 e `arquitetura_v1` continuam preservados, mas não são a
fonte das tabelas atuais. A correção de seleção de CUDA alterou descritores
faciais e algumas métricas; não se deve misturar valores das duas execuções.

SHA-256 do manifesto final:
`5fc6acd8c42c75520c0eecefa5366f3a33e0627d217b0b0cf61dd556876f34c9`.

## Tabela 1 — Gallagher corrigido

As cinco configurações compartilham 20 consultas, 20 identidades, 19 fotografias-
fonte, 589 fotografias na galeria, 588 candidatas após excluir a fonte e 884
relações de relevância. As identidades 14 e 15 compartilham uma foto de consulta.

| Método | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| Somente face | 0,900000 | 0,680000 | 0,577119 | 0,687856 | **0,952261** |
| Somente contexto global | 0,260000 | 0,195000 | 0,203512 | 0,234712 | 0,280864 |
| Face/contexto 0,9/0,1 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,946274 |
| Face/contexto 0,7/0,3 | 0,840000 | 0,675000 | 0,540571 | 0,682856 | 0,901460 |
| Face/contexto 0,5/0,5 | 0,770000 | 0,630000 | 0,481033 | 0,634905 | 0,813297 |

Fonte: `gallagher/evaluation/fusion_metrics.csv`, sob a execução atual.
SHA-256: `de0fcdd33c092b4c774521cb6bca65a4b080e5a15e9a719352dbee71f585335c`.
As médias foram recalculadas independentemente a partir das 100 linhas por consulta.

**Conclusão delimitada:** nenhuma fusão testada superou a busca facial no mAP
agregado deste protocolo. Isso não demonstra que contexto nunca ajuda, que
qualquer método de fusão é inferior ou que a causa da queda seja o fundo da foto.

## Tabelas 2 e 3 — comparação pareada

Uma comparação pareada acompanha a mesma consulta nos diferentes métodos.
Delta AP = AP do método menos AP facial. Valor positivo indica ganho;
negativo, perda. Empates usam tolerância de `1e-12`, antes do arredondamento.

| Método contra face | Ganhos | Empates | Perdas | Delta AP médio |
|---|---:|---:|---:|---:|
| Contexto global | 0 | 1 | 19 | -0,671397 |
| Fusão 0,9/0,1 | 2 | 9 | 9 | -0,005987 |
| Fusão 0,7/0,3 | 1 | 7 | 12 | -0,050801 |
| Fusão 0,5/0,5 | 0 | 4 | 16 | -0,138964 |

| Consulta | AP facial | Delta 0,9/0,1 | Delta 0,7/0,3 | Delta 0,5/0,5 |
|---|---:|---:|---:|---:|
| gallagher_id2_q1 | 0,964520 | -0,005616 | -0,028925 | -0,103393 |
| gallagher_id3_q1 | 0,898722 | +0,000616 | -0,018859 | -0,080399 |
| gallagher_id1_q1 | 0,980313 | -0,004089 | -0,016533 | -0,051865 |
| gallagher_id8_q1 | 0,956172 | -0,002093 | -0,020961 | -0,057799 |
| gallagher_id9_q1 | 1,000000 | 0,000000 | 0,000000 | -0,027093 |
| gallagher_id5_q1 | 0,992823 | 0,000000 | 0,000000 | -0,004746 |
| gallagher_id4_q1 | 0,882458 | -0,014572 | -0,095096 | -0,303908 |
| gallagher_id6_q1 | 0,933817 | 0,000000 | -0,029867 | -0,070833 |
| gallagher_id15_q1 | 0,889112 | -0,004167 | -0,071340 | -0,244683 |
| gallagher_id23_q1 | 0,986111 | 0,000000 | -0,015625 | -0,102927 |
| gallagher_id12_q1 | 1,000000 | 0,000000 | 0,000000 | 0,000000 |
| gallagher_id25_q1 | 0,768315 | -0,009471 | -0,082503 | -0,231488 |
| gallagher_id21_q1 | 0,976190 | -0,017857 | -0,065476 | -0,199863 |
| gallagher_id7_q1 | 1,000000 | 0,000000 | 0,000000 | 0,000000 |
| gallagher_id14_q1 | 1,000000 | -0,050000 | -0,350000 | -0,776474 |
| gallagher_id19_q1 | 0,950000 | -0,062500 | -0,270833 | -0,270833 |
| gallagher_id27_q1 | 0,950000 | +0,050000 | +0,050000 | -0,145833 |
| gallagher_id32_q1 | 1,000000 | 0,000000 | 0,000000 | -0,107143 |
| gallagher_id26_q1 | 0,916667 | 0,000000 | 0,000000 | 0,000000 |
| gallagher_id28_q1 | 1,000000 | 0,000000 | 0,000000 | 0,000000 |

Fonte: `gallagher/evaluation/fusion_metrics_per_query.csv`.
SHA-256: `2cd2d25e06e6a938e4071d0a024c7510688720ebbd95d61fd5f37879161848b7`.
Os deltas foram recalculados e confrontados com `gallagher/error_analysis/method_deltas.csv`.
A ordem da tabela acompanha o protocolo, não seleciona somente casos favoráveis.

Na fusão 0,9/0,1, o maior ganho foi +0,05 (consulta 27) e a maior perda,
-0,0625 (consulta 19). A mediana do delta foi zero. P@5 e P@10 médios iguais
aos do baseline não significam rankings idênticos ou AP idêntica.

Esta é uma análise descritiva completa das consultas, não um teste de
significância nem uma demonstração populacional. As consultas compartilham
álbum e galeria, e duas compartilham a foto-fonte. Não foram estimados intervalos
de confiança, alegada independência ou escolhidos pesos para maximizar o teste.

## Tabela 4 — verificações auxiliares, tarefas distintas

| Componente | Base | Consultas | mAP local | Papel |
|---|---|---:|---:|---|
| Facial | LFW | 1.672 | 0,965136 | recuperação customizada condicionada às imagens com face indexada |
| Global | INRIA Holidays | 500 | 0,842612 | verificação de recuperação visual com AP adaptada |

Fontes sob a execução atual:

- `lfw/evaluation/face_metrics.csv`: SHA-256
  `78adcfc10a43385d920c30f2ed20a0dc38f8011f1a4f4daa3ebb006c63620ee9`.
- `holidays/evaluation/global_metrics.csv`: SHA-256
  `3d2bf5a415f83c140ff0b1ce6a90d0fdc3a33c58c8bad404a9116ddc5ef2ce1e`.

LFW não usa aqui o protocolo oficial de verificação em pares. Foram escaneadas
13.233 imagens; 48 sem face ficaram fora do índice. A galeria facial tem 13.185
fotografias, 16.058 rostos e 13.184 candidatas por consulta. Não se pode chamar
este mAP de acurácia oficial LFW nem assegurar ausência de sobreposição com o
pré-treinamento dos modelos.

Holidays possui 1.491 imagens locais e 1.490 candidatas por consulta. A AP do
projeto soma as precisões nas posições relevantes e divide pelo total de
relevantes. O avaliador oficial integra trapézios na curva precisão–recall:
por exemplo, um único relevante na segunda posição dá 0,5 aqui e 0,25 lá.
Portanto, este número **não é diretamente comparável ao mAP oficial Holidays**.

Há três pares byte a byte idênticos em grupos diferentes da cópia local:
103100/103900, 103101/103901 e 103102/103902. Não se determinou se surgiram na
distribuição original. Este protocolo exclui a fonte por ID/caminho, não todas
as cópias por hash; a busca interativa usa exclusão por SHA-256. Não são
protocolos universalmente equivalentes. Os avaliadores auxiliares usam
ordenação vetorizada do PyTorch; não se alega desempate universal entre
equipamentos. Nenhuma comparação externa ou avaliação deduplicada foi feita.

## Correções A01/A02 reconferidas

- O gabarito foi reconstruído independentemente das anotações e da galeria
  global e coincidiu com as 884 relações, sem duplicatas ou fontes.
- As duas fotos relevantes da identidade 2 sem detecção permanecem no gabarito.
- Em todas as 20 consultas, a caixa facial selecionada contém o ponto dos olhos
  anotados; não há fallback para o rosto vizinho.
- Precision/Recall@5/@10 foram recalculados a partir dos Top-10 e do gabarito.
- Todos os métodos têm o mesmo universo de 588 candidatas. AP usa o ranking
  integral; somente as dez primeiras posições são persistidas para inspeção.
- O protocolo delimita identidade **anotada**: as anotações Gallagher não são
  um registro exaustivo de qualquer presença humana na fotografia.

## Comparação com a reprodução de 24/09

| Método/base | mAP de referência | mAP atual | Delta |
|---|---:|---:|---:|
| Gallagher facial | 0,951446 | 0,952261 | +0,000815 |
| Gallagher global | 0,280864 | 0,280864 | 0,000000 |
| Gallagher 0,9/0,1 | 0,946297 | 0,946274 | -0,000024 |
| Gallagher 0,7/0,3 | 0,901525 | 0,901460 | -0,000065 |
| Gallagher 0,5/0,5 | 0,813358 | 0,813297 | -0,000061 |
| LFW facial | 0,965135 | 0,965136 | +0,000001 |
| Holidays global | 0,842612 | 0,842612 | 0,000000 |

As consultas, relevâncias e contagens foram preservadas. Os descritores globais
foram idênticos; os faciais mudaram com as sessões CUDA. AP mudou em 37/1.672
consultas LFW e em 4/20 consultas do baseline facial Gallagher. Precision e
Recall@5/@10 foram iguais aos anteriores em todas as configurações.

A maior diferença de componente de descritor foi 0,060445 em LFW e 0,004250 em
Gallagher. A primeira não deve ser escondida sob uma alegação de igualdade:
no caso extremo houve também mudança na caixa detectada. Um controle usando
o adaptador atual em CPU reproduziu exatamente os vetores antigos em duas
imagens, incluindo esse caso extremo. Isso sustenta a explicação da mudança
de executor, mas não é uma reprodução CPU completa adicional. Os pesos atuais
conferem com os hashes registrados na auditoria de 21/09; isso não prova
retroativamente os provedores usados em cada execução antiga.

A referência `arquitetura_v1` pertence ao commit `954d74959f42f7eda3765df49023e8af193ce1cd`;
EX-035 e seus manifests permanecem em
`docs/evidencias/gallagher_a01_a02_20260922_v2/`.
Os 113 arquivos de `outputs/final/` e `arquitetura_v1` foram preservados.

## Regras de interpretação e alteração

Sem alegação de larga escala, validação agropecuária, superioridade universal,
calibração estatística da fusão, causalidade de fundo ou acurácia familiar.
O álbum privado testa funcionamento, não alimenta estas tabelas.
Uma nova tabela só pode substituir esta execução após protocolo declarado,
Git limpo, hashes e manifestos conferidos e explicação das diferenças.
A aprovação científica final do texto continua sendo dos autores humanos.
