# Evidências seguras — correção Gallagher A01/A02

Data da consolidação documental: **22/09/2026**.

Este diretório preserva o menor pacote textual necessário para rastrear o protocolo corrigido EX-033/EX-034 e as métricas EX-035. Ele não é um pacote público de dados, não autoriza redistribuição e não substitui decisões éticas, institucionais ou de licença.

## Escopo

- A01: relevância construída sobre as fotografias anotadas da galeria global, sem depender do sucesso do detector facial.
- A02: seleção da detecção que contém o ponto médio dos olhos anotados, sem fallback para o maior rosto.
- 20 consultas, 20 identidades, 884 relações de relevância.
- 589 fotografias no universo da galeria e 588 candidatas depois da exclusão da fonte.
- Cinco configurações avaliadas no mesmo protocolo.

## Arquivos preservados

| Arquivo neste diretório | Origem local ignorada pelo Git | Finalidade |
|---|---|---|
| `ex-033_gallagher_protocol_manifest.json` | `outputs/experiments/ex-033_gallagher_protocol_a01_a02_20260922_v2/gallagher_protocol_manifest.json` | configuração, entradas, resultados, ambiente e commit do gabarito corrigido |
| `ex-034_paired_protocol_manifest.json` | `outputs/experiments/ex-034_gallagher_paired_protocol_a01_a02_20260922_v2/paired_protocol_manifest.json` | configuração e hashes do emparelhamento face/contexto |
| `ex-035_fusion_run_manifest.json` | `outputs/experiments/ex-035_gallagher_paired_fusion_a01_a02_20260922_v2/fusion_run_manifest.json` | configuração comum das cinco modalidades e hashes das métricas |
| `ex-035_fusion_metrics.csv` | `outputs/experiments/ex-035_gallagher_paired_fusion_a01_a02_20260922_v2/fusion_metrics.csv` | cinco linhas de métricas agregadas vigentes |
| `ex-035_fusion_metrics_per_query.csv` | `outputs/experiments/ex-035_gallagher_paired_fusion_a01_a02_20260922_v2/fusion_metrics_per_query.csv` | 100 linhas sem fotografias, caixas, caminhos de imagem ou embeddings; permite recalcular agregados e rastrear consultas |

Os seis CSVs do protocolo em `data/evaluation/gallagher_corr_a01_a02_20260922_v2/` já estão preservados nos commits `b0abf4c` e `938911b` e não são duplicados aqui.

## Proveniência

| Execução | Commit registrado | Árvore limpa | SHA-256 do manifest original |
|---|---|---:|---|
| EX-033 | `b0abf4c0777cd0aa6fce34b96709e43043c3eb7c` | sim | `8066afb36f95545c4bea1a0b996a0d40acaef6d801fd8ed75bcf4b8aca8b4222` |
| EX-034 | `938911b5ffa117a4341bf35d7e8665e74a2ac49b` | sim | `d448900a1300aaf6eed730d66f3c90bec5ebab08e96126c75f34a962cabc513d` |
| EX-035 | `938911b5ffa117a4341bf35d7e8665e74a2ac49b` | sim | `a7ee9c119246b2c9b926376c4435ea24cabf72f7496095f74fdc729f60dbb1c8` |

Resultados originais EX-035:

- métricas agregadas: `b4a271dbc2ca8c15436cf18d2c2cdcba2f5b48d3f279f590e4b43bb10c52c639`;
- métricas por consulta: `111039208e5bc9f37b292f5cc0241de574fd56f7629d5885a8182aeda255dce6`;
- Top-10 local não copiado: `31eb9b6e8e811349d9aa06aaf295e27ae37f74a22f41e1b1641541d0fede7a12`.

Os hashes acima identificam os arquivos originais no momento da validação. `apply_patch` normalizou CRLF para LF nas cópias. A comparação após normalizar terminações de linha confirmou conteúdo textual idêntico nos cinco arquivos. Os hashes das cópias preservadas são:

| Cópia textual | SHA-256 da cópia LF |
|---|---|
| `ex-033_gallagher_protocol_manifest.json` | `389481e25baae4fe0fa3d55794367c6c616bc0fd1ed42ee658efc8cb1aa84557` |
| `ex-034_paired_protocol_manifest.json` | `457a51722f4764cbec87c1ab9c24f0d9030974d29e24bf632908d0bd356d305c` |
| `ex-035_fusion_run_manifest.json` | `eb548488dbeec98959b6a15dfb2e2b266cb485f8eb89ef3c8e279e8147cd9c22` |
| `ex-035_fusion_metrics.csv` | `3874c8beedbe59dc0ef31b664175f5f560fee45e1bf742331367fde0b472a47e` |
| `ex-035_fusion_metrics_per_query.csv` | `826245d0ea3996f21663d60a71865f741bf80da4280798ad8a9a00336ce634b5` |

## Resultado da EX-035 — referência anterior

Este pacote permanece inalterado como evidência da correção de 22/09. A reprodução de fechamento com CUDA verificado, em 28/09, fornece os números atuais em [resultados experimentais](../../resultados_experimentais_congelados.md). Não misture os valores das duas execuções em uma tabela do artigo.

| Configuração | mAP |
|---|---:|
| somente face | 0,951446 |
| somente contexto global | 0,280864 |
| fusão 0,9/0,1 | 0,946297 |
| fusão 0,7/0,3 | 0,901525 |
| fusão 0,5/0,5 | 0,813358 |

Nenhuma fusão superou o baseline somente facial no mAP agregado. Essa conclusão se restringe às 20 consultas Gallagher selecionadas.

## Material deliberadamente excluído

Não foram copiados:

- fotografias, datasets ou arquivos compactados;
- recortes faciais;
- embeddings ou descritores;
- pesos de modelos;
- figuras com pessoas;
- rankings Top-10, pois contêm caminhos, identificadores, rótulos e caixas vinculáveis desnecessárias;
- credenciais ou caches.

O manifest EX-035 preserva o hash do Top-10 sem redistribuir suas linhas.

## Limites de uso

- Os testes 35/35 foram executados anteriormente e registrados no terminal fornecido pelo usuário; não foram reexecutados nesta consolidação.
- A revisão documental recalculou médias a partir dos CSVs, mas não reproduziu inferência ou embeddings.
- Holidays permanece auxiliar, com AP adaptada e sem comparação direta ao benchmark oficial.
- Estudos de caso encerrados permanecem fora do núcleo do artigo.
- A01/A02 estão resolvidos, mas continuam abertas as pendências institucionais, éticas, bibliográficas e editoriais.
