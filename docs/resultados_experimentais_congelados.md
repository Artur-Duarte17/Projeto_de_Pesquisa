# Resultados experimentais oficiais

Data de congelamento: **16 de setembro de 2026**.

Este documento é a tabela canônica dos resultados reproduzidos no ambiente definitivo. Números encontrados em relatórios ou rascunhos anteriores não substituem estes resultados.

## Ambiente comum

- Python 3.10.20;
- PyTorch 2.11.0 com CUDA 12.8;
- ONNX Runtime GPU 1.23.2;
- NVIDIA GeForce RTX 3050 Ti Laptop GPU;
- métricas calculadas sobre o ranking integral;
- Precision@K com denominador fixo K;
- fotografia-fonte excluída do ranking e da relevância;
- Top-10 usado somente como limite de armazenamento e apresentação.

## Tabela principal

| Experimento | Base | Consultas | Candidatas por consulta | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Face por identidade | LFW | 1.672 | 13.184 | 0,457177 | 0,284629 | 0,901149 | 0,945308 | 0,965135 |
| Face multi-rosto | Gallagher | 20 | 586 | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,913949 |
| CBIR global | Holidays | 500 | 1.490 | 0,324000 | 0,177600 | 0,870215 | 0,917563 | 0,842612 |
| Contexto global para pessoa | Gallagher pareado | 20 | 588 | 0,260000 | 0,195000 | 0,203513 | 0,234716 | 0,280833 |
| Face/contexto 0,9/0,1 | Gallagher pareado | 20 | 588 | 0,870000 | 0,645000 | 0,566409 | 0,662866 | 0,908721 |
| Face/contexto 0,7/0,3 | Gallagher pareado | 20 | 588 | 0,820000 | 0,640000 | 0,533433 | 0,657866 | 0,865268 |
| Face/contexto 0,5/0,5 | Gallagher pareado | 20 | 588 | 0,750000 | 0,595000 | 0,473894 | 0,609914 | 0,778968 |
| Face | Agrishow 2022 | 1 | 123 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| Contexto global | Agrishow 2022 | 1 | 123 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,797298 |
| Face/contexto 0,9/0,1 | Agrishow 2022 | 1 | 123 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| Face/contexto 0,7/0,3 | Agrishow 2022 | 1 | 123 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997898 |
| Face/contexto 0,5/0,5 | Agrishow 2022 | 1 | 123 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,993485 |

No protocolo pareado, o baseline somente facial também foi executado no universo global de 588 candidatas. As duas fotografias sem face detectada aparecem apenas no final do ranking e todas as métricas reproduzem exatamente a avaliação facial de 586 candidatas da EX-012.

## O que cada linha responde

- **LFW:** o sistema recupera outras fotografias da mesma identidade em um baseline facial amplo?
- **Gallagher face:** o sistema encontra uma pessoa em fotografias com múltiplos rostos?
- **Holidays:** a ResNet50 recupera fotografias visualmente semelhantes?
- **Gallagher contexto:** a aparência da fotografia-fonte, sem usar o rosto, localiza outras fotografias que contêm a pessoa?
- **Gallagher fusão:** acrescentar contexto global ao sinal facial melhora a recuperação da pessoa selecionada?
- **Agrishow 2022:** o fluxo localiza uma pessoa pública em outras fotografias do mesmo evento agro, usando uma fonte sorteada antes das buscas?

As linhas não representam a mesma tarefa. O mAP do LFW não deve ser comparado ao mAP da Holidays como se os métodos fossem concorrentes diretos.

## Conclusões sustentadas

1. A busca facial apresentou desempenho elevado nos dois protocolos de identidade.
2. A ResNet50 apresentou bom desempenho na tarefa própria de similaridade global da Holidays.
3. Contexto visual isolado não substitui reconhecimento facial para localizar uma pessoa.
4. Nenhum dos três pesos de fusão superou o baseline facial em média.
5. A fusão 0,9/0,1 melhorou a AP de uma consulta, não alterou nove e piorou dez; portanto, ajuda casos isolados, mas não é consistentemente benéfica.
6. No estudo de caso Agrishow, face e fusão 0,9/0,1 empataram em mAP; aumentar o peso global reduziu levemente o resultado.
7. O estudo Agrishow demonstra a aplicação no evento selecionado, mas uma consulta e uma pessoa não sustentam generalização para todo o domínio agro.
8. Na Agrishow, o primeiro resultado irrelevante apareceu na posição 90 para face e fusão 0,9/0,1, mas já na posição 17 para contexto global.

## Limitações obrigatórias

- O Gallagher usa 20 identidades selecionadas entre as mais frequentes, uma consulta por identidade.
- LFW contém muitas identidades com somente uma fotografia relevante após excluir a consulta; isso eleva a frequência de AP igual a 1.
- Holidays possui mediana de uma imagem relevante por consulta, o que reduz Precision@K mesmo quando Recall e mAP são elevados.
- Os tempos de backend com descritores pré-calculados não representam latência completa de uma imagem nova.
- Gallagher e Holidays não constituem validação agro; a Agrishow constitui somente um estudo de caso aplicado em um evento agro.
- A Agrishow usa uma única pessoa, uma consulta sorteada e 90 imagens relevantes entre 123 candidatas; a alta prevalência ajuda a explicar P@5 e P@10 iguais a 1 em todos os métodos.
- As figuras com pessoas permanecem somente nos resultados locais e não devem ser publicadas automaticamente.

## Evidência reprodutível

| Execução | Evidência principal | Commit de execução |
|---|---|---|
| EX-009 | `outputs/experiments/ex-009_lfw_face_evaluation` | `ac13b6f` |
| EX-012 | `outputs/experiments/ex-012_gallagher_face_evaluation` | `66d41bc` |
| EX-014 | `outputs/experiments/ex-014_holidays_global_evaluation` | `c6af569` |
| EX-017 | `outputs/experiments/ex-017_gallagher_paired_fusion` | `532bb84` |
| EX-018 | `outputs/experiments/ex-018_paired_fusion_error_analysis` | `8cb1e12` |
| EX-022 | índices `ex-022_agrishow_face_index` e `ex-022_agrishow_global_index` | `6aa8dad` |
| EX-023 | `outputs/experiments/ex-023_agrishow_paired_fusion` | `ecc37f7` |
| EX-024 | `outputs/experiments/ex-024_agrishow_analysis` | `1bf4a95` |

Os diretórios de evidência são locais e ignorados pelo Git. Cada execução possui manifesto com configuração, hashes, versões do ambiente e commit correspondente.

## Regra para alterações futuras

Estes números somente podem ser substituídos por uma nova execução identificada, com protocolo documentado, manifesto de árvore limpa, hashes válidos e auditoria equivalente. Novas avaliações agro devem formar uma seção separada, sem reclassificar Gallagher como conjunto agro.
