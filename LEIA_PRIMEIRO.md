# Leia primeiro — estado canônico do projeto

Data de referência: **18 de setembro de 2026**.

## Núcleo ativo

O projeto ativo é um sistema de recuperação de fotografias com:

- busca facial multi-rosto com InsightFace/ArcFace;
- retorno da fotografia completa em buscas por pessoa;
- CBIR global com descritor ResNet50;
- fusão tardia de escores facial e global;
- avaliação e interface mínima em Streamlit.

Os componentes ativos ficam em `scripts/face`, `scripts/global`, `scripts/fusion`, `scripts/data`, `scripts/app` e nos módulos comuns em `scripts/`.

## Linha encerrada

A **Meta 1 de segmentação facial** — CelebAMask-HQ, U-Net, DeepLabV3 e experimentos faciais privados preliminares — foi encerrada como fluxo ativo. Seu código, checkpoints, logs, métricas e contexto foram preservados em arquivo privado e no histórico Git antes da remoção local.

Não recrie dependências, saídas ou datasets da Meta 1 para trabalhar no sistema atual, salvo quando a tarefa for explicitamente histórica.

## Resultados: aviso obrigatório

Os CSVs e relatórios produzidos antes desta data contêm resultados históricos. A auditoria encontrou:

- fotografia-fonte não excluída uniformemente nas abordagens;
- Precision@K com denominador incorreto quando havia menos de K retornos;
- AP calculada no Top-10 e apresentada como mAP;
- ambientes de execução inconsistentes.

Portanto, **nenhuma métrica histórica deve ser tratada como resultado oficial atual**. As execuções EX-007 a EX-027 corrigiram, reproduziram e ampliaram o protocolo; os resultados oficiais atuais estão reunidos em `docs/resultados_experimentais_congelados.md`.

## Contexto agro

O texto existente apresenta aplicação potencial no meio rural e em eventos. Isso é **contextualização**, não validação agro.

Em setembro de 2026, a coleção Agrishow 2022 do Wikimedia Commons foi selecionada como fonte autorizada para um estudo de caso. Após remover três recortes derivados, a EX-019 validou 124 originais; a EX-020 congelou 91 fotografias presentes e 33 ausentes; e a EX-021 sorteou a primeira fonte antes das buscas. A EX-022 indexou 1.590 rostos em 123 fotografias e 124 descritores globais. A EX-025 mediu o efeito da alta prevalência de relevantes por meio de um controle aleatório. A EX-026 ampliou a avaliação para dez fontes, nas quais somente face obteve mAP de 0,987261, contexto isolado 0,850717 e fusão 0,9/0,1 0,986167. A EX-027 acrescentou um caso dirigido com chapéu e sombra; nesse caso específico, a fusão 0,7/0,3 obteve mAP de 0,997749, contra 0,997102 da face. Isso demonstra a aplicação no evento e na pessoa avaliados, não desempenho geral em todo o agro. Consulte `docs/protocolo_colecao_agro.md` antes de ampliar essa afirmação.

Recuperação de maquinário permanece trabalho futuro.

## Documentos canônicos

1. Este arquivo: mapa público e curto do estado atual.
2. `README.md`: execução e estrutura do software ativo.
3. `docs/README.md`: mapa dos documentos e seu status.
4. `docs/resultados_experimentais_congelados.md`: tabela oficial para escrita acadêmica.
5. `docs/protocolo_avaliacao.md`: protocolo, decisões e auditoria das execuções atuais.
6. `docs/protocolo_colecao_agro.md`: fonte, direitos, protocolo, resultados e limites do estudo de caso agro.
7. Dossiê privado `Dossie_Tecnico_Historico_Projeto_CBIR` em DOCX, PDF e Markdown: consultar somente para história, decisões, métricas antigas e recuperação.

O dossiê e o material de submissão não podem ser enviados ao GitHub público.

## Git e arquivos locais

- Use `git add` somente com caminhos explícitos.
- Revise `git diff --cached` antes de cada commit.
- Não versionar dados, imagens, checkpoints, embeddings, índices, ambientes, outputs, PDFs de terceiros, documentos pessoais, dossiê ou manuscrito de submissão.
- Não fazer push sem confirmação expressa de Artur.

## Estado da correção metodológica

O código ativo já incorpora exclusão uniforme da imagem-fonte, Precision@K com denominador K, AP/mAP sobre o ranking integral, separação do Top-K salvo, hashes das imagens indexadas, manifestos de execução e testes sintéticos. Essa correção, por si só, não tornou oficiais os resultados históricos; cada índice e experimento precisa ser refeito antes de substituir os números antigos.

As EX-008 e EX-009 já foram refeitas e auditadas no ambiente definitivo. O índice LFW contém 16.058 faces de 13.185 fotografias. A avaliação de 1.672 consultas obteve P@5 de 0,457177, P@10 de 0,284629, Recall@5 de 0,901149, Recall@10 de 0,945308 e mAP de 0,965135. Esses números são o baseline LFW corrigido; não demonstram desempenho em eventos ou no contexto agro.

As EX-010 a EX-012 também foram refeitas e auditadas. No Gallagher, 20 consultas de identidades selecionadas entre as mais representadas obtiveram P@5 de 0,870000, P@10 de 0,645000, Recall@5 de 0,566409, Recall@10 de 0,662866 e mAP de 0,913949. O resultado valida o fluxo facial multi-rosto nesse acervo, mas o número limitado e a seleção das consultas devem permanecer explícitos.

As EX-013 e EX-014 validaram o CBIR global na Holidays com 500 consultas e mAP de 0,842612. As EX-015 a EX-017 implementaram a fusão pareada no Gallagher: somente face obteve mAP de 0,913949, somente contexto 0,280833 e as fusões 0,9/0,1, 0,7/0,3 e 0,5/0,5 obtiveram 0,908721, 0,865268 e 0,778968. A EX-018 mostrou que a fusão 0,9/0,1 melhorou uma consulta, não alterou nove e piorou dez.

## Próxima ordem técnica

1. as EX-007 a EX-018 foram concluídas e auditadas;
2. usar `docs/resultados_experimentais_congelados.md` como fonte numérica do artigo;
3. a EX-019 baixou e validou a coleção Agrishow 2022;
4. a EX-020 anotou e congelou a presença da pessoa-alvo;
5. a EX-021 sorteou a fotografia-fonte e congelou as consultas facial e global pareadas;
6. a EX-022 criou e auditou os índices facial e global da Agrishow;
7. a EX-023 avaliou os cinco métodos sem alterar retrospectivamente os pesos;
8. a EX-024 recalculou os rankings, analisou os erros e produziu a figura qualitativa com atribuição;
9. a EX-025 estabeleceu o controle aleatório para interpretar a alta proporção de imagens relevantes;
10. a EX-026 avaliou robustez em dez consultas, com seleção e relevância congeladas;
11. a EX-027 avaliou separadamente o caso dirigido de chapéu e sombra;
12. adaptar o manuscrito ao modelo vigente da Revista Principia usando somente os resultados congelados.
