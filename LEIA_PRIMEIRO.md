# Leia primeiro — estado canônico do projeto

Data de referência: **15 de setembro de 2026**.

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

Portanto, **nenhuma métrica histórica deve ser tratada como resultado oficial atual**. Antes de publicação, o protocolo precisa ser corrigido e todas as abordagens devem ser reexecutadas no mesmo ambiente congelado.

## Contexto agro

O texto existente apresenta aplicação potencial no meio rural e em eventos. Isso é **contextualização**, não validação agro. Até a data acima, não havia coleção agro autorizada nem avaliação quantitativa nesse domínio.

O próximo recorte recomendado é localizar uma pessoa ou artista em várias fotografias de um evento agro autorizado. Recuperação de maquinário permanece trabalho futuro.

## Documentos canônicos

1. Este arquivo: mapa público e curto do estado atual.
2. `README.md`: execução e estrutura do software ativo.
3. `docs/README.md`: mapa dos documentos e seu status.
4. Dossiê privado `Dossie_Tecnico_Historico_Projeto_CBIR` em DOCX, PDF e Markdown: consultar somente para história, decisões, métricas antigas e recuperação.

O dossiê e o material de submissão não podem ser enviados ao GitHub público.

## Git e arquivos locais

- Use `git add` somente com caminhos explícitos.
- Revise `git diff --cached` antes de cada commit.
- Não versionar dados, imagens, checkpoints, embeddings, índices, ambientes, outputs, PDFs de terceiros, documentos pessoais, dossiê ou manuscrito de submissão.
- Não fazer push sem confirmação expressa de Artur.

## Estado da correção metodológica

O código ativo já incorpora exclusão uniforme da imagem-fonte, Precision@K com denominador K, AP/mAP sobre o ranking integral, separação do Top-K salvo, hashes das imagens indexadas, manifestos de execução e testes sintéticos. Essa correção, por si só, não tornou oficiais os resultados históricos; cada índice e experimento precisa ser refeito antes de substituir os números antigos.

As EX-008 e EX-009 já foram refeitas e auditadas no ambiente definitivo. O índice LFW contém 16.058 faces de 13.185 fotografias. A avaliação de 1.672 consultas obteve P@5 de 0,457177, P@10 de 0,284629, Recall@5 de 0,901149, Recall@10 de 0,945308 e mAP de 0,965135. Esses números são o baseline LFW corrigido; não demonstram desempenho em eventos ou no contexto agro.

## Próxima ordem técnica

1. as EX-007, EX-008 e EX-009 foram concluídas e auditadas;
2. executar as EX-010 e EX-011 para recriar o índice Gallagher e suas consultas;
3. executar a EX-012 para avaliar a busca facial Gallagher e, depois, refazer Holidays;
4. conferir métricas diretamente nos CSVs e manifestos;
5. definir um protocolo próprio para a fusão antes de tratá-la como resultado final;
6. obter coleção agro autorizada;
7. adaptar o manuscrito ao modelo vigente da Revista Principia.
