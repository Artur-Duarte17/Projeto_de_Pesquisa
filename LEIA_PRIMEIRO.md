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

O código ativo já incorpora exclusão uniforme da imagem-fonte, Precision@K com denominador K, AP/mAP sobre o ranking integral, separação do Top-K salvo, hashes das imagens indexadas, manifestos de execução e testes sintéticos. Isso corrige o protocolo no código, mas ainda não produz resultados oficiais: os índices e experimentos precisam ser refeitos.

## Próxima ordem técnica

1. a EX-007 foi concluída: `laboratorio/cibir_gpu` foi validado com CUDA e o ambiente substituído foi removido;
2. as EX-008 e a preparação da EX-009 foram concluídas: o índice facial LFW e o protocolo de 1.672 consultas estão congelados;
3. executar a avaliação facial LFW da EX-009 e, depois, Gallagher e Holidays no mesmo ambiente;
4. conferir métricas diretamente nos CSVs e manifestos;
5. definir um protocolo próprio para a fusão antes de tratá-la como resultado final;
6. obter coleção agro autorizada;
7. adaptar o manuscrito ao modelo vigente da Revista Principia.
