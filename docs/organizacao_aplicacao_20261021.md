# Manutenção e proposta de organização — 21/10/2026

## Correção arquitetural

O teste de dependências encontrou sete subcasos falhos: quatro comandos ainda importavam bibliotecas antigas, e três dessas bibliotecas também importavam outros módulos legados. Havia implementações paralelas de carregamento de modelos, busca, fusão e métricas, apesar da existência do núcleo compartilhado.

Os comandos abaixo agora usam diretamente `retrieval/adapters` e `retrieval/domain`:

- `scripts/face/03_evaluate_face.py`;
- `scripts/face/05_analyze_face_errors.py`;
- `scripts/global/03_evaluate_global.py`;
- `scripts/fusion/02_evaluate_fusion.py`.

Foram retirados `scripts/face_lib.py`, `scripts/global_lib.py`, `scripts/fusion_lib.py` e `scripts/retrieval_common.py`. O histórico Git preserva suas versões anteriores. O teste arquitetural não foi alterado para aceitar as dependências antigas.

As rotas genéricas passam a compartilhar as políticas canônicas de modelos, exclusão da própria fotografia por caminho/hash, ranking e métricas, incluindo o denominador solicitado em Precision@K. A interface já utilizava esse núcleo. As execuções científicas congeladas não foram recalculadas nem substituídas.

## Validação

- Suíte completa: **93 testes aprovados**, código de saída zero.
- Os quatro comandos alterados iniciaram com `--help`, código zero.
- Um exercício adicional percorreu as quatro rotas com índices sintéticos e as implementações reais de arquivos, ranking e métricas. Apenas a extração dos modelos foi substituída por stubs. Foram conferidos exclusão da fonte, P@5 de 0,2 para uma candidata relevante e geração do relatório de erros.
- `git diff --check` passou.
- Os 72 arquivos da execução científica e os 47 arquivos do manifesto local da monografia tiveram hashes conferidos, sem divergência.

O teste de usuário anterior percorreu a aplicação no navegador com uma coleção isolada de 61 imagens JPEG/HEIC, ampliada para 62 na atualização. Preparação, seleção de rosto, buscas facial/global/fusão, CPU/CUDA, limite de resultados, paginação, persistência e remoção dos índices funcionaram. Esse teste é de funcionamento; não estima acurácia nem AP/mAP do novo acervo. A exportação CSV da tabela permaneceu inconclusiva na automação. O servidor e o álbum temporários foram encerrados/removidos, preservando as fotografias.

Logs, capturas e dados do teste permanecem locais em `outputs/`, ignorados pelo Git. Fotografias, recortes, vetores, pesos, vídeos, a conversa de WhatsApp e os pacotes restritos da monografia não integram esta entrega de manutenção.

## Proposta de separação, ainda não aplicada

A proposta é manter a raiz atual como repositório da aplicação e preservar os materiais científicos no arquivo local. Nenhuma pasta foi realocada nesta rodada.

| Manter junto da aplicação | Preservar junto da pesquisa |
|---|---|
| `.git`, configurações Git/Streamlit e README de uso | Monografia, bibliografia e entregas acadêmicas |
| Interface, núcleo compartilhado e preparador de álbuns | Datasets, consultas, gabaritos e protocolos |
| Indexadores e comandos de busca usados no fluxo normal | Avaliadores, análises científicas e orquestradores finais |
| Testes operacionais, dependências e documentação técnica | Testes científicos, execuções congeladas e evidências |
| Ambiente Python e dados privados de operação, ignorados pelo Git | Scripts históricos e materiais das etapas anteriores |

Antes de uma movimentação, será necessário:

1. Tornar opcional a fonte Gallagher fixa do perfil Pesquisa, preservando a possibilidade de fusão em álbuns próprios.
2. Manter os indexadores chamados pelo preparador de álbuns; as pastas atuais misturam indexação operacional e avaliação científica.
3. Separar os testes mistos que importam orquestradores científicos e também verificam regras da aplicação.
4. Consolidar dependências: arquivos antigos como `requirements/app.txt` e `fusion.txt` ainda divergem do lock canônico.
5. Preservar os caminhos relativos necessários aos geradores da monografia e às execuções, além de inventariar pesos e referências externas à raiz.
6. Capturar alterações e arquivos locais não commitados antes de retirar os materiais acadêmicos da árvore ativa.

Uma **cópia congelada** é um retrato identificado por data, commit e hashes. Ela reúne código e dependências de referência, documentos, dados/resultados necessários e proveniência, permitindo verificar o estado preservado quando a aplicação evoluir. O núcleo compartilhado permanece na aplicação e também precisa estar disponível nessa referência científica. Um bundle Git sozinho não inclui arquivos ignorados ou alterações não commitadas.

Essa cópia de preservação ainda não foi criada. Uma cópia no mesmo computador também não comprova backup independente. A configuração do GitHub será mantida, sem reescrita do histórico; a reorganização da árvore e qualquer publicação posterior dos materiais restritos dependem de escopo próprio.

O ZIP preliminar para registro de software, preparado antes da correção arquitetural, identifica a versão anterior. Depois de estabilizar o código e decidir a organização, será necessário gerar outro pacote e outro SHA-512.
