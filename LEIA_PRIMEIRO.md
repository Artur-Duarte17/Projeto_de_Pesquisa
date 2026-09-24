# Leia primeiro — estado canônico do projeto

Data de referência: **24 de setembro de 2026**.

## Objetivo atual

Finalizar uma única versão do sistema e produzir um artigo científico em inglês sobre a pergunta:

> Em coleções fotográficas com múltiplas pessoas, acrescentar contexto visual global por fusão tardia melhora a recuperação baseada em identidade facial?

O sistema é o meio experimental. A conclusão principal é limitada ao protocolo Gallagher descrito em `docs/protocolo_avaliacao.md`.

## Resultado científico vigente

As correções A01 e A02 foram implementadas antes da reavaliação:

- o gabarito de relevância passou a incluir todas as fotografias anotadas da galeria global, sem depender do sucesso do detector facial;
- a pessoa-alvo da consulta passou a ser escolhida pela caixa facial que contém o ponto médio dos olhos anotados, sem fallback para um rosto vizinho.

A EX-035 avaliou 20 consultas, 588 candidatas por ranking e cinco configurações predefinidas. O maior mAP foi o baseline somente facial (`0,951446`). Nenhuma fusão testada o superou.

## O que permanece ativo

- indexação e busca facial;
- indexação e busca global;
- fusão tardia;
- protocolo Gallagher corrigido;
- avaliações auxiliares LFW e INRIA Holidays;
- interface Streamlit;
- testes metodológicos e manifestos.

## O que não pertence mais à versão ativa

- estudo de caso Agrishow;
- avaliação planejada WIDER FACE;
- protocolos Gallagher substituídos;
- documentos da proposta original e planejamentos concluídos;
- manuscrito SIBGRAPI e entregas intermediárias;
- geradores antigos de documentos.

Esse material foi preservado em `C:\Projeto_de_Pesquisa_arquivo_local\2026-09-24_pre_finalizacao`, junto com um bundle completo do Git e o estado não commitado anterior à organização. Saídas exploratórias e materiais locais antigos também foram movidos para lá; a referência `outputs/final/` e as evidências `outputs/experiments/` ficaram na árvore ativa.

## Evidência, execução e publicação

Os resultados EX-033/034/035 são evidência produzida e auditada. Depois da reorganização, **Artur executou 37 testes automatizados, todos aprovados**, e a validação completa LFW/Holidays/Gallagher no commit limpo `954d749`. A nova execução está em `outputs/validation_runs/arquitetura_v1/`, preservando a referência anterior em `outputs/final/`. Os índices, gabaritos e recortes relevantes têm hashes iguais; métricas por consulta, IDs, posições e escores dos Top-10 salvos coincidiram. Apenas tempos de execução e caminhos de recortes mudaram. Os rankings completos não são persistidos, portanto não houve comparação direta de cada posição além do Top-10.

Depois haverá um teste privado de aceitação com um álbum familiar. Esse teste responderá se a aplicação funciona como produto em uma coleção real, mas não substituirá a avaliação científica Gallagher.

## Regras que não podem ser quebradas

1. Não usar métricas Gallagher anteriores à correção A01/A02 como resultado final.
2. Não afirmar que a fusão melhorou o resultado agregado.
3. Não apresentar Gallagher, LFW ou Holidays como validação agropecuária.
4. Não usar `large scale` no título ou nas conclusões.
5. Não versionar fotografias, embeddings ou dados biométricos.
6. Não submeter o mesmo manuscrito simultaneamente a mais de um local.
7. Não declarar um teste como executado enquanto o autor não o tiver executado e conferido.

## Próximo marco

O lock de dependências foi regenerado, os **37 testes passaram** e a reprodução científica da arquitetura atual foi concluída e conferida em 24/09/2026. A referência antiga pertence ao commit `3cfba6e`; a nova execução, ao commit `954d749`. O próximo marco técnico é o teste privado de aceitação com o álbum familiar, incluindo a interface e a escolha explícita da pessoa a buscar.
