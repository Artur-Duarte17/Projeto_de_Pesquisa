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

Esse material foi preservado em `C:\Projeto_de_Pesquisa_arquivo_local\2026-09-24_pre_finalizacao`, junto com um bundle completo do Git e o estado não commitado anterior à organização.

## Evidência, execução e publicação

Os resultados EX-033/034/035 são evidência produzida e auditada. Entretanto, antes da redação final do artigo ainda será realizada uma execução limpa da versão organizada. O autor executará o orquestrador `scripts/run_final_validation.py`; as saídas LFW, Holidays e Gallagher serão conferidas por hashes, manifestos, contagens e métricas.

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

O lock de dependências foi regenerado e Artur executou a suíte ativa: **27 testes passaram em 24/09/2026**. O próximo marco é executar e conferir a validação científica completa com LFW, Holidays e Gallagher; depois, fazer o teste privado de aceitação com o álbum familiar. A aprovação dos testes automatizados ainda não demonstra que o fluxo completo termina com os modelos e imagens reais.
