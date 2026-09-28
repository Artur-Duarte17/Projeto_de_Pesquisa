# Leia primeiro — estado canônico do projeto

Data de referência: **28 de setembro de 2026**.

## Objetivo atual

Finalizar uma única versão do sistema e produzir um artigo científico em inglês sobre a pergunta:

> Em coleções fotográficas com múltiplas pessoas, acrescentar contexto visual global por fusão tardia melhora a recuperação baseada em identidade facial?

O sistema é o meio experimental. A conclusão principal é limitada ao protocolo Gallagher descrito em `docs/protocolo_avaliacao.md`.

## Resultado científico vigente

As correções A01 e A02 foram implementadas antes da reavaliação:

- o gabarito de relevância passou a incluir todas as fotografias anotadas da galeria global, sem depender do sucesso do detector facial;
- a pessoa-alvo da consulta passou a ser escolhida pela caixa facial que contém o ponto médio dos olhos anotados, sem fallback para um rosto vizinho.

A reprodução de fechamento avaliou 20 consultas, 588 candidatas por ranking e cinco configurações predefinidas. O maior mAP foi o baseline somente facial (`0,952261`), seguido pela fusão 0,9/0,1 (`0,946274`). Nenhuma fusão testada o superou. A EX-035 e a reprodução de 24/09 continuam preservadas como referências anteriores; os valores atuais e as diferenças estão em `docs/resultados_experimentais_congelados.md`.

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

O fechamento de 28/09 foi executado pelo assistente com autorização de Artur: **93 testes passaram**, a aceitação integrada com 171 imagens familiares passou e a reprodução LFW/Holidays/Gallagher terminou no commit limpo `91a2a28360ded41fd86abf3371c989d571bf95bd`. As saídas estão em `outputs/validation_runs/fechamento_20260928_cuda/`.

A aceitação revelou um defeito de seleção do dispositivo: o InsightFace 0.2.1 não ativava CUDA apenas com `ctx_id=0`. O adaptador foi corrigido; os manifestos agora registram os provedores reais e os hashes dos pesos. Ativar CUDA alterou descritores faciais e algumas métricas, mas preservou a conclusão principal. Não houve novos modelos, treinamento ou mudança do gabarito.

O álbum familiar foi usado apenas para aceitação funcional privada. Os 207 arquivos originais e os dois álbuns preexistentes ficaram intactos. Os índices do álbum exclusivo de validação foram removidos, podendo ser recriados. Isso não é uma medida de acurácia familiar nem substitui Gallagher. Consulte `docs/fechamento_tecnico_2026.md` para cobertura e limitações.

## Regras que não podem ser quebradas

1. Não usar métricas Gallagher anteriores à correção A01/A02 como resultado final.
2. Não afirmar que a fusão melhorou o resultado agregado.
3. Não apresentar Gallagher, LFW ou Holidays como validação agropecuária.
4. Não usar `large scale` no título ou nas conclusões.
5. Não versionar fotografias, embeddings ou dados biométricos.
6. Não submeter o mesmo manuscrito simultaneamente a mais de um local.
7. Não declarar um teste como executado sem evidência de execução e conferência; registrar quando a execução foi assistida e autorizada pelo autor.

## Próximo marco

O fechamento técnico e documental foi concluído no escopo testado. O próximo marco é a preparação do artigo em inglês, com revisão e validação pelos autores humanos. Antes da submissão, continuam necessárias as decisões institucionais de TC/calendário, a elegibilidade ética e de uso dos dados, a aprovação da autoria e a compatibilidade editorial. O funcionamento do software não resolve essas decisões nem garante aceite.
