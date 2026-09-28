# Fechamento técnico e científico — versão final

Data: **28/09/2026**. **Estado: FECHAMENTO TÉCNICO E DOCUMENTAL CONCLUÍDO NO ESCOPO TESTADO.**

O projeto é uma pesquisa experimental acompanhada de um protótipo local de
recuperação de fotografias. As três etapas autorizadas — congelamento,
validação técnica e consolidação das evidências — foram concluídas.
Isso não é declaração de artigo pronto para submissão, aceite garantido ou TC aprovado.

## 1. Escopo congelado

- Um único artigo em inglês; Gallagher principal, LFW/Holidays auxiliares.
- Busca facial, global e fusão tardia experimental, sem novos modelos ou treinamento.
- Aplicação local com preparação, atualização e remoção de álbuns, escolha
  explícita da pessoa e paginação.
- HEIC/HEIF e leitura controlada de JPEGs grandes; originais preservados.
- Álbum familiar somente para aceitação funcional privada, não para o artigo.
- Sem nuvem, outra arquitetura, novo dataset ou alegação de larga escala.
- Daqui em diante, somente defeitos, inconsistências e bloqueadores justificam
  mudanças funcionais. Melhorias opcionais são trabalhos futuros.

## 2. Autorização, commits e preservação

Artur autorizou os testes, a reprodução e os commits locais pelo assistente.
Não houve push, publicação, submissão ou contato externo.

| Marco | Commit | Papel |
|---|---|---|
| Estado recebido | `8d114969f3d38cf9f68c390139f8a91dfda83bb9` | branch principia-rework; alterações locais anteriores preservadas |
| Primeira candidata | `ea43982004c73bcd82dde95c71f9027ef7fca435` | interface/álbuns, leitor controlado, validação e proveniência |
| Candidata executada | `91a2a28360ded41fd86abf3371c989d571bf95bd` | correção CUDA facial, cinco regressões e confirmação do executor |

A consolidação documental posterior não altera o código científico executado.
O hash dessa consolidação é informado na entrega da conversa, evitando referência
circular do documento ao próprio commit.

## 3. Defeito encontrado e corrigido

Escolher CUDA na interface não ativava CUDA nos modelos faciais do
InsightFace 0.2.1: `ctx_id=0` não substituía os provedores das sessões ONNX.
A aceitação inicial revelou `CPUExecutionProvider` nos dois modelos, mesmo
com o pedido de CUDA. O teste de ambiente genérico não detectava esse caso.

O adaptador agora carrega as DLLs já existentes pelo PyTorch e configura
explicitamente CUDA em detecção e reconhecimento. A ativação é conferida;
falha gera orientação para selecionar CPU, sem fallback silenciosamente
rotulado como CUDA. O provedor CPU continua disponível para operadores sem
implementação CUDA: ativar o provedor não significa que todo operador execute na GPU.

Isso segue o mecanismo documentado pelo [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
consultado em 28/09, e foi verificado nos modelos efetivamente carregados.
Pesos, biblioteca InsightFace e regras científicas não foram substituídos.

A tentativa inicial em `outputs/validation_runs/fechamento_20260928/` foi
interrompida antes de concluir LFW. Seus arquivos parciais foram preservados,
não fornecem métricas finais e não devem ser confundidos com a execução válida.

## 4. Testes e aceitação da aplicação

- Suíte inicial: 85 aprovados; após proveniência: 88; candidata CUDA: **93 aprovados**.
- Análise sintática: 55 arquivos Python ativos, sem erro.
- As 19 interfaces de comando responderam a `--help`, sem erro de importação.
- As buscas facial, global e por fusão foram realmente executadas no terminal
  com os índices novos: dez fotos distintas por busca, sem a fonte.
- Lock conferido: 82 versões fixadas e duas rodas CUDA por URL, sem divergências.
  `uv pip check` aprovou 88 pacotes instalados. Quatro pacotes legados de
  aquisição permanecem instalados fora do lock; não foram usados neste fluxo.
  Não houve instalação, atualização ou remoção de dependências.

A aceitação final usou os modelos reais e a coleção familiar atual:

| Verificação | Resultado observado |
|---|---|
| Perfil inicial | Uso pessoal; sem Gallagher automático |
| Preparar álbum pela interface | 171 imagens: 59 HEIC e 112 JPEG; 552 descritores faciais; zero falhas de leitura |
| Consulta com múltiplos rostos | HEIC com 11 detecções; seleção e troca explícita da pessoa |
| Busca facial | resultados reais; a troca da pessoa invalida o resultado antigo |
| Busca global e paginação | 170 candidatas, fonte excluída, oito páginas, sem teto de 50 |
| Fusão no perfil Pesquisa | 170 candidatas, escores finitos e fonte excluída |
| Atualização | mesmo destino; hashes dos quatro arquivos de índices/metadados iguais, sem pasta v2 |
| JPEG muito grande | consulta e prévia de 199.756.800 pixels; decodificação global reduzida e controlada |
| Executor facial | CUDA e CPU registrados nos dois modelos, com CUDA ativado |
| Remoção | recusada sem confirmação; removeu somente o álbum exclusivo de teste |
| Preservação | hashes dos 207 arquivos originais e dos dois álbuns preexistentes inalterados |

A automação Streamlit AppTest dirigiu telas e subprocessos reais; somente o
widget de upload recebeu os bytes programaticamente. O upload também foi
testado no navegador real: HEIC, busca e segunda página. Um artefato do simulador
com formulários antigos após `st.rerun()` foi contornado abrindo uma sessão nova,
sem alterar a interface para satisfazer o teste.

A execução final de aceitação possui 12 checkpoints aprovados e levou cerca
de 189 segundos; não é benchmark de desempenho. Relatório privado:
`outputs/acceptance_runs/fechamento_20260928_cuda/acceptance_report.json`.
O primeiro harness falhou nesse artefato de simulação; as tentativas e logs
continuam preservados, sem serem chamados de aprovações finais.

Os índices dos álbuns exclusivos de validação foram removidos pela rotina
protegida da aplicação. São regeneráveis a partir das fotos, que não foram
alteradas. Os álbuns pessoais existentes não foram atualizados nem removidos.
Cópias locais de consulta/cache permanecem na área ignorada da aplicação.

## 5. Reprodução científica e conferência independente

Comando realmente executado, com Git limpo na candidata:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/run_final_validation.py --run-root outputs/validation_runs/fechamento_20260928_cuda
```

Não repetir no mesmo destino; outra reprodução exige pasta nova.
Os baixadores não foram executados: as bases e os pesos locais existentes
foram reutilizados, sem nova aquisição.

| Etapa | Evidência |
|---|---|
| Ambiente CUDA | verificação existente aprovada |
| LFW | 13.233 imagens escaneadas, 16.058 rostos, 48 imagens sem detecção, zero falhas de leitura; 1.672 consultas |
| Holidays | 1.491 imagens e 500 consultas; zero falhas de leitura e zero reduções de JPEG |
| Gallagher | 589 imagens globais, 1.303 rostos em 587 fotos, duas sem detecção; 20 consultas, 884 relevâncias e cinco métodos |
| Proveniência | 13 manifestos no commit 91a2a28, todos dirty=false; 92 referências e 51 arquivos únicos conferidos |
| Integridade das matrizes | alinhamento com metadados, descritores finitos e normalizados, IDs únicos nas unidades previstas |
| Integridade dos dados | SHA-256 de 15.265 fontes indexadas conferidos com os arquivos atuais |
| Referências preservadas | 113 arquivos de outputs/final e arquitetura_v1 inalterados |

A01/A02 foram reconferidos independentemente: relevância reconstruída pelas
anotações, duas fotos omitidas pelo detector ainda relevantes, fontes excluídas,
vinte caixas contendo o ponto dos olhos anotados. Foram recalculados os
agregados e Precision/Recall@5/@10 dos exemplos salvos.

Resultado principal atual: facial **0,952261**, global 0,280864, fusões
0,946274 / 0,901460 / 0,813297. Nenhuma fusão superou o baseline facial.
A comparação descritiva cobre todas as vinte consultas; 0,9/0,1 teve dois
ganhos, nove empates e nove perdas. O [registro dos resultados](resultados_experimentais_congelados.md)
contém tabelas, deltas, hashes e limitações, sem selecionar apenas casos favoráveis.

Os vetores globais foram idênticos aos anteriores; os faciais mudaram após
ativar CUDA. A AP mudou em 37 consultas LFW e quatro consultas do baseline
Gallagher, sem mudança em Precision/Recall@5/@10. Um controle CPU em duas
imagens, incluindo a maior divergência LFW, reproduziu exatamente os vetores
de referência. Não se alega igualdade entre executores nem uma segunda
reprodução CPU completa.

Manifesto final: `outputs/validation_runs/fechamento_20260928_cuda/final_validation_manifest.json`.
SHA-256: `5fc6acd8c42c75520c0eecefa5366f3a33e0627d217b0b0cf61dd556876f34c9`.

## 6. Relação com a auditoria anterior

| Achados | Estado neste fechamento |
|---|---|
| A01/A02 — gabarito e pessoa-alvo | correções anteriores reconferidas e reproduzidas |
| A03/A04 — Holidays, duplicatas e AP | uso auxiliar delimitado; sem equivalência ao benchmark ou à exclusão interativa por hash |
| A06/A07/A08/A13 — população, dependência, pesos e LFW | limitações explicitadas; análise pareada descritiva, sem generalização ou causalidade |
| A10 — modelos e executor | hashes dos pesos e provedores reais registrados; defeito CUDA corrigido |
| A14 — interface versus experimento | aceitação concluída; interface continua exploração, não execução automática do protocolo |
| A05/A11 — manuscrito e geração antiga | histórico fora da versão ativa; tabelas atuais rastreadas, artigo novo ainda não escrito |
| A09/A12 e decisões autorais/editoriais | não encerrados por testes; exigem decisões humanas/institucionais |

Não foi reaberta a auditoria integral, acrescentado treinamento ou buscado
um resultado favorável à fusão.

## 7. Critérios encerrados e limites

- [x] Escopo congelado.
- [x] Suíte atual e aceitação real aprovadas.
- [x] HEIC, JPEG grande, escolha da pessoa, paginação e ciclo de álbuns verificados.
- [x] Dependências conferidas e candidata commitada antes da reprodução.
- [x] Reprodução completa em destino novo e com Git limpo.
- [x] Hashes, contagens, agregados, relevância e deltas conferidos.
- [x] Fotos, embeddings, modelos e credenciais ausentes dos arquivos versionados na inspeção.
- [x] Documentos canônicos consolidados e ligados à execução válida.

Este fechamento comprova funcionamento **nos caminhos e dados testados**.
Não comprova ausência de qualquer defeito, desempenho em qualquer máquina,
recuperação de toda presença humana, larga escala ou acurácia familiar.
Top-10 foi inspecionado; rankings integrais foram usados pelo avaliador, mas não
persistidos nem comparados posição a posição independentemente.
Não foi feito teste de carga, avaliação demográfica, revisão jurídica,
estudo de usabilidade ou instalação em um ambiente novo do zero.

O preset Gallagher da interface continua lendo a referência preservada em
`outputs/final/gallagher/`. Para exploração dos índices desta reprodução,
escolha Pesquisa → Índices personalizados e indique as pastas face_index e
global_index desta execução. A tela avisa que não substitui os protocolos.
Uma aplicação já aberta deve ser reiniciada para carregar os modelos/caches
da candidata; a instância pré-existente de Artur não foi encerrada por esta rodada.

## 8. Próximo marco e compreensão do autor

A base técnica está consolidada para elaborar o artigo em inglês. Antes de
submeter, faltam: revisão e validação humana do método e texto; regras de
entrega/aceite e calendário do TC; elegibilidade ética e condições de uso dos
dados/modelos; aprovação da autoria; política editorial e eventual depósito
institucional. Nada disso é substituído pelo Git limpo ou pela suíte aprovada.

Artur deve conseguir explicar: diferença entre detectar rosto e comparar
identidade; fotografia inteira como unidade recuperada; origem do gabarito;
exclusão da fonte; AP/mAP versus probabilidade; pesos fixos sem calibração;
diferença entre uso interativo e protocolo; e por que um resultado negativo,
delimitado, não é uma falha de implementação.

A origem, os limites de licença e o uso real de IA estão em
[dados, modelos e privacidade](dados_modelos_privacidade.md). A revisão
bibliográfica foi ajustada para não prometer novidade, causalidade ou aceite
sem evidência. A declaração final de IA e a interpretação científica exigem
validação pelos autores humanos.
