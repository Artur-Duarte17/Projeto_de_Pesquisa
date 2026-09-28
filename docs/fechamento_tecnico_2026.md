# Fechamento técnico e científico — versão final

Data de início: 28/09/2026. **Estado: EM VALIDAÇÃO.**

Este documento registra a execução autorizada por Artur das etapas de
congelamento, validação técnica e consolidação da pesquisa. Não é aprovação
para submissão nem conclusão das obrigações institucionais do TC.

## Escopo congelado

- Um único artigo científico em inglês.
- Gallagher como avaliação principal; LFW e INRIA Holidays como verificações
  auxiliares de tarefas diferentes.
- Álbum familiar somente como teste privado de funcionamento, sem fotos,
  recortes, descritores ou rankings vinculáveis no Git ou no artigo.
- Indexação e busca facial, busca global e fusão tardia experimental.
- Aplicação local com preparação, atualização e remoção de álbuns, seleção
  explícita da pessoa e paginação dos resultados.
- HEIC/HEIF e leitura controlada de JPEGs grandes dentro dos limites documentados.
- Sem novos modelos, bases, treinamento, nuvem ou arquitetura; sem alegação
  de larga escala. Novas funcionalidades ficam fora deste fechamento.

A partir deste marco, mudanças funcionais devem corrigir um defeito,
inconsistência ou bloqueador verificável. Melhorias opcionais são trabalhos futuros.

## Responsabilidades e preservação

Artur autorizou a execução dos testes e validações pelo assistente e os commits
locais depois de testes aprovados. Não há autorização de push, publicação,
submissão ou contato externo. Os originais familiares e as evidências científicas
anteriores serão preservados. Preparação, atualização e remoção usarão somente
um álbum criado especificamente para validação.

## Evidências iniciais

- HEAD inicial: `8d114969f3d38cf9f68c390139f8a91dfda83bb9`, branch `principia-rework`.
- Reprodução anterior: `outputs/validation_runs/arquitetura_v1/`, commit limpo
  `954d74959f42f7eda3765df49023e8af193ce1cd`, de 24/09/2026.
- Suíte atual executada em 28/09/2026: **85 testes aprovados**.
- Após acrescentar três regressões de proveniência dos modelos: **88 testes aprovados**.
- A aceitação integrada inicial passou, mas revelou que o pedido `cuda` era
  ignorado pelos modelos faciais do InsightFace 0.2.1. A primeira reprodução
  científica deste fechamento foi interrompida antes de concluir a indexação LFW;
  seus artefatos parciais foram preservados e não constituem resultados finais.
- O adaptador agora carrega as DLLs existentes pelo PyTorch e seleciona
  explicitamente `CUDAExecutionProvider` em detecção e reconhecimento. Se a
  ativação falhar, informa erro em vez de chamar CPU de CUDA. **93 testes passaram**
  após cinco regressões adicionais; os dois modelos reais ativaram CUDA no teste
  de diagnóstico. A aceitação e a reprodução serão repetidas nesta candidata.
- Verificação existente `validate_environment.py --require-cuda`: **aprovada**.
- Pasta familiar atual: 171 imagens compatíveis (59 HEIC e 112 JPEG); vídeos e
  arquivo sem extensão não são fotografias indexáveis.

## Checkpoint

- [x] Escopo congelado e execução autorizada.
- [x] Testes automatizados iniciais aprovados.
- [x] Ambiente CUDA verificado.
- [x] Dependências instaladas compatíveis: `uv pip check` verificou 88 pacotes.
- [ ] Aceitação integrada com modelos e fotografias reais.
- [ ] Revisão de dependências, arquivos privados e commit da versão candidata.
- [ ] Reprodução científica completa da candidata em destino isolado.
- [ ] Conferência de hashes, métricas e comparações por consulta.
- [ ] Documentação canônica atualizada e revisão final aprovada.

Próximo passo: repetir a aceitação integrada e a reprodução científica depois
da correção de seleção do processador. Não declarar sucesso de uma etapa apenas
porque sua implementação existe.

## Limites do encerramento

Teste funcional não equivale a uma avaliação de acurácia da identidade familiar.
A confirmação humana já fornecida por Artur de uma busca correta não é um gabarito
exaustivo. A reprodução confirma somente o protocolo executado e os artefatos
conferidos. Questões institucionais, éticas, autorais, editoriais e a redação do
artigo permanecem separadas deste fechamento técnico.
