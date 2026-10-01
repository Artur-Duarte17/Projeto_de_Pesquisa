# Objetivos, seções e evidências

| Objetivo | Seções | Método/implementação | Evidência | Conclusão permitida |
|---|---|---|---|---|
| Representar e recuperar fotografias por identidade | 1, 2, 3 | SCRFD/ArcFace, máximo facial por foto | face_model; manifestos face_index atuais | componente funcional, sem prova de identidade nem modelo novo |
| Representar conteúdo global | 2, 3, 4 | ResNet50 DEFAULT, retirada da camada final e L2 | global_model; manifestos globais | imagem inteira, não fundo isolado |
| Comparar face, global e fusão | 4, 5, 6 | cinco pesos fixos; mesma galeria/consultas/gabarito | fusion_metrics e fusion_metrics_per_query | nenhuma fusão superou face no mAP deste protocolo |
| Avaliação rastreável | 4, apêndice | seleção anotada, exclusão da fonte, AP integral | manifesto final, protocolo e extrato documental | contagens e proveniência verificáveis; sem nova inferência nesta redação |
| Demonstrar uso local | 3, 5 | Streamlit, seleção de pessoa, gestão de álbuns | fechamento e aceitação privada de 28/09 | fluxos testados, não acurácia familiar ou estudo de usabilidade |
| Discutir limites | 2, 4, 5, 6 | análise descritiva pareada e literatura | fontes controladas e pendências | resultado circunscrito, sem significância ou causalidade |

## Correspondência com a proposta assinada

Original de seis páginas consultado sem modificação; aprovação informada por
Artur. Objetivo geral e itens a–f na p. 3 do PDF. Extração local tem falhas de
acentuação; objetivos foram também conferidos visualmente na página original. Não foram reproduzidas assinaturas.

| Proposta | Trabalho realizado | Ajuste explícito |
|---|---|---|
| Geral: todas as fotos, eficiência e grande acervo | protótipo e comparação Gallagher | sem garantia de todas as ocorrências ou grande escala |
| a: coletar, organizar e rotular bases e acervo | bases existentes/anotações originais; família privada | sem benchmark familiar anotado ou nova coleta nesta rodada |
| b: comparar e implementar algoritmos faciais | integração SCRFD/ArcFace pré-treinados | sem comparação entre detectores/modelos; RetinaFace não é o detector atual |
| c: descritores globais CNN pré-treinada | ResNet50/torchvision sem treinamento novo | atendido no escopo implementado |
| d: FAISS e grande escala | NumPy/CSV e busca exaustiva vetorizada | FAISS não integra versão ativa; escala não avaliada |
| e: fusão e interface amigável | pesos fixos e Streamlit com ciclo de álbuns | funcionalidade demonstrada, usabilidade não mensurada |
| f: P@K, R@K, mAP e latência | métricas de ranking e tempos instrumentais | sem benchmark controlado de latência |

Execução local substitui nuvem prevista na proposta. Processamento local reduz
transmissão, mas não garante conformidade com LGPD. O título provisório remove
escala. Ajustes não alteram retroativamente a proposta; dependem de confirmação
institucional por Artur/orientação. Nenhum terceiro foi contatado.

Auditoria histórica explica A01/A02 e limites. Tabelas usam exclusivamente
fechamento_20260928_cuda. Os 93 testes e a aceitação são evidência registrada,
não reexecução desta rodada.
