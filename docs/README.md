# Documentação do projeto

Leia primeiro o mapa canônico na raiz: [`LEIA_PRIMEIRO.md`](../LEIA_PRIMEIRO.md).

## Estado dos documentos

| Documento/pasta | Status | Uso correto |
|---|---|---|
| `../LEIA_PRIMEIRO.md` | canônico e atual | ponto de entrada para escopo, alertas e próximos passos |
| `README.md` da raiz | operacional | estrutura e comandos do software ativo |
| `protocolo_avaliacao.md` | canônico e atual | definições de exclusão, métricas, ranking e manifestos |
| `resultados_experimentais_congelados.md` | canônico e atual | tabela oficial de métricas, interpretações e limitações |
| `historico_privado/` | privado; ignorado pelo Git | dossiê completo em DOCX, PDF e Markdown; consultar apenas para história e recuperação |
| `artigo_sibgrapi_2026/` | rascunho histórico | fonte de texto/figuras; não contém validação agro e não está pronto para submissão |
| `entregas/` | entregas e rascunhos históricos | preservar; conferir data e protocolo antes de reutilizar resultados |
| `Referencias/` | acervo bibliográfico local | conferir licença; PDFs de terceiros não devem ir ao GitHub |

## Fontes históricas

- [Relatório explicativo acumulado](relatorio_estado_atual.md) — material de apoio; quando houver divergência, prevalecem o protocolo e a tabela congelada.
- [Plano de revisão bibliográfica](plano_revisao_bibliografica.md).
- [Referências e ideias da proposta original](referencias_proposta_original.md).
- [Texto extraído da proposta original](proposta_original_extraida.md).
- [Síntese da revisão bibliográfica](sintese_revisao_bibliografica.md).
- [Bibliografia anotada inicial](bibliografia_anotada.md).
- `entregas/Relatorio_Meta2_CIBIR.docx` — relatório histórico.
- `entregas/Artigo_CIBIR_Rascunho.docx` — rascunho histórico.

## Protocolo atual

- [Protocolo de avaliação](protocolo_avaliacao.md) — regras obrigatórias para os próximos experimentos.
- [Resultados experimentais congelados](resultados_experimentais_congelados.md) — números oficiais atuais.
- `tests/test_retrieval_methodology.py` — testes sintéticos do protocolo.

## Regra de evidência

Resultados registrados antes da auditoria de 15/09/2026 não são automaticamente válidos para o artigo da Revista Principia. Para números atuais, use somente `resultados_experimentais_congelados.md` e confirme a execução correspondente em `protocolo_avaliacao.md`.
