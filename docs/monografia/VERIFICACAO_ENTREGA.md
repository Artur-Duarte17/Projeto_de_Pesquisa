# Verificação documental da entrega

Conferência documental original concluída em 29/09/2026. Revisão e aprovação por Artur e pela
orientação continuam pendentes. Este registro descreve verificações documentais,
não uma nova auditoria integral nem reprodução científica.

## Resultado

- PDF completo: **40 páginas**, seis capítulos, um apêndice, resumo/abstract,
  listas e referências; **21 fontes citadas, três figuras e sete tabelas**.
- Tectonic 0.17.0+20260731: compilação final concluída com código de saída zero.
  BibTeX concluído; log final sem referências indefinidas, caracteres ausentes,
  caixas excedendo a margem ou avisos de nova passagem necessária.
- O processo emite diagnóstico de configuração Fontconfig, mas usa as fontes
  TeX do pacote e gera o PDF. Texto extraído e páginas renderizadas foram
  examinados; não se observou ausência de glifos por esse diagnóstico.
- PDF sem marcadores `??`; todas as chaves citadas existem na bibliografia e
  todas as 21 entradas são citadas. Remissões internas resolvidas.
- As páginas foram renderizadas por PDFium e examinadas em folhas de contato;
  fórmulas, diagrama e comparação pareada também foram abertos ampliados.
  Ajustes finais de paginação, bibliografia e apêndice foram recompilados,
  renderizados e reexaminados. Não há conteúdo cortado observado.
- A versão originalmente conferida continha marcações editoriais de revisão. A limpeza de 01/10/2026 removeu essas marcações da fonte; a nova compilação ainda precisa ser verificada antes de substituir o PDF de entrega.

## Coerência numérica e preservação

O hash do manifesto científico original continua sendo
`5fc6acd8c42c75520c0eecefa5366f3a33e0627d217b0b0cf61dd556876f34c9`.
Os **72 arquivos, 81.607.118 bytes**, foram novamente comparados por tamanho e
SHA-256 ao inventário inicial: nenhum alterado. Isso verifica preservação
nesta rodada, não existência de backup independente.

As médias das 100 linhas Gallagher foram comparadas aos cinco agregados, com
tolerância de 1e-12. Conferidos 20 pares de consultas por método, 588 candidatos
por ranking e 884 relações de relevância. A análise pareada foi calculada antes
de retirar identificadores da apresentação. Os hashes das figuras e tabelas
geradas foram conferidos contra `PROVENIENCIA_FIGURAS_TABELAS.json`.

A leitura dos cabeçalhos/matrizes em modo de memória mapeada confirmou as formas
(1303, 512) facial Gallagher, (589, 2048) global Gallagher, (16058, 512) facial
LFW e (1491, 2048) global Holidays, sem carregar modelos ou executar inferência.

Não foram recalculadas APs individuais a partir de rankings integrais, pois
esses rankings não foram persistidos. Não foram repetidos os 93 testes ou
a aceitação familiar. Seus números são evidências registradas no fechamento.

## Origem dos elementos visuais e tabelas

| Elemento | Origem e transformação |
|---|---|
| Tabela 2.1, trabalhos relacionados | Síntese das fontes citadas em 2.6; limites de leitura em CONTROLE_FONTES.md. Não reproduz números externos. |
| Figura 3.1, fluxo | Diagrama autoral documental da arquitetura e adaptadores existentes; não é saída experimental nem tela com pessoas. |
| Tabela 4.1, coleções | Manifestos de indexação/protocolo e aceitação atual, confrontados com fechamento_tecnico_2026.md. |
| Tabela 5.1 e Figura 5.1 | fusion_metrics.csv atual; tabelas/agregados.tex e figuras/map_gallagher.pdf gerados por gerar_resultados.py. |
| Tabela 5.2 e Figura 5.2 | fusion_metrics_per_query.csv atual; diferenças pareadas, contagens e distribuição sem identificadores. Pontos muito próximos podem se sobrepor visualmente. |
| Tabela 5.3 | face_metrics.csv de LFW e global_metrics.csv de Holidays, cópias idênticas no pacote leve. Protocolos distintos explicitados. |
| Tabela A.1 | Mapa documental de arquivos e usos; não contém observações experimentais novas. |
| Tabela A.2 | Objetivos a–f da proposta original, p. 3 do PDF, extraídos e conferidos visualmente; paráfrases confrontadas com implementação/fechamento. |
| Logo | Único elemento institucional copiado do template recebido; sem documentos pessoais ou aprovação de terceiros. |

## Escopo, privacidade e limites

O PDF, seus módulos e o pacote leve foram examinados para evitar inclusão de
fotografias, recortes, embeddings, pesos, credenciais, caminhos pessoais e
rankings vinculáveis. Os únicos gráficos experimentais mostram métricas e
distribuições sem identificação. Isso não constitui certificação de anonimização.

O diff foi revisado: alterações restritas à área autorizada de documentação e
ao `.gitignore`. `git diff --check` passou. O índice Git permanece sem alterações
preparadas para commit. Código científico, aplicação, testes, dependências,
protocolos, bases, índices e resultados originais não foram modificados.

Custos, tempos de desempenho, probabilidades de identidade, significância,
causalidade e aprovação institucional não foram inferidos. Ausência de ganho
médio é limitada ao protocolo atual. LFW e Holidays permanecem auxiliares;
família apenas aceitação privada; Agrishow/WIDER históricos.

Leituras bibliográficas são proporcionais às afirmações e documentadas no controle.
Costache tem acesso limitado ao resumo institucional; termos atuais de Gallagher
e LFW não foram encerrados. O modelo bibliográfico histórico não comprova
conformidade com norma vigente. Essas pendências não foram ocultadas.

## Conferência futura

`MANIFESTO_ENTREGA.json` identifica o PDF e as fontes por SHA-256. Qualquer alteração
posterior muda a versão verificável e exige atualização do registro após nova
compilação. O próximo passo humano está em GUIA_DE_REVISAO_ARTUR.md e PENDENCIAS.md.
