# Pendências da monografia — roteiro para Artur

Atualizado em **01/10/2026**. A monografia possui seis capítulos, três figuras, sete tabelas e 21 referências. A fonte foi auditada novamente em 01/10/2026 e recebeu uma limpeza editorial para retirar marcas de rascunho e referências à assistência automatizada. **A revisão de conteúdo por Artur e Gabriel continua necessária antes da entrega oficial**. A conferência feita está registrada em [VERIFICACAO_ENTREGA.md](VERIFICACAO_ENTREGA.md).

## Primeiro: o que fazer para enviar a Gabriel

Você não precisa resolver todas as questões institucionais antes de pedir a leitura dele. A ordem prática é:

- [ ] **1. Fazer sua primeira leitura.** Leia resumo, metodologia, resultados e conclusão, nesta ordem. Use o [guia de revisão](GUIA_DE_REVISAO_ARTUR.md). Em cada trecho que você não entenda, que não corresponda ao sistema ou que não se sinta confortável em defender, anote **página + dúvida**. Se tiver pouco tempo, envie primeiro as dúvidas mais importantes; não espere conhecer todos os detalhes de memória.
- [x] **2. Limpeza editorial inicial.** A fonte foi ajustada para retirar a página de nota editorial, referências à assistência automatizada e marcas editoriais de rascunho. Resultados, limitações e conclusões científicas foram preservados.
- [x] **3. Preparar a fonte para a próxima compilação.** Capa, folha de rosto, metadados, introdução, conclusão e apêndice foram limpos. **Ainda é necessário recompilar e conferir o novo PDF antes de substituir a cópia de entrega.**
- [ ] **4. Enviar para ele.** Você compartilha esse PDF e pede que indique **o que considera obrigatório corrigir para a defesa**. Ele pediu para recebê-lo quanto antes. Não é preciso esperar ficha catalográfica, escolha de banca, artigo aceito ou monografia em formato definitivo para ele começar a ler. **Concluído quando:** ele tiver recebido o rascunho; a aprovação vem depois da leitura e eventuais correções.

**Situação em 01/10:** a limpeza editorial da fonte foi concluída. Continuam pendentes a leitura humana, a recompilação/conferência do PDF atualizado, a revisão do orientador e as decisões institucionais.

## O que perguntar junto com o rascunho

Gabriel disse que provavelmente o trabalho atual basta para a defesa, mas **“vamos olhar”** indica que ele ainda precisa ler. Envie perguntas curtas e objetivas:

1. **Escopo e título:** a proposta incluía comparar modelos faciais, usar FAISS e avaliar grande escala e latência. Hoje a pesquisa avalia busca facial, busca pela imagem inteira e fusão em um conjunto delimitado; a aplicação local funciona. O texto mostra expressamente o que ficou de fora. **Pergunta:** “Com esse recorte e essas limitações, o que o senhor considera obrigatório acrescentar antes da defesa?” A [comparação de modelos](../plano_comparacao_facial.md) é uma proposta, ainda não um resultado. Só iniciar experimentos novos depois de decidir se eles são necessários e se cabem no prazo.
2. **Modelo do TC:** Gabriel afirmou que existe formato específico e prefere LaTeX. A fonte atual está em LaTeX, mas foi adaptada de um template anterior; não há comprovação de conformidade com o modelo vigente. **Pergunta:** “Qual é o modelo/manual atual e devo entregar apenas PDF ou também os arquivos LaTeX?” Depois de receber o material, conferir capa, folha de rosto, resumo, citações, referências e demais exigências.
3. **Datas e banca:** constavam **06/11/2026 para entrega** e **10/12/2026 para defesa** no calendário fornecido anteriormente. Essas datas não foram confirmadas novamente neste fechamento. **Pergunta:** “Esses prazos continuam válidos? Até quando o senhor precisa receber a versão para revisão e como solicitamos a banca?” Se o calendário mudou, ajustar o cronograma pelo documento atualizado.
4. **Uso das bases de rostos:** Gabriel respondeu que, sendo bases públicas, não seria necessário documento institucional. Isso é a orientação recebida, mas não identifica as condições de cada base nem resolve sozinho o que pode ser divulgado. **Pergunta:** “Para este TC, há alguma declaração, consulta ou restrição de depósito por usarmos imagens faciais de bases públicas sem exibir as fotos?” Se houver dúvida institucional específica, solicitar o canal responsável em vez de presumir uma autorização geral.

Milvus, RetinaFace, LVFace e TopoFR apareceram como sugestões na conversa. Cada mudança exigiria implementação e nova avaliação; obter primeiro a leitura de Gabriel sobre o trabalho efetivamente escrito.

## Depois: o que falta para a entrega institucional

Estes pontos **não impedem o envio do rascunho ao orientador**, mas precisam ser resolvidos antes da entrega oficial ou da divulgação indicada:

| Pendência | Em palavras simples | Como resolver / critério de conclusão |
|---|---|---|
| Aprovação do conteúdo | A compilação correta prova que o arquivo abre e que as referências e tabelas conferem; não substitui a leitura de Artur e Gabriel. | Incorporar as correções acordadas; guardar a versão revisada e a confirmação de que Gabriel autoriza encaminhá-la para a banca. |
| Regras aplicáveis ao seu TC | O PPC do curso e as regras locais dizem qual formato e quais etapas são aceitos para sua matrícula. | Conferir modelo e normas vigentes com Gabriel ou coordenação. Ajustar e validar o documento antes do protocolo de entrega. |
| Procedimentos finais | Banca, ficha catalográfica, assinaturas, ata e depósito podem exigir trâmites próprios; não são campos para inventar no rascunho. | Obter da coordenação a lista oficial e cumprir cada item no prazo informado. Registrar o que foi exigido e o que foi entregue. |
| Uso e divulgação dos dados | A monografia atual não mostra fotografias, recortes, embeddings ou rankings vinculáveis. Ainda é necessário saber se as condições das bases permitem o uso e eventual divulgação pretendidos. | Registrar origem e termos de Gallagher e LFW; confirmar exigências institucionais para dados faciais e depósito. Qualquer imagem de pessoa no texto ou publicação futura exige verificação própria antes de incluir. |
| Cópia de segurança | Git guarda código e documentos, mas não guarda as bases, pesos e saídas volumosas usadas nos resultados. Se o computador falhar, o Git sozinho não reconstrói a execução congelada. | Guardar uma **cópia privada em outro dispositivo ou armazenamento sob seu controle** da execução, dados e pesos necessários, junto ao código/documentação; conferir que a cópia abre e que os arquivos essenciais estão íntegros. Não colocar esse pacote em repositório público. |
| Artigo em inglês | Continua como trabalho derivado da pesquisa e da monografia após estabilizar escopo e resultados. | Preparar e submeter separadamente quando o texto estiver validado. Pela rota atual de defesa por monografia, aceite do artigo não é requisito para mandar o rascunho a Gabriel. Confirmar o enquadramento final com a coordenação. |

### Quanto reservar para a cópia de segurança?

Inventário anterior: **72 arquivos da execução (~78 MiB)**, `data/raw` (~3,58 GiB) e três pesos (~280 MiB). Com código e documentação, a reserva preliminar é **ao menos 5 GiB**, sujeita a conferência no dia da cópia. O diretório histórico separado não comprova backup da execução de 28/09. **Nenhuma cópia independente foi comprovada.** Isso é uma tarefa sua com meu apoio técnico, mas não deve atrasar o envio do rascunho ao orientador.

## Seu próximo passo agora

Depois de recompilar a fonte limpa, leia resumo, método, resultados e conclusão e anote dúvidas no formato **“página X: não entendi esta afirmação”**. O objetivo imediato é entregar ao orientador uma versão tecnicamente coerente e sem metadocumentação de rascunho; a versão de banca vem depois das correções dele.
