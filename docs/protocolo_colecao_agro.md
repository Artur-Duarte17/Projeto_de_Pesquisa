# Protocolo e resultados da coleção agro Agrishow 2022

Data de referência: **18 de setembro de 2026**.

## 1. Estado da decisão

A coleção **Agrishow 2022**, disponível no Wikimedia Commons, foi escolhida para um estudo de caso aplicado ao contexto agro. A fonte, os direitos de uso, os 124 originais, o gabarito, as consultas pareadas, os índices, a avaliação quantitativa e a análise qualitativa estão concluídos.

A afirmação sustentada é: **o sistema foi avaliado em um estudo de caso da Agrishow 2022, com uma pessoa pública, dez consultas de robustez definidas antes das respectivas buscas e um caso dirigido de chapéu e sombra**. Não é correto generalizar esse resultado para outros eventos, pessoas, máquinas ou para todo o domínio agro.

## 2. Por que esta coleção foi escolhida

A categoria pública reúne fotografias da cerimônia de abertura da 27ª Agrishow, realizada em 25 de abril de 2022. A auditoria da API do Wikimedia Commons encontrou:

| Item | Resultado auditado |
|---|---:|
| Arquivos na categoria | 127 |
| Licença | 127 arquivos em CC BY 2.0 |
| Autor institucional informado | Palácio do Planalto em 127 arquivos |
| Fotografias creditadas a Isac Nóbrega/PR | 119 |
| Fotografias creditadas a Anderson Riedel/PR | 8 |
| Tamanho total dos originais | 424,13 MiB |
| Títulos de versões recortadas | 3 |
| Originais selecionados após remover recortes | 124 |

A fonte apresenta um mesmo adulto público em várias fotografias do mesmo evento, em enquadramentos, distâncias, poses, grupos e condições de oclusão diferentes. Isso corresponde ao cenário pedido pelo orientador: usar uma fotografia como consulta e localizar a mesma pessoa em outras fotografias de um evento agro.

A pessoa recorrente descrita pela própria fonte é Jair Bolsonaro, então presidente da República. Para manter o experimento tecnicamente neutro, os arquivos experimentais usarão o identificador `agrishow2022_person_01`. O nome civil será usado apenas onde for necessário explicar a proveniência e conferir as anotações.

## 3. Direitos de uso e atribuição

A página individual auditada informa licença **Creative Commons Attribution 2.0 Generic (CC BY 2.0)** e registra que a licença do Flickr foi conferida pelo mecanismo FlickreviewR do Wikimedia Commons.

As condições mínimas para qualquer fotografia reproduzida no artigo são:

1. informar o fotógrafo indicado no arquivo;
2. atribuir a fonte institucional indicada;
3. incluir ligação ou referência à licença CC BY 2.0;
4. indicar recorte, marcação ou outra alteração visual realizada;
5. não sugerir apoio institucional ou pessoal ao trabalho.

A licença autoral permite reutilização com atribuição, mas não elimina os cuidados éticos próprios de reconhecimento facial. Por isso:

- somente a identidade de um adulto público será anotada como alvo;
- rostos de participantes comuns não serão identificados nominalmente;
- embeddings, recortes faciais e imagens baixadas permanecerão locais e fora do Git;
- o GitHub público conterá apenas código, protocolo e resultados agregados;
- o artigo mostrará somente as imagens essenciais à análise qualitativa, com atribuição completa;
- qualquer dúvida ética ou editorial deverá ser resolvida antes da submissão.

## 4. Prevenção de vazamento e viés

Três arquivos cujo título contém `(cropped)` são derivados de fotografias também presentes na categoria. Eles serão excluídos antes da indexação para impedir que versões quase idênticas da mesma imagem sejam tratadas como recuperações independentes.

A consulta não será escolhida depois de observar qual fotografia produz o melhor resultado. Por decisão do pesquisador, a fotografia-fonte será sorteada uma única vez entre as 91 fotografias marcadas como `present`. O sorteio usará uma semente registrada e a menor chave SHA-256 de `seed`, `target_id` e `image_id`, tornando a escolha verificável e reproduzível. Não será permitido sortear novamente com base na qualidade do ranking. O procedimento será:

1. inventariar e validar os 124 originais;
2. congelar o gabarito manual antes de qualquer busca;
3. sortear e congelar uma fotografia-fonte entre as 91 imagens `present`;
4. produzir um recorte facial da pessoa-alvo na fonte sorteada e manter a fotografia completa correspondente;
5. registrar `source_image_id`, caminho e SHA-256;
6. excluir a fotografia-fonte das abordagens facial, global e de fusão;
7. não alterar pesos ou anotações depois de conhecer o resultado.

O sorteio da EX-021 foi executado antes de qualquer busca, com a semente `823363566498139414`. Entre as 91 candidatas, a regra `minimum_sha256(seed, target_id, image_id)` selecionou `agrishow2022_52029027741` (`52029027741.jpg`). O SHA-256 do original é `9fa6366ac5aa1fc1e13c0ab2671dbe3b965d879fee0641ddcd9c5256100615af`. Essa fotografia será excluída de todos os rankings.

Na segunda etapa da EX-021, o InsightFace detectou 15 rostos na fotografia-fonte e produziu uma revisão numerada. A pessoa-alvo foi confirmada visualmente como o índice `1`, com caixa `(3224, 781, 3588, 1178)`. O recorte foi expandido de forma determinística para `(3009, 582, 3803, 1376)` e salvo em PNG sem perdas, com 794 × 794 pixels e SHA-256 `a0d47aed47dfaa46952ce5d17b6dc808df80bf45e8982ba089f3222e79ef020c`. Como o recorte contém partes de outros participantes, a consulta facial usa explicitamente o maior rosto; a nova detecção desse rosto apresentou sobreposição de `0,885247` com a caixa-alvo esperada. A consulta facial e a fotografia completa estão congeladas, sem que nenhuma busca tenha sido executada.

O procedimento pode ser auditado ou reproduzido por:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/prepare_agrishow_face_query.py --device cuda
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/prepare_agrishow_face_query.py --device cuda --target-face-index 1
```

O primeiro comando gera a revisão numerada. O segundo somente deve ser executado depois da confirmação visual do índice; por padrão, ele se recusa a sobrescrever uma consulta já congelada.

## 5. Anotação de relevância

Cada uma das 124 fotografias deve receber uma decisão manual sobre a presença da pessoa-alvo:

- `present`: a pessoa está visível e a identidade pode ser confirmada;
- `absent`: a pessoa não aparece;
- `uncertain`: visibilidade insuficiente para decisão segura.

Os casos `uncertain` precisam de uma segunda revisão antes do congelamento do protocolo. Nenhuma fotografia ambígua será convertida automaticamente em relevante ou irrelevante com base no ranking do próprio sistema. A relevância final deve ser definida independentemente das pontuações produzidas pelos modelos.

A primeira revisão marcou 89 fotografias como `present`, 33 como `absent` e duas como `uncertain`. A segunda revisão confirmou a presença da pessoa-alvo nas duas fotografias incertas, incluindo um caso em que o chapéu projeta sombra sobre o rosto. O gabarito congelado contém, portanto, **91 fotografias `present`, 33 `absent` e nenhuma `uncertain`**. O SHA-256 do CSV congelado é `f5d9de8155fb5db61b32f0dc6aca4fc75e4e66740152ce74d06cd134d403edf1`.

O script `scripts/data/prepare_agrishow_review.py` valida os 124 arquivos contra o inventário, cria miniaturas locais e prepara uma galeria para a primeira revisão manual. A galeria permite marcar `present`, `absent` ou `uncertain`, registrar observações, abrir o original quando necessário e exportar o CSV. O arquivo exportado deve substituir `data/annotations/agrishow_2022_presence_review.csv` somente depois que as 124 decisões tiverem sido preenchidas. Tanto as miniaturas quanto as anotações permanecem fora do Git.

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/prepare_agrishow_review.py
```

Depois da resolução dos casos incertos, o congelamento é validado e registrado por:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/freeze_agrishow_annotations.py
```

O CSV congelado deverá conter, no mínimo:

| Coluna | Conteúdo |
|---|---|
| `image_id` | identificador estável da fotografia original |
| `target_id` | `agrishow2022_person_01` |
| `presence` | `present`, `absent` ou `uncertain` |
| `review_status` | estado da revisão humana |
| `notes` | observação curta sobre oclusão, escala ou dúvida |

## 6. Protocolo e resultado experimental

A unidade recuperada continuará sendo a fotografia completa. A consulta facial será um recorte da pessoa-alvo; a consulta global será a fotografia-fonte completa. Serão comparadas as mesmas cinco configurações já congeladas no Gallagher:

1. somente face;
2. somente contexto global;
3. fusão com pesos 0,9 para face e 0,1 para contexto;
4. fusão com pesos 0,7 para face e 0,3 para contexto;
5. fusão com pesos 0,5 para face e 0,5 para contexto.

Os pesos não serão ajustados usando o resultado da Agrishow. As métricas serão Precision@5, Precision@10, Recall@5, Recall@10 e mAP sobre o ranking integral. Também serão relatados:

- quantidade de imagens indexadas e de imagens com face detectada;
- número de fotografias relevantes depois da exclusão da fonte;
- falhas de leitura, detecção e extração;
- tamanho do rosto, pose, chapéu, oclusão e densidade de pessoas nos casos selecionados;
- exemplos qualitativos de acerto e erro com atribuição das imagens.

Como se trata de uma coleção pequena e de uma pessoa principal, seus resultados serão apresentados como **estudo de caso aplicado**, não como prova geral de desempenho em todo o domínio agro.

A EX-022 examinou os 124 originais sem falha de leitura. O índice global contém 124 descritores de 2.048 dimensões. O índice facial contém 1.590 rostos distribuídos por 123 fotografias; a única imagem sem face detectada estava marcada como ausente, então nenhuma fotografia relevante foi perdida. A fonte sorteada está presente nos dois índices.

A EX-023 excluiu a fotografia-fonte, deixando 123 candidatas e 90 relevantes. O ranking integral e o Top-10 foram auditados, sem vazamento da fonte, duplicidade ou valor não finito.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| 0,0 / 1,0 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,797298 |
| 0,9 / 0,1 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,998235 |
| 0,7 / 0,3 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,997898 |
| 0,5 / 0,5 | 1,000000 | 1,000000 | 0,055556 | 0,111111 | 0,993485 |

Os dez primeiros resultados foram relevantes em todos os métodos. O Recall@K é baixo porque existem 90 imagens relevantes: cinco acertos representam `5/90` e dez acertos representam `10/90`. Face e fusão 0,9/0,1 empataram em mAP. O contexto isolado ainda organizou grande parte das imagens relevantes, provavelmente por terem sido produzidas no mesmo evento, mas ficou abaixo do sinal facial. Aumentar seu peso não melhorou o baseline facial.

A EX-024 recalculou os cinco rankings integrais e reproduziu exatamente o Top-10 e a AP da EX-023. Na busca facial, o primeiro item irrelevante apareceu na posição 90; no contexto global, apareceu na posição 17. A fusão 0,9/0,1 manteve a AP da face, mas compartilhou nove das dez primeiras imagens com o baseline, demonstrando que houve reordenação interna.

A figura qualitativa usa o recorte-consulta e as cinco primeiras fotografias recuperadas pela busca facial. As cinco imagens são relevantes e as caixas amarelas foram inspecionadas sobre o rosto correto. O painel cobre fotografia de grupo, cavalgada, fundo institucional e palco, com seis registros individuais de fotógrafo, fonte, licença, página original e modificação. Após o encerramento experimental, a cópia canônica da figura permanece privada em `publication_private/principia/figuras/agrishow_face_top5.png`, acompanhada de atribuições, legenda e manifesto.

A EX-025 mostrou por que o Top-K precisa ser interpretado com cautela: com 90 relevantes em 123 candidatas, a AP aleatória esperada é 0,741369, e um ranking aleatório tem probabilidade exata de 20,34% de acertar todas as cinco primeiras posições e 3,81% de acertar todas as dez primeiras. Em 200.000 permutações, nenhuma atingiu a AP facial de 0,998235; 14.348 atingiram ou superaram a AP global de 0,797298.

A EX-026 manteve a consulta inicial e selecionou nove fontes adicionais por ordem SHA-256 com semente registrada, antes das buscas. Uma fonte foi recusada na revisão facial prévia porque a pessoa-alvo não podia ser confirmada; sem consultar rankings, ela foi substituída pela próxima fonte ainda não usada na ordem congelada. O protocolo final possui dez consultas, 123 candidatas e 90 relevantes por consulta.

| Configuração face/global | Precision@5 | Precision@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| 1,0 / 0,0 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,987261 |
| 0,0 / 1,0 | 0,940000 | 0,920000 | 0,052222 | 0,102222 | 0,850717 |
| 0,9 / 0,1 | 0,980000 | 0,980000 | 0,054444 | 0,108889 | 0,986167 |
| 0,7 / 0,3 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,980514 |
| 0,5 / 0,5 | 0,980000 | 0,990000 | 0,054444 | 0,110000 | 0,960858 |

Somente face obteve o maior mAP médio. O contexto isolado degradou as dez consultas em relação à face. As fusões melhoraram uma consulta cada, mas também degradaram a maioria delas; portanto, o contexto pode ajudar casos específicos, porém não melhorou o resultado médio.

A EX-027 avaliou separadamente a fonte previamente anotada por chapéu e sombra. A face obteve mAP de 0,997102; o contexto isolado, 0,826134; e as fusões 0,9/0,1, 0,7/0,3 e 0,5/0,5 obtiveram 0,997506, 0,997749 e 0,988609. O pequeno ganho de 0,000647 da fusão 0,7/0,3 sobre a face é específico desse caso dirigido e não altera a conclusão agregada.

Mesmo após essa ampliação, todas as consultas representam a mesma pessoa no mesmo evento, com alta prevalência de relevantes e fotografias possivelmente correlacionadas. O resultado é evidência aplicada no cenário estudado, não validação geral do domínio agro.

## 7. Aquisição reproduzível

O script `scripts/data/download_agrishow_2022.py` consulta a API do Wikimedia Commons, exige CC BY 2.0 para todos os itens, remove por padrão os três recortes derivados e produz inventário com URL, autoria, licença, dimensões, tamanho e SHA-1.

Sem a opção de download, ele audita somente metadados:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/download_agrishow_2022.py
```

O download integral é deliberadamente explícito:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/data/download_agrishow_2022.py --download-images --workers 1 --request-delay 8
```

O download é sequencial e inclui uma pausa entre solicitações para respeitar os limites do servidor. Se o Wikimedia mantiver uma limitação HTTP 429 após as tentativas espaçadas, a execução é interrompida em vez de solicitar os demais arquivos. Cada arquivo baixado é aceito somente quando tamanho e SHA-1 coincidem com o registro do Wikimedia. A operação é retomável: arquivos já válidos são reutilizados. Os dados ficam em `data/raw/agrishow_2022`, pasta ignorada pelo Git.

## 8. Fontes primárias

- Categoria da coleção: <https://commons.wikimedia.org/wiki/Category:Agrishow_(2022)>.
- Exemplo de descrição, autoria e licença: <https://commons.wikimedia.org/wiki/File:25_04_2022_Cerim%C3%B4nia_de_abertura_da_27%C2%BA_Agrishow_-_Feira_Internacional_de_Tecnologia_Agr%C3%ADcola_em_A%C3%A7%C3%A3o_(52027983552).jpg>.
- Texto da licença CC BY 2.0: <https://creativecommons.org/licenses/by/2.0/>.

## 9. Próximas execuções

| ID | Atividade | Responsável pela execução | Situação |
|---|---|---|---|
| EX-019 | baixar e validar os 124 originais | Artur | concluída e aprovada; 124 arquivos, 420,43 MiB, zero falhas |
| EX-020 | revisar inventário e congelar presença da pessoa-alvo | Artur, com conferência | concluída e congelada; 91 presentes, 33 ausentes, zero incertos |
| EX-021 | criar consultas pareadas e relevância | preparação e conferência concluídas | concluída; fonte, recorte facial, exclusão e hashes congelados antes das buscas |
| EX-022 | indexar faces e descritores globais | Artur | concluída e aprovada; 1.590 faces em 123 imagens e 124 descritores globais |
| EX-023 | avaliar os cinco métodos | execução e auditoria concluídas | concluída e aprovada; fonte excluída, 123 candidatas e 90 relevantes |
| EX-024 | analisar erros e gerar figura qualitativa | execução e inspeção concluídas | concluída e aprovada; rankings reproduzidos, figura e seis atribuições validadas |
| EX-025 | comparar a consulta inicial com rankings aleatórios | execução e auditoria concluídas | concluída e aprovada; 200.000 permutações e probabilidades exatas de P@K |
| EX-026 | avaliar robustez em dez consultas | preparação, revisão e auditoria concluídas | concluída e aprovada; dez fontes, 50 rankings e 500 resultados Top-10 |
| EX-027 | avaliar o caso dirigido de chapéu e sombra | preparação, revisão e auditoria concluídas | concluída e aprovada; consulta separada da amostra de robustez |

## 10. Encerramento e reprodução

A EX-028 preservou no dossiê privado os resultados, decisões, limitações, hashes e 30 evidências textuais pequenas. Também preservou em `publication_private/principia/figuras` a figura qualitativa final e seus arquivos de atribuição. Em seguida, removeu 45 alvos locais pesados ou intermediários, incluindo as fotografias baixadas, recortes, páginas de revisão, índices, rankings, saídas experimentais e utilitários temporários das EX-019 a EX-027. Foram removidos 410 arquivos, que somavam 524.304.264 bytes; a diferença observada de espaço livre no volume foi de 525.447.168 bytes.

Essa limpeza não invalida os resultados: os CSVs públicos de consulta e relevância, os scripts do projeto, os documentos canônicos, os hashes, a proveniência Wikimedia e o histórico Git permitem reconstruir o ciclo. Uma nova execução deve baixar novamente os 124 originais, verificar tamanho e SHA-1 contra o inventário de origem e seguir, em ordem, o protocolo registrado neste documento. Os resultados continuam limitados à pessoa e ao evento avaliados.
