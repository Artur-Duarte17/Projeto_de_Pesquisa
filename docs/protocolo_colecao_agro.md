# Protocolo da coleção agro candidata

Data de referência: **16 de setembro de 2026**.

## 1. Estado da decisão

A coleção **Agrishow 2022**, disponível no Wikimedia Commons, foi escolhida como candidata principal para a validação aplicada ao contexto agro. A escolha da fonte e a auditoria de direitos de uso estão concluídas; o download integral, a anotação da pessoa-alvo e os experimentos ainda não foram executados.

Consequentemente, este documento não autoriza afirmar que o sistema já foi validado no agro. A afirmação correta, até a conclusão das próximas execuções, é: **foi identificada uma coleção agro aberta e foi definido um protocolo de validação**.

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

A consulta não será escolhida depois de observar qual fotografia produz o melhor resultado. O procedimento será:

1. inventariar e validar os 124 originais;
2. definir critérios de qualidade da consulta antes de executar a busca;
3. escolher e congelar uma fotografia-fonte representativa;
4. produzir um recorte facial da pessoa-alvo e manter a fotografia completa correspondente;
5. registrar `source_image_id`, caminho e SHA-256;
6. excluir a fotografia-fonte das abordagens facial, global e de fusão;
7. não alterar pesos ou anotações depois de conhecer o resultado.

## 5. Anotação de relevância

Cada uma das 124 fotografias deve receber uma decisão manual sobre a presença da pessoa-alvo:

- `present`: a pessoa está visível e a identidade pode ser confirmada;
- `absent`: a pessoa não aparece;
- `uncertain`: visibilidade insuficiente para decisão segura.

Os casos `uncertain` precisam de uma segunda revisão antes do congelamento do protocolo. Nenhuma fotografia ambígua será convertida automaticamente em relevante ou irrelevante com base no ranking do próprio sistema. A relevância final deve ser definida independentemente das pontuações produzidas pelos modelos.

O CSV congelado deverá conter, no mínimo:

| Coluna | Conteúdo |
|---|---|
| `image_id` | identificador estável da fotografia original |
| `target_id` | `agrishow2022_person_01` |
| `presence` | `present`, `absent` ou `uncertain` |
| `review_status` | estado da revisão humana |
| `notes` | observação curta sobre oclusão, escala ou dúvida |

## 6. Protocolo experimental planejado

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
| EX-019 | baixar e validar os 124 originais | Artur | em andamento; 2 arquivos válidos preservados após limitação HTTP 429, retomada sequencial preparada |
| EX-020 | revisar inventário e congelar presença da pessoa-alvo | Artur, com conferência | pendente |
| EX-021 | criar consultas pareadas e relevância | Artur executa; código versionado | pendente |
| EX-022 | indexar faces e descritores globais | Artur | pendente |
| EX-023 | avaliar os cinco métodos | Artur | pendente |
| EX-024 | analisar erros e gerar figura qualitativa | Artur | pendente |
