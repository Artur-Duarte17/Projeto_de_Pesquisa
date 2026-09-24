# Relatorio do estado atual do projeto

Data do relatorio: 24/06/2026

## 1. Resumo em linguagem simples

O projeto esta construindo um sistema para procurar fotografias em um acervo usando inteligencia artificial.

Em vez de procurar por nome de arquivo ou por texto, o sistema olha para o conteudo da imagem. Hoje ele trabalha com duas ideias principais:

1. **Busca facial**: recebe uma foto de uma pessoa, detecta o rosto, transforma esse rosto em um vetor numerico chamado **embedding facial** e procura outros rostos parecidos no acervo.
2. **Busca global da imagem inteira**, tambem chamada de **CBIR global**: recebe uma imagem completa, transforma a imagem em um vetor numerico chamado **embedding global** e procura imagens visualmente parecidas.

O objetivo cientifico da v1 e comparar tres formas de recuperacao fotografica:

1. Busca por rosto.
2. Busca pela imagem inteira.
3. Fusao entre busca por rosto e busca pela imagem inteira.

O sistema ja tem resultados experimentais nas tres partes. A busca facial funcionou bem para encontrar pessoas. A busca global possui uma avaliacao auxiliar na **INRIA Holidays**, calculada com a metrica adaptada do projeto. A fusao simples foi testada no protocolo Gallagher corrigido, mas nao superou a busca facial isolada no cenario de procurar pessoas.

Isso nao invalida o projeto. Pelo contrario: mostra um resultado importante para artigo, porque indica que cada tecnica serve melhor para um tipo de consulta.

## 2. Objetivo atual do projeto

O objetivo oficial da v1 e:

> Construir e avaliar um sistema reprodutivel de recuperacao fotografica que compara busca facial, busca global por imagem inteira e fusao face + global.

Esse objetivo esta alinhado com a proposta porque envolve:

- deteccao e reconhecimento facial;
- extracao de descritores visuais globais;
- indices separados para vetores faciais e globais;
- comparacao por similaridade;
- avaliacao com metricas como **Precision@K**, **Recall@K** e **mAP**;
- possibilidade de interface minima para demonstracao.

## 3. Organizacao atual do codigo

Os scripts novos foram separados por finalidade:

```text
scripts/
  face/
    01_index_faces.py
    02_search_person.py
    03_evaluate_face.py
    04_prepare_gallagher_eval.py
    05_analyze_face_errors.py

  global/
    01_index_global_resnet.py
    02_search_global.py
    03_evaluate_global.py

  fusion/
    01_fusion_search.py
    02_evaluate_fusion.py

  data/
    download_gallagher.py
    download_holidays.py

  app/
    streamlit_app.py

  retrieval_common.py
  face_lib.py
  global_lib.py
  fusion_lib.py
```

Essa separacao e importante porque evita misturar tarefas diferentes:

- `face/` cuida da busca por pessoas.
- `global/` cuida da busca por imagem inteira.
- `fusion/` combina os resultados.
- `data/` baixa e prepara bases de teste.
- `app/` fica reservado para a interface.

## 4. Git, GitHub e tamanho do repositorio

O projeto foi preparado para nao subir bases, modelos, embeddings e outputs grandes para o GitHub.

Isso e importante porque projetos de visao computacional costumam gerar milhares de imagens, arquivos `.npy`, modelos e relatorios pesados. Esses arquivos devem ficar localmente ou em armazenamento proprio, nao versionados no Git.

Hoje o Git versiona principalmente:

- scripts;
- configuracoes;
- requirements;
- README;
- documentacao.

E deixa fora do Git:

- `data/raw/`;
- `data/query/`;
- `outputs/`;
- embeddings;
- checkpoints;
- relatorios gerados automaticamente.

O repositorio remoto usado e:

```text
https://github.com/Artur-Duarte17/Projeto_de_Pesquisa.git
```

Ultimos commits relevantes:

```text
4eac234 Use CUDA PyTorch for global retrieval
edf4449 Add INRIA Holidays dataset workflow
486f458 Report effective torch device for global retrieval
a484055 Add face retrieval error analysis
24703a0 Add Gallagher face evaluation preparation
0bffafd Add Gallagher dataset downloader
491116b Add face evaluation query protocol
```

## 5. Ambiente de execucao

O ambiente principal usado foi:

```text
C:\Projeto_de_Pesquisa\laboratorio\cibir_face_new
```

O modulo facial usa **InsightFace** com **ONNX Runtime GPU**.

O modulo global usa **PyTorch** e **Torchvision** com suporte CUDA:

```text
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
CUDA disponivel: True
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

Isso significa que a maquina consegue usar a GPU para acelerar partes do processamento.

Observacao importante: a GPU ja esta funcionando, mas o ganho no modulo global ainda e limitado porque o script atual processa imagens uma por vez. Para acelerar melhor, o proximo ajuste tecnico seria processar imagens em **lotes**, tambem chamado de **batch processing**.

## 6. Bases de dados usadas

### 6.1 LFW

**LFW**, ou **Labeled Faces in the Wild**, e uma base classica de rostos usada para validar reconhecimento facial.

No projeto, ela foi usada como validacao facial inicial.

Indice facial gerado:

```text
Pasta de entrada:
C:\Projeto_de_Pesquisa\data\raw\lfw\lfw_home\lfw_funneled

Saida:
C:\Projeto_de_Pesquisa\outputs\face_index
```

Resultado do indice:

```text
Imagens com faces indexadas: 13.185
Embeddings faciais: 16.058
Dimensao de cada embedding facial: 512
```

Como uma imagem pode ter mais de um rosto detectado, o numero de embeddings pode ser maior que o numero de imagens.

### 6.2 Gallagher

A base **Gallagher** foi adicionada porque ela e mais parecida com o problema real do projeto: varias pessoas aparecem em fotos comuns, como eventos, albuns e grupos.

Ela e util porque permite testar:

- deteccao de varios rostos na mesma foto;
- busca por pessoa em um conjunto fotografico;
- retorno da foto inteira, mesmo quando a busca encontra apenas um rosto;
- comparacao entre busca facial e busca global no mesmo acervo.

Dados locais:

```text
Imagens: 589
Faces anotadas: 931
Identidades: 32
```

Indice facial Gallagher:

```text
Saida:
C:\Projeto_de_Pesquisa\outputs\experiments\ex-010_gallagher_face_index

Imagens com faces detectadas: 587
Embeddings faciais: 1.303
Dimensao de cada embedding facial: 512
```

Indice global Gallagher:

```text
Saida:
C:\Projeto_de_Pesquisa\outputs\global_index_gallagher

Imagens indexadas: 589
Embeddings globais: 589
Dimensao de cada embedding global: 2048
```

### 6.3 INRIA Holidays

A base **INRIA Holidays** foi adicionada para avaliar corretamente a parte de **CBIR global**, isto e, busca por similaridade visual da imagem inteira.

Ela e uma base classica de recuperacao de imagens. As imagens sao organizadas em grupos. A primeira imagem de cada grupo e usada como consulta, e as outras imagens do mesmo grupo sao consideradas relevantes.

Dados locais:

```text
Imagens: 1.491
Grupos/consultas: 500
Relacoes de relevancia: 991
```

Indice global Holidays:

```text
Saida:
C:\Projeto_de_Pesquisa\outputs\experiments\ex-013_holidays_global_index

Imagens indexadas: 1.491
Embeddings globais: 1.491
Dimensao de cada embedding global: 2048
```

## 7. O que e um embedding

Um **embedding** e uma representacao numerica de uma imagem ou rosto.

Em linguagem simples: o modelo de inteligencia artificial olha para a imagem e gera uma lista de numeros que resume as caracteristicas importantes dela.

No projeto existem dois tipos:

1. **Embedding facial**: representa um rosto. E usado para reconhecer se dois rostos parecem ser da mesma pessoa.
2. **Embedding global**: representa a imagem inteira. E usado para encontrar imagens visualmente parecidas.

Depois que os embeddings sao gerados, o sistema compara esses vetores usando **similaridade cosseno**. Quanto maior a similaridade, mais parecidos os vetores sao considerados.

## 8. Busca facial

### 8.1 Como funciona

O fluxo da busca facial e:

```text
foto de consulta
-> detectar rosto
-> gerar embedding facial
-> comparar com todos os embeddings faciais do indice
-> agrupar resultados por foto inteira
-> retornar as fotos completas onde a pessoa aparece
```

O ponto principal e que o sistema nao retorna apenas o recorte do rosto. Ele retorna a foto inteira, porque esse e o comportamento esperado pelo usuario.

### 8.2 Por que uma imagem pode ter varios embeddings

Uma unica foto pode ter varias pessoas. Por isso, o indice facial nao deve ser:

```text
uma imagem = um embedding
```

O correto e:

```text
uma imagem = varios rostos detectados
cada rosto = um embedding proprio
```

Isso permite encontrar uma pessoa mesmo quando ela aparece em uma foto de grupo.

### 8.3 Resultado facial vigente no Gallagher

A avaliacao facial vigente usa as 20 consultas corrigidas da EX-033/EX-034 e o mesmo universo global da EX-035. O gabarito vem das anotacoes cruzadas com 589 fotografias, independentemente de o detector ter encontrado rosto. Depois de excluir a fonte, cada ranking possui 588 candidatas.

Metricas:

```text
Precision@5:  0.9000
Precision@10: 0.6800
Recall@5:     0.5771
Recall@10:    0.6879
mAP:          0.9514
```

Interpretacao simples:

- **Precision@5 = 0.9000** significa que, entre as 5 primeiras fotos retornadas, a maior parte estava correta.
- **Precision@10 = 0.6800** usa denominador fixo dez e inclui identidades com menos de dez respostas relevantes.
- **Recall@10 = 0.6879** significa que, dentro do limite de 10 resultados, o sistema encontrou uma parte relevante das fotos corretas.
- **mAP = 0.9514** mostra que os resultados corretos tendem a aparecer bem ranqueados neste protocolo.

As duas fotos da identidade 2 sem face detectada permanecem relevantes. A consulta `gallagher_id6_q1` agora usa a caixa que contem o ponto medio dos olhos anotados, em vez do rosto vizinho. Os valores anteriores da EX-012 e EX-017 permanecem historicos.

Conclusao delimitada: a busca facial funciona bem para encontrar pessoas nas 20 consultas avaliadas; isso nao demonstra generalizacao para outros albuns ou eventos.

## 9. Busca global com ResNet50

### 9.1 Como funciona

A busca global usa a imagem inteira, nao apenas o rosto.

O fluxo e:

```text
imagem de consulta
-> ResNet50 pre-treinada
-> embedding global de 2048 dimensoes
-> comparacao por similaridade cosseno
-> retorno das imagens visualmente parecidas
```

O modelo usado como baseline global e:

```text
ResNet50 pre-treinada no ImageNet
```

Esse modelo e adequado como baseline academico porque e uma CNN conhecida e amplamente usada em visao computacional.

### 9.2 Resultado global no INRIA Holidays

A avaliacao global local no INRIA Holidays usou 500 consultas. Ela e auxiliar: a AP do projeto e calculada de forma diferente da integracao oficial do benchmark.

Metricas:

```text
Precision@5:  0.3240
Precision@10: 0.1776
Recall@5:     0.8702
Recall@10:    0.9176
mAP:          0.8426
Tempo medio do ranking: 0,704 ms por consulta
```

Interpretacao simples:

- O **mAP local = 0.8426** indica bom ordenamento segundo a metrica adaptada do projeto, mas nao deve ser comparado diretamente ao mAP oficial da Holidays.
- O **Recall@10 = 0.9176** mostra que o sistema encontrou a maioria das imagens relevantes dentro dos 10 primeiros resultados.
- A Precision@10 parece baixa, mas isso acontece porque muitos grupos do Holidays tem apenas 1 ou 2 imagens relevantes. Se a consulta tem poucas respostas corretas possiveis, o restante do Top-10 conta como falso positivo.
- O tempo informado mede somente similaridade e ordenacao com descritores ja calculados; nao inclui a extracao do descritor de uma imagem nova.

Conclusao: o modulo global com ResNet50 apresenta evidencia auxiliar positiva para similaridade visual. Nenhuma nova avaliacao Holidays foi executada na correcao A01/A02.

### 9.3 Resultado global contextual no Gallagher para buscar pessoa

O protocolo atual usa a fotografia-fonte completa como consulta global e avalia se o contexto ajuda a recuperar outras fotografias que contêm a pessoa selecionada.

Metricas:

```text
Precision@5:  0.2600
Precision@10: 0.1950
Recall@5:     0.2035
Recall@10:    0.2347
mAP:          0.280864
```

Interpretacao:

O resultado contextual e superior ao experimento historico que comparava um recorte de rosto com fotografias completas, mas continua baixo porque semelhanca de ambiente, objetos e composicao nao determina a identidade da pessoa.

Conclusao: CBIR global nao substitui reconhecimento facial quando a tarefa e encontrar uma pessoa especifica.

## 10. Fusao face + global

### 10.1 O que e fusao

Fusao significa combinar dois sinais diferentes:

1. O score facial, vindo da comparacao entre rostos.
2. O score global, vindo da comparacao entre imagens inteiras.

A fusao simples testada foi:

```text
score_final = peso_face * score_face + peso_global * score_global
```

Pesos testados:

```text
0.9 face / 0.1 global
0.7 face / 0.3 global
0.5 face / 0.5 global
```

### 10.2 Resultado vigente da fusao no Gallagher

O protocolo pareado corrigido usa o recorte ligado ao alvo anotado para a busca facial e a fotografia-fonte completa para o contexto global. A mesma fonte e excluida do ranking e da relevancia em todas as configuracoes.

Tabela comparativa:

| Metodo | Precision@5 | Precision@10 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| Face isolado | 0.9000 | 0.6800 | 0.6879 | 0.951446 |
| Contexto global isolado | 0.2600 | 0.1950 | 0.2347 | 0.280864 |
| Fusao 0.9/0.1 | 0.9000 | 0.6800 | 0.6879 | 0.946297 |
| Fusao 0.7/0.3 | 0.8400 | 0.6750 | 0.6829 | 0.901525 |
| Fusao 0.5/0.5 | 0.7700 | 0.6300 | 0.6349 | 0.813358 |

### 10.3 Interpretacao da fusao

Neste cenario corrigido, a fusao simples nao melhorou o mAP do resultado facial.

Na comparacao pareada, o componente facial recebe o recorte do alvo e o componente global recebe a fotografia-fonte completa. O contexto da cena pode ajudar ou prejudicar consultas individuais, mas, em media, nao superou o sinal facial neste conjunto.

Essa conclusao e boa para o artigo porque mostra uma comparacao experimental real:

```text
Facial funciona melhor para identidade.
Global possui evidencia auxiliar para similaridade visual geral.
Fusao simples nem sempre melhora; depende do tipo de consulta e do cenario.
```

Uma melhoria futura seria testar uma fusao mais inteligente, como **reranking**:

```text
1. primeiro buscar por face;
2. depois usar o global apenas para reorganizar resultados proximos;
3. evitar que o global derrube resultados faciais fortes.
```

## 11. Analise de erros

Foi criado tambem um script de analise de erros para a busca facial:

```text
scripts/face/05_analyze_face_errors.py
```

Ele gera:

- falsos positivos;
- verdadeiros positivos;
- queries mais dificeis;
- exemplos visuais.

Saidas principais:

```text
C:\Projeto_de_Pesquisa\outputs\reports\gallagher_face\error_analysis
```

Essa etapa e importante para artigo porque nao basta mostrar apenas numeros. Tambem e necessario explicar onde o sistema acerta, onde erra e por que erra.

## 12. Interface Streamlit

Ja existe um script de interface minima:

```text
scripts/app/streamlit_app.py
```

Ela ainda nao deve ser tratada como o centro cientifico do projeto. A interface serve para demonstrar o sistema.

Prioridade correta:

```text
1. validar os algoritmos;
2. medir resultados;
3. comparar metodos;
4. depois melhorar interface.
```

Neste momento, a interface pode entrar como proxima etapa, porque os principais experimentos ja foram executados.

## 13. Onde estamos no planejamento

Status atual:

| Etapa | Status |
| --- | --- |
| Organizar Git/GitHub e ignorar arquivos grandes | Feito |
| Criar indice facial multi-rosto | Feito |
| Buscar pessoa e retornar foto inteira | Feito |
| Avaliar busca facial | Feito |
| Baixar/preparar Gallagher | Feito |
| Baixar/preparar INRIA Holidays | Feito |
| Criar indice global ResNet50 | Feito |
| Avaliar CBIR global | Feito |
| Ativar PyTorch com GPU/CUDA | Feito |
| Avaliar fusao face + global | Feito com protocolo pareado atual |
| Gerar relatorio/documentacao | Feito |
| Interface Streamlit minima | Proximo passo recomendado |
| Material final para artigo | Proximo passo depois da interface/exemplos |

Em termos do cronograma original, o projeto esta no fim da etapa de fusao e entrando na etapa de demonstracao e escrita.

## 14. Estado cientifico do projeto

O projeto ja tem um nucleo experimental corrigido e documentado, porque possui:

- um problema claro: recuperacao fotografica por conteudo;
- duas trilhas tecnicas: face e CBIR global;
- bases diferentes para validar cada trilha;
- metricas quantitativas;
- comparacao entre metodos;
- analise de erro;
- scripts reproduziveis;
- Git/GitHub organizado.

O ponto mais importante para o artigo e nao vender a fusao como se ela sempre melhorasse. O resultado correto e:

> Na avaliacao Gallagher corrigida, a fusao simples nao superou o modulo facial no cenario de busca por identidade. A Holidays fornece avaliacao auxiliar do componente global com metrica adaptada, sem comparacao direta ao benchmark oficial.

Essa frase e sustentada pelos artefatos atuais, dentro das limitacoes descritas. Agrishow permanece estudo historico/complementar privado e nao integra a conclusao principal.

## 15. O que ainda falta

### 15.1 Curto prazo

Proximas tarefas obrigatorias antes da redacao ou submissao:

1. Obter alinhamento da coordenacao/orientacao sobre modalidade, calendario, aceite, banca e deposito institucional.
2. Resolver licencas, base de tratamento e decisao institucional/etica sobre biometria e divulgacao.
3. Revisar referencias, metadados e delimitacao da contribuicao com validacao humana dos autores.
4. Confirmar compatibilidade editorial e politica de repositorio institucional.
5. Definir o que pode ser divulgado; fotografias e recortes nao devem ser publicados automaticamente.

A interface e outras melhorias tecnicas podem ser retomadas depois dessas decisoes. Nenhum artigo foi iniciado neste fechamento documental.

### 15.2 Medio prazo

Melhorias possiveis:

1. Implementar fusao por reranking.
2. Testar CLIP como descritor global alternativo.
3. Testar busca por duas pessoas na mesma foto.
4. Comparar a ResNet50 com outro descritor global sob o mesmo protocolo.
5. Melhorar a interface.

### 15.3 Fora da v1

Itens que devem ficar para depois:

- deploy online;
- interface polida;
- segmentacao integrada ao fluxo principal;
- FAISS para indice grande;
- treinamento de modelo novo;
- PIPA como dependencia obrigatoria.

## 16. Conclusao

O projeto saiu de um estado antigo e pouco claro para uma estrutura mais organizada, reproduzivel e avaliavel.

Hoje existem resultados concretos para:

```text
busca facial;
busca global;
fusao face + global;
analise de erros;
uso de GPU;
preparo de bases;
controle de arquivos grandes fora do Git.
```

A conclusao tecnica atual, delimitada, e:

```text
A busca facial obteve o maior mAP nas 20 consultas Gallagher corrigidas.
O CBIR global com ResNet50 possui avaliacao auxiliar para similaridade visual, com metrica Holidays adaptada.
A fusao simples entre face e contexto global nao superou, em media, a busca facial no Gallagher corrigido.
Agrishow permanece historico/complementar privado e nao sustenta a conclusao principal.
```

Portanto, A01/A02 e a consolidacao experimental Gallagher estao encerradas, mas o projeto ainda nao esta pronto para submissao. Permanecem pendencias institucionais, eticas, documentais, bibliograficas e editoriais. A redacao academica deve comecar somente depois das decisoes correspondentes e da validacao humana dos autores.
