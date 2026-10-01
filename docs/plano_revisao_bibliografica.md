# Plano de revisao bibliografica

Data: 25/06/2026

## 1. Objetivo deste documento

Este documento organiza a pesquisa bibliografica necessaria para transformar o projeto em artigo, TCC ou relatorio final.

O projeto ja possui implementacao e resultados experimentais. Agora precisamos fortalecer a fundamentacao teorica com artigos, surveys, bases de dados e referencias tecnicas confiaveis.

O objetivo da revisao bibliografica e sustentar a seguinte narrativa:

> Este trabalho implementa e avalia um sistema de recuperacao fotografica por conteudo, comparando reconhecimento facial, CBIR global com ResNet50 e fusao simples de scores.

## 2. Regra principal

A Pesquisa Profunda do ChatGPT, Google Scholar, Semantic Scholar e outras ferramentas podem ser usadas para encontrar trabalhos.

Mas a citacao final deve ser sempre do material original:

- artigo cientifico;
- livro;
- documentacao tecnica oficial;
- pagina oficial de dataset;
- proceedings de conferencia;
- repositorio oficial quando for necessario documentar ferramenta ou base.

Nao cite "ChatGPT" como fonte tecnica do artigo.

## 3. Temas que precisam de referencias

### 3.1 Recuperacao de imagens por conteudo

Termos para pesquisar:

```text
content based image retrieval
CBIR survey
image retrieval deep learning survey
visual image retrieval
query by image content
```

Perguntas que a bibliografia deve responder:

- O que e CBIR?
- Por que buscar por conteudo visual e diferente de buscar por metadados?
- Como a area evoluiu de descritores manuais para deep learning?
- Quais metricas sao usadas em recuperacao de imagens?

Onde entra no artigo:

```text
Introducao
Trabalhos relacionados
Fundamentacao teorica
```

Referencias iniciais:

- Latif et al. A Decade Survey of Content Based Image Retrieval using Deep Learning.
- Liu et al. Deep Learning for Instance Retrieval: A Survey.

### 3.2 CNNs pre-treinadas como extratores globais

Termos para pesquisar:

```text
pretrained CNN feature extraction image retrieval
ResNet50 feature extractor image retrieval
deep features for image retrieval
transfer learning visual retrieval
```

Perguntas que a bibliografia deve responder:

- Por que usar uma CNN pre-treinada como extrator de caracteristicas?
- Por que ResNet50 e uma escolha razoavel como baseline?
- O que significa usar embeddings globais da imagem inteira?

Onde entra no artigo:

```text
Metodologia
Modulo CBIR global
Configuracao experimental
```

Referencia obrigatoria:

- He et al. Deep Residual Learning for Image Recognition.

### 3.3 Reconhecimento facial por embeddings

Termos para pesquisar:

```text
face recognition embeddings
ArcFace face recognition
InsightFace ArcFace
deep face recognition survey
face verification embeddings cosine similarity
```

Perguntas que a bibliografia deve responder:

- O que e reconhecimento facial baseado em embeddings?
- Como embeddings faciais permitem comparar identidade?
- Por que similaridade cosseno e comum nesse tipo de sistema?
- Qual a importancia de modelos como ArcFace?

Onde entra no artigo:

```text
Fundamentacao teorica
Modulo de reconhecimento facial
Metodologia
```

Referencia obrigatoria:

- Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

### 3.4 Fusao de scores ou descritores

Termos para pesquisar:

```text
score fusion image retrieval
late fusion visual retrieval
multimodal fusion image retrieval
face and context fusion person retrieval
weighted score fusion retrieval
```

Perguntas que a bibliografia deve responder:

- O que e fusao tardia, ou late fusion?
- Como combinar scores de metodos diferentes?
- Em quais casos a fusao melhora?
- Por que a fusao pode nao melhorar sempre?

Onde entra no artigo:

```text
Metodologia
Experimentos
Discussao dos resultados
```

Observacao:

Essa e uma parte que ainda precisa de pesquisa mais forte. Precisamos de referencias para justificar a fusao simples por pesos e para discutir por que ela pode falhar.

### 3.5 Avaliacao em recuperacao de imagens

Termos para pesquisar:

```text
precision at k recall at k mean average precision image retrieval
image retrieval evaluation metrics
information retrieval precision recall mAP
```

Perguntas que a bibliografia deve responder:

- O que e Precision@K?
- O que e Recall@K?
- O que e Average Precision?
- O que e mean Average Precision, ou mAP?
- Por que essas metricas sao adequadas para ranking?

Onde entra no artigo:

```text
Metodologia experimental
Metricas
Resultados
```

### 3.6 Bases de dados

Termos para pesquisar:

```text
LFW dataset face recognition
INRIA Holidays dataset image retrieval
Gallagher collection person recognition dataset
People in photo albums dataset PIPA
```

Perguntas que a bibliografia deve responder:

- Qual o papel da LFW em reconhecimento facial?
- Qual o papel da INRIA Holidays em CBIR?
- Por que datasets de albuns/fotos pessoais sao importantes?
- Quais limitacoes existem nessas bases?

Onde entra no artigo:

```text
Bases de dados
Configuracao experimental
Limitacoes
```

Referencias iniciais:

- Pagina oficial INRIA Holidays.
- Huang et al. Labeled Faces in the Wild.
- Trabalhos associados a Gallagher ou colecoes pessoais, se encontrados.

### 3.7 Etica, privacidade e vies em reconhecimento facial

Termos para pesquisar:

```text
face recognition privacy concerns
face recognition bias survey
ethical issues facial recognition
biometric data privacy face recognition
```

Perguntas que a bibliografia deve responder:

- Quais riscos existem ao usar reconhecimento facial?
- Por que consentimento e privacidade sao relevantes?
- O que pode ser dito sobre vies em sistemas faciais?
- Como limitar o escopo do projeto para uso academico e controlado?

Onde entra no artigo:

```text
Limitacoes
Aspectos eticos
Trabalhos futuros
Conclusao
```

## 4. Prompt recomendado para Pesquisa Profunda

Use este prompt na Pesquisa Profunda do ChatGPT Plus:

```text
Estou desenvolvendo um projeto academico sobre recuperacao de imagens fotograficas por conteudo combinando reconhecimento facial e CBIR global. O sistema usa embeddings faciais com InsightFace/ArcFace, descritores globais com ResNet50 pre-treinada, similaridade cosseno, avaliacao com Precision@K, Recall@K e mAP, e bases como LFW, Gallagher e INRIA Holidays.

Pesquise artigos cientificos relevantes para fundamentar:
1. Content-Based Image Retrieval classico e com deep learning;
2. uso de CNNs pre-treinadas como extratores de caracteristicas globais;
3. reconhecimento facial baseado em embeddings;
4. fusao de scores ou combinacao de descritores em sistemas de recuperacao;
5. avaliacao de sistemas de recuperacao de imagens;
6. bases de dados LFW, INRIA Holidays, Gallagher ou datasets semelhantes de albuns pessoais;
7. limitacoes, vies e privacidade em reconhecimento facial.

Priorize artigos revisados por pares, surveys, CVPR, ICCV, ECCV, ACM, IEEE, Springer, Elsevier, arXiv relevante e paginas oficiais de datasets.

Para cada referencia, retorne:
- titulo;
- autores;
- ano;
- venue ou fonte;
- link/DOI/arXiv;
- BibTeX;
- resumo de 3 linhas;
- por que ela e util para meu projeto;
- em qual secao do artigo/TCC ela deve entrar.

Separe as referencias por tema e indique quais sao obrigatorias, recomendadas e opcionais.
```

## 5. Prompt para buscar trabalhos relacionados mais proximos

Use tambem este prompt:

```text
Encontre trabalhos relacionados a sistemas de recuperacao de fotografias pessoais ou colecoes de eventos que combinem reconhecimento de pessoas, reconhecimento facial, contexto visual e recuperacao por conteudo.

Meu projeto busca fotos em acervos fotograficos usando:
- busca por rosto;
- busca visual global;
- retorno de fotos inteiras;
- fusao entre score facial e score global;
- avaliacao com Precision@K, Recall@K e mAP.

Procure trabalhos proximos em person retrieval, photo album search, face retrieval in photo collections, people in photo albums, event photo retrieval e multimodal image retrieval.

Para cada trabalho, diga:
- qual problema ele resolve;
- qual metodo usa;
- quais dados usa;
- quais metricas usa;
- como ele se aproxima ou se diferencia do meu projeto.
```

## 6. Referencias-base iniciais

Esta lista deve ser conferida e ampliada durante a pesquisa.

### CBIR e image retrieval

1. Latif et al. A Decade Survey of Content Based Image Retrieval using Deep Learning.
2. Liu et al. Deep Learning for Instance Retrieval: A Survey.
3. Jégou, Douze e Schmid. Trabalhos associados ao INRIA Holidays e image retrieval.

### Redes convolucionais e descritores globais

1. He, Zhang, Ren e Sun. Deep Residual Learning for Image Recognition.

### Reconhecimento facial

1. Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
2. Huang et al. Labeled Faces in the Wild.

### Bases de dados

1. INRIA Holidays dataset.
2. Labeled Faces in the Wild.
3. Gallagher collection ou trabalhos relacionados a fotos pessoais.
4. PIPA, se for util para discussao teorica.

### Etica e privacidade

Ainda precisa de pesquisa especifica.

## 7. Como usar o resultado da pesquisa

Quando a Pesquisa Profunda retornar a lista, organizar assim:

```text
docs/bibliografia_anotada.md
```

Formato sugerido:

```text
## Tema

### Referencia

- Citacao:
- BibTeX:
- Ideia principal:
- Como entra no projeto:
- Secao sugerida:
- Forca da referencia: obrigatoria/recomendada/opcional
```

Depois disso, usar a bibliografia para escrever:

```text
docs/plano_artigo.md
docs/resultados_experimentais.md
docs/relatorio_meta2.md
```

## 8. Lacunas atuais

As lacunas bibliograficas mais importantes agora sao:

1. Fusao de scores em recuperacao de imagens.
2. Recuperacao de pessoas em albuns ou colecoes fotograficas.
3. Discussao etica de reconhecimento facial.
4. Justificativa teorica para metricas de ranking.
5. Trabalhos recentes de CBIR com deep learning para comparar com ResNet50.

## 9. Proximo passo

Executar a Pesquisa Profunda com os prompts acima, salvar a resposta em um arquivo ou copiar para a conversa, e depois transformar a resposta em uma bibliografia anotada revisada.

