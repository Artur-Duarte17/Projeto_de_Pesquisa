# Referencias e ideias aproveitaveis da proposta original

Fonte analisada: `C:\OneDriePessoal\OneDrive\Documentos\Projeto IA\Projeto IA.docx`

Data da extracao: 25/06/2026

## 1. Como este documento deve ser usado

Este arquivo registra o que ja existia na proposta original do projeto para nao perdermos a fundamentacao inicial.

Ele nao substitui uma revisao bibliografica completa. Algumas referencias da proposta original aparecem com links ausentes ou incompletos, entao precisam ser conferidas antes de entrar no artigo final.

Uso recomendado:

```text
1. manter como memoria tecnica da proposta;
2. reaproveitar argumentos de objetivo, justificativa e metodologia;
3. validar cada referencia original antes de citar;
4. complementar com a Pesquisa Profunda e Google Scholar.
```

## 2. Ideia central da proposta original

Titulo original:

```text
Recuperacao de Imagens Fotograficas por Conteudo e Reconhecimento Facial em Colecoes de Grande Escala
```

Objetivo geral extraido da proposta:

```text
Desenvolver um prototipo de sistema inteligente de filtragem e busca fotografica que integre tecnicas de deteccao/identificacao facial e recuperacao de imagens por similaridade, capaz de identificar e reunir de maneira eficiente e precisa todas as fotografias que contenham uma pessoa especifica ou sejam visualmente semelhantes a uma imagem de consulta em um grande acervo.
```

Esse objetivo continua alinhado com o estado atual do projeto.

## 3. Pontos da proposta que ja foram atendidos

| Item da proposta | Estado atual |
| --- | --- |
| Busca por rosto | Implementada com InsightFace/embeddings faciais |
| Busca por similaridade visual | Implementada com ResNet50 global |
| Retorno de fotos inteiras | Implementado |
| Embeddings faciais | Implementados com vetores de 512 dimensoes |
| Descritores globais | Implementados com vetores ResNet50 de 2048 dimensoes |
| Metricas Precision@K, Recall@K e mAP | Implementadas |
| Interface Streamlit | Implementada em versao minima |
| Fusão face + global | Implementada e avaliada |
| Avaliacao comparativa | Implementada para face, global e fusao |
| Repositorio organizado | Implementado com Git/GitHub e arquivos grandes fora do Git |

## 4. Pontos planejados que ficaram fora da v1

Esses pontos estavam na proposta, mas foram conscientemente deixados para versao futura ou trabalho futuro:

| Item | Motivo |
| --- | --- |
| FAISS | A v1 usa NumPy/cosseno para manter simplicidade e reprodutibilidade |
| Elasticsearch | Mais relevante para produto ou escala maior |
| Indexacao de 100 mil imagens | Fora do escopo prático atual |
| CelebA | O projeto atual priorizou LFW, Gallagher e INRIA Holidays |
| Fine-tuning de modelo facial | Nao foi necessario para o MVP |
| Cifragem de embeddings | Importante para produto, mas fora da v1 academica |
| Selecao manual de rosto na interface | Identificada como melhoria futura |
| Avaliacao de vies demografico | Importante, mas requer protocolo e dados adequados |

## 5. Referencias listadas na proposta original

### 5.1 CBIR com deep learning

Referencia da proposta:

```text
AHMED, A. S.; IBRAHEEM, I. N. Recent advances in content-based image retrieval using deep learning techniques: A survey. In: AIP Conference Proceedings, v. 3219, n. 1, 2024.
```

Uso sugerido:

```text
Revisao bibliografica sobre CBIR com deep learning.
```

Status:

```text
Conferir link/DOI antes de citar.
```

### 5.2 LGPD

Referencia da proposta:

```text
BRASIL. Lei no 13.709, de 14 de agosto de 2018. Lei Geral de Protecao de Dados Pessoais. Diario Oficial da Uniao, Brasilia, DF, 15 ago. 2018.
```

Uso sugerido:

```text
Secao de privacidade, dados biometricos e limitacoes eticas.
```

Status:

```text
Referencia obrigatoria se o texto discutir reconhecimento facial e dados biometricos no Brasil.
```

### 5.3 ArcFace

Referencia da proposta:

```text
DENG, J. et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), p. 4690-4699, 2019.
```

Uso sugerido:

```text
Fundamentar embeddings faciais modernos e reconhecimento facial baseado em margem angular.
```

Status:

```text
Referencia obrigatoria.
```

### 5.4 Volume de fotos digitais

Referencia da proposta:

```text
LACAILLE, M. Rise Above Research estimates that 1.6 trillion photos will be taken in 2023. Mediaclip Blog, 25 jan. 2023.
```

Uso sugerido:

```text
Motivacao do problema e crescimento de acervos fotograficos.
```

Status:

```text
Pode ser usada como fonte contextual, mas nao deve ser uma das referencias cientificas principais.
```

### 5.5 Elasticsearch para busca por similaridade

Referencia da proposta:

```text
ONDAS, R.; SUHM, B. Overview of image similarity search in Elasticsearch. Elastic Blog, 28 fev. 2023.
```

Uso sugerido:

```text
Trabalho futuro/produto, busca vetorial e escalabilidade.
```

Status:

```text
Referencia tecnica, nao artigo cientifico principal.
```

### 5.6 Survey de CBIR

Referencia da proposta:

```text
SRIVASTAVA, D. et al. Content-based image retrieval: a survey on local and global features selection, extraction, representation, and evaluation parameters. IEEE Access, v. 11, p. 95410-95431, 2023.
```

Uso sugerido:

```text
Referencia forte para fundamentacao de CBIR, descritores locais/globais e metricas.
```

Status:

```text
Referencia recomendada/forte. Conferir DOI e BibTeX.
```

### 5.7 CBIR-ANR

Referencia da proposta:

```text
VIEIRA, G. S.; FONSECA, A. U.; SOARES, F. CBIR-ANR: A content-based image retrieval with accuracy noise reduction. Software Impacts, v. 15, p. 100486, 2023. DOI: 10.1016/j.simpa.2023.100486.
```

Uso sugerido:

```text
Referencia importante para discutir reducao de ruido, re-ranqueamento e possivel melhoria futura da fusao.
```

Status:

```text
Referencia muito relevante, especialmente por estar ligada ao orientador/projeto.
```

### 5.8 Survey de reconhecimento facial

Referencia da proposta:

```text
WANG, X. et al. A Survey of Face Recognition. arXiv preprint arXiv:2212.13038, 2023.
```

Uso sugerido:

```text
Fundamentacao de reconhecimento facial moderno, embeddings, desafios e tendencias.
```

Status:

```text
Referencia recomendada. Verificar se existe versao publicada alem do arXiv.
```

## 6. Links soltos encontrados na proposta

Os links abaixo apareceram no final do documento original. Eles precisam ser limpos, verificados e transformados em citacoes corretas:

```text
https://www.mediaclip.ca/en/blog/rise-above-research-estimates-that-1-6-trillion-photos-will-be-taken-in-2023/2023/
https://matjournals.net/engineering/index.php/JoAESP/article/view/1387
https://arxiv.org/pdf/2401.08281
https://www.elastic.co/blog/overview-image-similarity-search-in-elastic
https://www.sciencedirect.com/science/article/abs/pii/S0957417423012769
https://arxiv.org/abs/2312.10089
https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
https://arxiv.org/abs/2212.13038
https://pmc.ncbi.nlm.nih.gov/articles/PMC9960175/
https://training.continuumlabs.ai/knowledge/vector-databases/faiss-facebook-ai-similarity-search
```

## 7. Trechos metodologicos aproveitaveis

### 7.1 Pipeline proposto

A proposta original descreve um fluxo ainda valido:

```text
1. deteccao e alinhamento facial;
2. extracao de embedding facial;
3. extracao de descritor visual global;
4. indexacao dos vetores;
5. consulta por imagem;
6. comparacao por similaridade;
7. fusao e re-ranqueamento;
8. exibicao dos resultados ao usuario.
```

O projeto atual implementa esse pipeline com uma simplificacao:

```text
NumPy + similaridade cosseno no lugar de FAISS/Elastic.
```

Essa decisao deve ser explicada no artigo como escolha de v1/prototipo.

### 7.2 Metricas

A proposta ja previa:

```text
Precision@K
Recall@K
mAP
tempo medio de consulta
```

Essas metricas ja foram implementadas e devem aparecer no artigo.

### 7.3 Fusão

A proposta previa soma ponderada com pesos para face e global:

```text
score_final = alfa * score_face + beta * score_global
```

Isso foi exatamente o que testamos.

Resultado atual:

```text
A fusao simples nao superou a busca facial no Gallagher.
```

Esse resultado deve ser apresentado como avaliacao experimental, nao como falha.

## 8. Como isso entra no artigo

### Introducao

Usar da proposta:

- problema do grande volume de fotos;
- dificuldade de busca manual;
- demanda de fotografos e usuarios comuns;
- necessidade de busca por pessoa e por similaridade visual.

### Fundamentacao teorica

Usar da proposta:

- definicao de CBIR;
- semantic gap;
- deep learning em CBIR;
- embeddings faciais;
- ArcFace;
- indexacao vetorial;
- metricas de recuperacao.

### Metodologia

Usar da proposta:

- pipeline modular;
- indices separados para face e global;
- ResNet50 como extrator global;
- InsightFace/ArcFace para embeddings faciais;
- fusao ponderada;
- Streamlit como interface.

### Discussao

Usar da proposta:

- expectativa de que a fusao poderia melhorar;
- nosso resultado mostrando que ela nao melhora automaticamente;
- proposta futura de re-ranqueamento e indexacao eficiente.

### Limitacoes e trabalhos futuros

Usar da proposta:

- FAISS;
- ElasticSearch;
- escalabilidade;
- selecao manual de rosto;
- privacidade/LGPD;
- avaliacao de vies;
- acervos maiores.

## 9. Lacunas da proposta original

A proposta original e boa, mas precisa ser ajustada ao que foi realmente implementado:

1. Ela prometia FAISS/Elastic, mas a v1 usa NumPy.
2. Ela citava CelebA, mas a execucao atual usa LFW, Gallagher e INRIA Holidays.
3. Ela esperava acervo de 10 mil a 100 mil imagens, mas a avaliacao atual e menor.
4. Ela sugeria criptografia de embeddings, ainda nao implementada.
5. Ela previa selecao manual de rosto, ainda nao implementada.
6. Ela sugeria que a fusao poderia melhorar; nossos resultados mostram que a fusao simples nao melhorou no cenario testado.

Essas lacunas nao impedem artigo. Elas devem virar:

```text
delimitacao da v1
limitacoes
trabalhos futuros
```

## 10. Proximo uso

Este documento deve alimentar:

```text
docs/bibliografia_anotada.md
docs/plano_artigo.md
docs/resultados_experimentais.md
docs/relatorio_meta2.md
```

