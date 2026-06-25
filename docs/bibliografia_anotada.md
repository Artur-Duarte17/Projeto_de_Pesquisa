# Bibliografia anotada inicial

Data: 25/06/2026

## 1. Observacao

Este documento organiza artigos e fontes uteis identificados nas pesquisas profundas e na proposta original.

As pesquisas profundas nao devem ser citadas. As referencias abaixo apontam para artigos, livros, relatorios ou paginas oficiais.

Classificacao:

```text
Obrigatoria: deve entrar no artigo/TCC.
Recomendada: entra se houver espaco e melhora a discussao.
Complementar: util para trabalhos futuros ou aprofundamento.
Validar: veio das pesquisas profundas, mas precisa de verificacao bibliografica antes da citacao final.
```

## 2. CBIR classico e deep learning

### Smeulders et al. - Content-Based Image Retrieval at the End of the Early Years

- Classificacao: obrigatoria.
- Tema: CBIR classico, semantic gap, descritores visuais.
- Fonte verificada: IEEE/ACM.
- Link: https://ieeexplore.ieee.org/document/895972/
- DOI: `10.1109/34.895972`
- Uso no projeto: fundamentar o que e CBIR e por que descritores visuais de baixo nivel possuem limitacoes semanticas.
- Secao sugerida: fundamentacao teorica.

### Datta et al. - Image Retrieval: Ideas, Influences, and Trends of the New Age

- Classificacao: obrigatoria.
- Tema: evolucao de image retrieval, busca por similaridade, desafios praticos.
- Fonte verificada: ACM Computing Surveys.
- Link: https://dl.acm.org/doi/10.1145/1348246.1348248
- DOI: `10.1145/1348246.1348248`
- Uso no projeto: conectar CBIR classico com sistemas modernos e a ideia de ranking visual.
- Secao sugerida: trabalhos relacionados.

### Dubey - A Decade Survey of Content Based Image Retrieval using Deep Learning

- Classificacao: obrigatoria.
- Tema: CBIR com deep learning.
- Fonte verificada: arXiv/IEEE.
- Link: https://arxiv.org/abs/2012.00641
- DOI: `10.1109/TCSVT.2021.3080920`
- Uso no projeto: justificar a transicao de descritores manuais para features profundas.
- Secao sugerida: trabalhos relacionados.

### Wan et al. - Deep Learning for Content-Based Image Retrieval: A Comprehensive Study

- Classificacao: recomendada.
- Tema: estudo sistematico de CNNs em CBIR.
- Fonte indicada nas pesquisas profundas: ACM Multimedia 2014.
- DOI indicado: `10.1145/2647868.2654948`
- Uso no projeto: reforcar que deep features reduzem o semantic gap.
- Secao sugerida: trabalhos relacionados.

### Philbin et al. - Object Retrieval with Large Vocabularies and Fast Spatial Matching

- Classificacao: complementar.
- Tema: recuperacao classica com SIFT, vocabulario visual e verificacao espacial.
- DOI indicado: `10.1109/CVPR.2007.383172`
- Uso no projeto: contextualizar o CBIR classico antes de deep learning.
- Secao sugerida: fundamentacao historica, se houver espaco.

### Jegou, Douze e Schmid - Hamming Embedding and Weak Geometric Consistency for Large Scale Image Search

- Classificacao: recomendada.
- Tema: large-scale image search, Hamming Embedding, INRIA Holidays.
- Fonte verificada: pagina oficial INRIA Holidays e ECCV.
- Link dataset: https://thoth.inrialpes.fr/~jegou/data.php.html
- Uso no projeto: fundamentar o uso da INRIA Holidays e a tradicao de avaliacao de CBIR.
- Secao sugerida: bases de dados e trabalhos relacionados.

## 3. CNNs pre-treinadas e descritores globais

### Razavian et al. - CNN Features Off-the-Shelf: An Astounding Baseline for Recognition

- Classificacao: obrigatoria.
- Tema: CNN pre-treinada como extrator generico.
- Fonte verificada: arXiv/CVF.
- Link: https://arxiv.org/abs/1403.6382
- Uso no projeto: justificar uso de rede pre-treinada sem treino proprio.
- Secao sugerida: metodologia do modulo global.

### Babenko et al. - Neural Codes for Image Retrieval

- Classificacao: obrigatoria.
- Tema: ativacoes de CNN como descritores globais para recuperacao.
- Fonte verificada: arXiv/ECCV.
- Link: https://arxiv.org/abs/1404.1777
- Uso no projeto: fundamentar embeddings globais como vetores comparaveis por similaridade.
- Secao sugerida: metodologia do CBIR global.

### He et al. - Deep Residual Learning for Image Recognition

- Classificacao: obrigatoria.
- Tema: arquitetura ResNet.
- Fonte verificada: CVF/IEEE.
- Link: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
- DOI: `10.1109/CVPR.2016.90`
- Uso no projeto: citar a arquitetura usada para extrair embeddings globais de 2048 dimensoes.
- Secao sugerida: metodologia.

### Gordo et al. - Deep Image Retrieval: Learning Global Representations for Image Search

- Classificacao: recomendada.
- Tema: representacoes globais profundas para image retrieval.
- Fonte verificada: arXiv/ECCV.
- Link: https://arxiv.org/abs/1604.01325
- Uso no projeto: mostrar que descritores globais profundos sao uma linha consolidada em image retrieval.
- Secao sugerida: trabalhos relacionados.

### Radenovic, Tolias e Chum - Fine-Tuning CNN Image Retrieval with No Human Annotation

- Classificacao: recomendada.
- Tema: fine-tuning e GeM pooling para image retrieval.
- Fonte verificada: arXiv/TPAMI.
- Link: https://arxiv.org/abs/1711.02512
- DOI: `10.1109/TPAMI.2018.2846566`
- Uso no projeto: trabalho futuro, pois nossa v1 usa ResNet50 sem fine-tuning.
- Secao sugerida: trabalhos futuros.

## 4. Reconhecimento facial por embeddings

### Schroff, Kalenichenko e Philbin - FaceNet

- Classificacao: obrigatoria.
- Tema: embeddings faciais, triplet loss, distancia em espaco vetorial.
- Fonte verificada: arXiv/CVF.
- Link: https://arxiv.org/abs/1503.03832
- Uso no projeto: explicar a ideia geral de mapear faces para um espaco vetorial de similaridade.
- Secao sugerida: fundamentacao facial.

### Wang et al. - CosFace

- Classificacao: recomendada.
- Tema: margem cosseno para reconhecimento facial.
- Fonte verificada: arXiv/IEEE.
- Link: https://arxiv.org/abs/1801.09414
- DOI: `10.1109/CVPR.2018.00552`
- Uso no projeto: ponte entre FaceNet e ArcFace, reforcando normalizacao e similaridade cosseno.
- Secao sugerida: reconhecimento facial moderno.

### Deng et al. - ArcFace

- Classificacao: obrigatoria.
- Tema: Additive Angular Margin Loss, embeddings faciais discriminativos.
- Fonte verificada: arXiv/IEEE/CVF/InsightFace.
- Link: https://arxiv.org/abs/1801.07698
- DOI: `10.1109/CVPR.2019.00482`
- Uso no projeto: fundamentar o uso de InsightFace/ArcFace para gerar embeddings faciais.
- Secao sugerida: metodologia facial.

### Huang et al. - Labeled Faces in the Wild

- Classificacao: obrigatoria.
- Tema: benchmark de reconhecimento facial em ambiente nao controlado.
- Fonte verificada: HAL/UMass.
- Link: https://inria.hal.science/inria-00321923
- Uso no projeto: explicar LFW como validacao inicial do modulo facial.
- Secao sugerida: bases de dados.

## 5. Albuns pessoais, pessoas em fotos e contexto visual

### Costache et al. - Picture Management Using Person Retrieval for Consumer Image Collections

- Classificacao: obrigatoria para trabalhos relacionados.
- Tema: recuperacao de pessoas em colecoes de imagens de consumidores.
- Uso no projeto: um dos trabalhos mais proximos, pois combina pista facial e nao facial e retorna imagens da colecao.
- Diferenca para o projeto: usa descritores/classificadores antigos e nao mAP/Precision@K como no nosso protocolo.
- Status: validar DOI/fonte antes da citacao final.

### Choi et al. - Face Annotation for Personal Photos Using Context-Assisted Face Recognition

- Classificacao: recomendada.
- Tema: contexto ajudando reconhecimento facial em fotos pessoais.
- Uso no projeto: sustenta a ideia de que situacao, fundo e agrupamento podem ajudar identidade.
- Diferenca para o projeto: foco em anotacao de faces, nao ranking de fotos completas.
- Status: validar DOI/fonte antes da citacao final.

### Cooray e O'Connor - Enhancing Person Annotation for Personal Photo Management Applications

- Classificacao: complementar.
- Tema: anotacao semiautomatica de pessoas usando face, body patch e evento.
- Uso no projeto: justificar trabalho futuro com contexto de evento.
- Status: validar DOI/fonte antes da citacao final.

### Shimizu et al. - Learning People Co-occurrence Relations by Using Relevance Feedback for Retrieving Group Photos

- Classificacao: recomendada.
- Tema: recuperacao de fotos de grupo usando coocorrencia e feedback.
- Uso no projeto: fundamentar futura busca por casal/grupo e reranking contextual.
- Status: validar DOI/fonte antes da citacao final.

### Zhang et al. - Beyond Frontal Faces: Improving Person Recognition Using Multiple Cues

- Classificacao: obrigatoria para discutir PIPA e multiplas pistas.
- Tema: PIPA/PIPER, face, corpo, poselets e contexto.
- Fonte verificada: arXiv/CVF.
- Link: https://arxiv.org/abs/1501.05703
- Uso no projeto: mostrar que rosto isolado nao basta em fotos reais e que pistas adicionais ajudam.
- Secao sugerida: trabalhos relacionados.

### Oh et al. - Person Recognition in Personal Photo Collections

- Classificacao: obrigatoria para trabalhos relacionados.
- Tema: reconhecimento de pessoas em PIPA com face, corpo e cena.
- Fonte verificada: arXiv/CVF/ACM.
- Link: https://arxiv.org/abs/1509.03502
- Uso no projeto: referencia forte para albuns pessoais e uso de cena/global image.
- Diferenca para o projeto: avalia classification accuracy, nao recuperacao ranqueada de fotos.

### Li et al. - A Multi-Level Contextual Model for Person Recognition in Photo Albums

- Classificacao: obrigatoria para fusao/contexto.
- Tema: fusao de face/corpo/contexto, CRF, album/foto/grupo.
- Fonte verificada: CVF/Adobe Research.
- Link: https://openaccess.thecvf.com/content_cvpr_2016/papers/Li_A_Multi-Level_Contextual_CVPR_2016_paper.pdf
- Uso no projeto: justificar fusao de pistas e contexto visual em albuns.
- Diferenca para o projeto: classificacao fechada de identidade, nao busca por similaridade.

### Li et al. - Sequential Person Recognition in Photo Albums With a Recurrent Network

- Classificacao: recomendada.
- Tema: relacoes entre pessoas e contexto de cena em albuns.
- Uso no projeto: reforcar que cena e relacoes sociais podem ajudar identidade.
- Status: validar link/fonte final.

### Naaman et al. - Leveraging Context to Resolve Identity in Photo Albums

- Classificacao: complementar.
- Tema: contexto temporal, local e social em albuns.
- Uso no projeto: trabalhos futuros com metadados/eventos.
- Status: validar DOI/fonte antes da citacao final.

### Zhao et al. - Automatic Person Annotation of Family Photo Album

- Classificacao: complementar.
- Tema: anotacao em albuns familiares com face, corpo e contexto.
- Uso no projeto: referencia historica para contexto em albuns.
- Status: validar fonte final.

## 6. Fusão de scores, late fusion e avaliacao

### Snoek, Worring e Smeulders - Early versus Late Fusion in Semantic Video Analysis

- Classificacao: recomendada.
- Tema: early fusion vs late fusion.
- Fonte verificada: ACM/PDF institucional.
- Link: https://dl.acm.org/doi/10.1145/1101149.1101236
- Uso no projeto: justificar a escolha por fusao tardia de scores em vez de concatenacao de vetores.
- Secao sugerida: metodologia da fusao.

### Atrey et al. - Multimodal Fusion for Multimedia Analysis: A Survey

- Classificacao: obrigatoria para fusao.
- Tema: taxonomia de fusao multimodal.
- Fonte verificada: Springer/ACM index.
- Link: https://dl.acm.org/doi/abs/10.1007/s00530-010-0182-0
- Uso no projeto: classificar a nossa fusao como score-level/late fusion.
- Secao sugerida: fundamentacao da fusao.

### Jain, Nandakumar e Ross - Score Normalization in Multimodal Biometric Systems

- Classificacao: obrigatoria para fusao de scores.
- Tema: normalizacao de scores em sistemas biometricos multimodais.
- Fonte verificada: Pattern Recognition/ACM index.
- Link: https://dl.acm.org/doi/10.1016/j.patcog.2005.01.012
- Uso no projeto: justificar normalizar scores antes de combinar modalidades diferentes.
- Secao sugerida: metodologia da fusao.

### Manning, Raghavan e Schutze - Introduction to Information Retrieval

- Classificacao: obrigatoria para metricas.
- Tema: precision, recall, average precision, ranking.
- Fonte verificada: Cambridge/MIT Press.
- Link: https://assets.cambridge.org/97805218/65715/frontmatter/9780521865715_frontmatter.pdf
- Uso no projeto: fundamentar Precision@K, Recall@K, AP e mAP.
- Secao sugerida: metricas de avaliacao.

### Muller et al. - Performance Evaluation in Content-Based Image Retrieval: Overview and Proposals

- Classificacao: recomendada.
- Tema: avaliacao em CBIR.
- DOI indicado: `10.1016/S0167-8655(00)00118-5`
- Uso no projeto: reforcar metricas e protocolo de avaliacao em CBIR.
- Status: validar link/fonte final.

## 7. Bases de dados

### INRIA Holidays

- Classificacao: obrigatoria.
- Tema: benchmark de CBIR por grupos de imagens similares.
- Fonte verificada: pagina oficial THOTH/INRIA.
- Link: https://thoth.inrialpes.fr/~jegou/data.php.html
- Uso no projeto: justificar a avaliacao global com 1.491 imagens e 500 grupos/consultas.
- Secao sugerida: bases de dados.

### PIPA - People in Photo Albums

- Classificacao: recomendada.
- Tema: pessoas em albuns pessoais.
- Fonte verificada: Exposing.ai e trabalhos Zhang/Oh/Li.
- Link: https://exposing.ai/pipa/
- Uso no projeto: discutir porque albuns pessoais sao mais dificeis que LFW.
- Secao sugerida: trabalhos relacionados e trabalhos futuros.

### Gallagher

- Classificacao: obrigatoria para nosso experimento.
- Tema: fotos pessoais/eventos com varias pessoas.
- Uso no projeto: base pratica para avaliar multi-rosto, busca por pessoa e fusao.
- Status: precisamos validar a citacao original da base/artigo associado antes do texto final.

## 8. Etica, privacidade e vies

### LGPD - Lei Geral de Protecao de Dados

- Classificacao: obrigatoria se houver secao de privacidade.
- Tema: dados biometricos como dados sensiveis no Brasil.
- Uso no projeto: justificar escopo local, cuidado com dados e limitacoes.
- Secao sugerida: aspectos eticos e limitacoes.

### Buolamwini e Gebru - Gender Shades

- Classificacao: recomendada.
- Tema: disparidades interseccionais em sistemas comerciais de analise facial.
- Fonte verificada: Proceedings of Machine Learning Research.
- Link: https://proceedings.mlr.press/v81/buolamwini18a.html
- Uso no projeto: explicar por que reconhecimento facial exige cuidado com vies.
- Secao sugerida: limitacoes.

### NIST - Face Recognition Vendor Test Part 3: Demographic Effects

- Classificacao: recomendada.
- Tema: efeitos demograficos em algoritmos de reconhecimento facial.
- Fonte verificada: NIST.
- Link: https://www.nist.gov/publications/face-recognition-vendor-test-part-3-demographic-effects
- DOI: `10.6028/NIST.IR.8280`
- Uso no projeto: reforcar que avaliacao de vies e importante, mas ficou fora da v1.
- Secao sugerida: limitacoes e trabalhos futuros.

### Drozdowski et al. - Demographic Bias in Biometrics

- Classificacao: recomendada.
- Tema: survey de vies demografico em biometria.
- Fonte verificada: arXiv/IEEE.
- Link: https://arxiv.org/abs/2003.02488
- DOI: `10.1109/TTS.2020.2992344`
- Uso no projeto: embasar discussao de IA responsavel.
- Secao sugerida: limitacoes.

### Ensuring Privacy in Face Recognition

- Classificacao: complementar.
- Tema: privacidade em reconhecimento facial, geracao de dados, inferencia e armazenamento.
- Fonte verificada: Springer Nature.
- Link: https://link.springer.com/article/10.1007/s42452-025-06987-2
- Uso no projeto: trabalho futuro para protecao de embeddings/dados.
- Secao sugerida: trabalhos futuros.

## 9. Referencias a usar com cuidado

As pesquisas profundas citaram alguns trabalhos interessantes, mas eles devem ser usados so depois de validacao:

```text
EVENT-Retriever.
SA-Person.
CLIP-BLIP-FAISS para fotos pessoais.
WildFusion.
Mitigating Bias in Face Recognition Using Skewness-Aware Reinforcement Learning.
Sistemas comerciais/academicos de fotografia de eventos com selfie.
```

Motivo:

```text
Sao uteis para contexto moderno ou produto, mas podem fugir do escopo da v1.
Alguns podem ser recentes, preprint, tecnicos ou pouco conectados diretamente ao nosso experimento.
```
