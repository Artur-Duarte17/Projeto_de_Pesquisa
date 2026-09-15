# Sintese da revisao bibliografica

Data: 25/06/2026

## 1. Como este documento foi produzido

Este documento resume o conteudo util extraido dos arquivos em:

```text
docs/fontes_bibliograficas_brutas/
```

Esses arquivos foram tratados como material bruto de apoio. Eles nao devem ser citados no artigo. O que deve ser citado sao os artigos, livros, relatorios tecnicos e paginas oficiais que eles apontam.

## 2. Conclusao central

Os levantamentos bibliograficos reforcam que o projeto esta em uma intersecao real entre tres areas:

```text
1. CBIR classico e CBIR com deep learning.
2. Reconhecimento facial baseado em embeddings.
3. Fusao de evidencias para recuperacao ranqueada de fotos.
```

A leitura mais importante e:

```text
A literatura geralmente trata essas partes separadamente.
O projeto fica interessante porque junta busca por rosto, busca global da imagem inteira,
retorno de fotos completas e avaliacao com Precision@K, Recall@K e mAP.
```

Essa e a principal oportunidade de artigo.

## 3. Tese recomendada para o artigo

Uma formulacao forte e defensavel:

> A literatura proxima ou privilegia identidade/rosto, ou privilegia contexto visual/ranking; este trabalho avalia as duas dimensoes em um unico sistema de recuperacao de fotos completas, comparando busca facial, CBIR global e fusao simples de scores.

Versao mais curta:

> Um sistema de recuperacao fotografica que compara embeddings faciais, descritores globais ResNet50 e fusao tardia de scores.

## 4. O que ficou mais forte depois da revisao

### 4.1 O uso de ResNet50 esta bem fundamentado

Os levantamentos apontam varias referencias que sustentam o uso de CNN pre-treinada como extrator de caracteristicas globais:

- CNN Features Off-the-Shelf.
- Neural Codes for Image Retrieval.
- Deep Residual Learning for Image Recognition.
- Deep Image Retrieval.
- Fine-Tuning CNN Image Retrieval with No Human Annotation.

Conclusao para o texto:

```text
Usar ResNet50 pre-treinada como baseline global nao e improviso.
E uma escolha academica defensavel para uma v1, especialmente quando nao ha treino proprio.
```

### 4.2 O uso de ArcFace/InsightFace esta bem fundamentado

Os levantamentos reforcam que FaceNet, CosFace e ArcFace formam a linha principal de reconhecimento facial moderno por embeddings.

Conclusao para o texto:

```text
O modulo facial deve ser explicado como reconhecimento por embedding.
Nao e classificacao fixa de pessoas; e comparacao vetorial por similaridade.
```

Isso e importante porque nosso sistema consegue receber uma pessoa de consulta sem precisar treinar uma classe nova para ela.

### 4.3 A fusao tardia e mais defensavel que concatenar vetores

Os levantamentos apontam que face e global vivem em espacos diferentes:

```text
ArcFace: vetor facial de identidade.
ResNet50: vetor global de cena/imagem inteira.
```

Por isso, faz mais sentido combinar scores normalizados do que concatenar embeddings crus.

Conclusao para o texto:

```text
A fusao implementada no projeto e uma late fusion, ou fusao tardia, por soma ponderada de scores.
```

Isso combina com o que ja implementamos:

```text
score_final = alpha * score_face + beta * score_global
```

### 4.4 Nosso resultado negativo na fusao e publicavel

As pesquisas deixam claro que fusao pode ajudar, mas nao e garantia.

No nosso experimento Gallagher:

```text
Face isolado foi melhor que fusao simples.
```

Isso deve ser escrito como:

```text
A fusao simples adicionou ruido quando a consulta era centrada em identidade facial.
```

Nao devemos escrever como:

```text
A fusao falhou.
```

Interpretacao correta:

```text
O peso global pode prejudicar o ranking quando o objetivo e encontrar uma pessoa especifica,
principalmente quando o descritor global captura fundo, grama, objetos e outras pessoas.
```

### 4.5 A diferenca entre datasets precisa ficar explicita

As pesquisas reforcam um ponto metodologico critico:

```text
LFW nao mede a mesma coisa que INRIA Holidays.
INRIA Holidays nao mede a mesma coisa que Gallagher.
Gallagher se aproxima mais do uso real com fotos de grupo.
```

Como escrever:

```text
LFW foi usado como validacao inicial de embeddings faciais.
INRIA Holidays foi usado para avaliar CBIR global por imagem inteira.
Gallagher foi usado para avaliar recuperacao de pessoas em fotos completas e fusao.
```

Isso evita uma critica forte: misturar metricas de tarefas diferentes.

## 5. Contribuicao cientifica sugerida

Contribuicoes que podemos defender:

1. Implementacao modular de um sistema de recuperacao fotografica com trilha facial e trilha global.
2. Avaliacao separada de busca facial e CBIR global em bases adequadas a cada tarefa.
3. Comparacao experimental entre face isolado, global isolado e fusao tardia.
4. Demonstracao de que a fusao simples nao melhora automaticamente a busca por identidade.
5. Interface minima que demonstra busca por pessoa, busca por imagem semelhante e fusao.

## 6. Trabalhos diretamente proximos

Os mais importantes para a secao de trabalhos relacionados sao:

| Trabalho | Utilidade para o projeto |
| --- | --- |
| Picture Management Using Person Retrieval for Consumer Image Collections | Muito proximo: busca pessoa e retorna foto inteira |
| Face Annotation for Personal Photos Using Context-Assisted Face Recognition | Mostra contexto ajudando reconhecimento em fotos pessoais |
| Person Recognition in Personal Photo Collections | PIPA e combinacao de face, corpo e cena |
| Beyond Frontal Faces | PIPER, multiplas pistas para reconhecer pessoas |
| A Multi-Level Contextual Model for Person Recognition in Photo Albums | Fusão explicita de pistas e contexto multinivel |
| Sequential Person Recognition in Photo Albums | Cena e relacoes entre pessoas em albuns |
| Combination of Content Analysis and Context Features for Digital Photograph Retrieval | Conteudo + contexto em ranking, com mAP |

Esses trabalhos nao fazem exatamente o mesmo que o nosso. Isso e bom: eles ajudam a mostrar a lacuna.

## 7. Lacuna que o artigo deve explorar

Frase recomendada:

```text
Embora trabalhos anteriores explorem reconhecimento de pessoas em albuns ou recuperacao visual por conteudo,
poucos avaliam, em uma mesma arquitetura, busca por rosto, busca global por imagem inteira e fusao tardia,
retornando fotos completas e usando metricas de ranking como Precision@K, Recall@K e mAP.
```

## 8. O que evitar

Evitar promessas que o projeto atual ainda nao cumpre:

```text
Nao dizer que escalamos para milhoes de imagens.
Nao dizer que usamos FAISS na v1.
Nao dizer que resolvemos privacidade/criptografia.
Nao dizer que avaliamos vies demografico.
Nao dizer que a fusao melhorou, porque no Gallagher ela nao melhorou.
```

Escrever corretamente:

```text
FAISS, criptografia, selecao manual de rosto, avaliacao de vies e acervos maiores sao trabalhos futuros.
```

## 9. Estrutura sugerida para o artigo

```text
1. Introducao
   - explosao de acervos fotograficos
   - problema de busca manual
   - busca por pessoa e por similaridade visual

2. Trabalhos relacionados
   - CBIR classico e deep learning
   - reconhecimento facial por embeddings
   - person recognition em albuns
   - fusao de scores e ranking

3. Metodologia
   - indexacao facial multi-rosto
   - indexacao global ResNet50
   - busca por similaridade cosseno
   - fusao tardia ponderada
   - interface Streamlit

4. Experimentos
   - LFW como validacao facial inicial
   - Gallagher para pessoa/fusao
   - INRIA Holidays para CBIR global
   - metricas Precision@K, Recall@K, mAP e tempo

5. Resultados
   - tabela facial
   - tabela global
   - tabela fusao
   - exemplos visuais

6. Discussao
   - face melhor para identidade
   - global melhor para cena/similaridade visual
   - fusao simples pode prejudicar
   - diferenca entre datasets

7. Limitacoes e trabalhos futuros
   - FAISS
   - CLIP
   - selecao manual de rosto
   - busca por casal
   - privacidade
   - vies

8. Conclusao
```

## 10. Documentos derivados

Criados nesta etapa:

1. `docs/bibliografia_anotada.md`

Proximos documentos recomendados:

1. `docs/resultados_experimentais.md`
2. `docs/plano_artigo.md`
3. `docs/relatorio_meta2.md`
