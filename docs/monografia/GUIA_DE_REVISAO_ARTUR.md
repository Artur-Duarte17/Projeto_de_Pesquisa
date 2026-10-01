# Guia de compreensão e revisão de Artur

Esta é uma versão completa para sua revisão, ainda sem validação humana ou
autorização institucional de entrega. Comece pelo resumo, pela seção 5.1 e pela
conclusão. Depois leia o método com as perguntas abaixo ao lado.

## O que foi construído e o que foi pesquisado

Foi construída uma aplicação local que prepara álbuns, extrai representações,
busca fotografias semelhantes e permite atualizar ou remover álbuns do índice.
A pesquisa usa essa implementação para uma pergunta específica: acrescentar
semelhança global da fotografia à semelhança facial melhora o ranking de fotos
da mesma pessoa? Ter uma aplicação funcional e responder à pergunta são
resultados diferentes. Não criamos ou treinamos um novo modelo de IA.

## As peças do experimento

**Detecção** localiza uma região que parece conter um rosto. **Representação
facial** transforma esse rosto em um vetor. Comparar vetores permite ordenar
semelhanças de identidade, mas o escore não é probabilidade de identidade.
SCRFD detecta; ArcFace/InsightFace representa. FaceNet aparece na fundamentação,
mas não foi implementado. ResNet50 descreve a aparência global, incluindo
pessoas, objetos e cena; não analisa somente o fundo.

Uma **consulta** indica a pessoa procurada. A **galeria** é o conjunto de
fotografias candidatas. O **gabarito** identifica quais delas contêm a pessoa,
a partir das anotações da coleção. Tiramos a fotografia-fonte dos resultados
para não contar a própria imagem consultada como um acerto trivial.

No Gallagher, há 589 fotos e 20 consultas de 20 pessoas, originadas em 19 fotos.
Cada ranking tem 588 candidatas. As 884 relações de relevância não são 884 fotos
diferentes: uma foto pode ser relevante para mais de uma consulta. A anotação
define a relevância mesmo quando o detector falha. Três recortes candidatos
foram descartados durante a preparação; isso limita a generalização para
consultas difíceis. Não se substituiu a pessoa anotada por uma vizinha.

## Como ler as métricas

P@5 pergunta quantos dos primeiros cinco resultados são relevantes, dividido
por cinco. R@5 pergunta que fração de todas as fotos relevantes foi encontrada
nessas posições. AP valoriza colocar as fotos relevantes cedo no ranking.
Com relevantes nas posições 1, 3 e 5, AP = (1 + 2/3 + 3/5)/3 ≈ 0,756.
Esse exemplo é didático, não resultado de uma consulta real.

mAP é a média das APs das consultas. Um mAP de 0,952261 não significa 95,2% de
probabilidade de identificar qualquer pessoa e não garante esse desempenho
em uma nova coleção. As consultas compartilham fotos e contextos e não são
vinte experimentos independentes.

## O que a fusão fez

A fusão soma escores facial e global com pesos fixos. Em 0,9/0,1, o escore facial
recebe peso 0,9 e o global, 0,1. Esses pesos não foram aprendidos nem otimizados
com validação independente. Colocar dois escores na faixa [0,1] não garante
que eles tenham a mesma distribuição ou significado.

| Configuração | mAP Gallagher |
|---|---:|
| Somente face | 0,952261 |
| Somente global | 0,280864 |
| Fusão 0,9/0,1 | 0,946274 |
| Fusão 0,7/0,3 | 0,901460 |
| Fusão 0,5/0,5 | 0,813297 |

A frase principal é: **nenhuma fusão avaliada superou a busca facial no mAP
agregado deste protocolo**. A fusão 0,9/0,1 melhorou duas consultas, empatou
nove e piorou nove. Não podemos concluir que contexto nunca ajuda, que o fundo
causou a queda, que toda fusão é inferior ou que há significância estatística.

## Aplicação e avaliações auxiliares

Na interface, você escolhe um rosto e navega pelos resultados. Na avaliação,
consultas, gabarito, exclusões e pesos são fixados para comparação. O uso de um
álbum familiar verifica operações práticas, não prova qualidade científica
nem autoriza exibir as fotografias. Os 93 testes foram registrados antes da
redação e não foram repetidos nesta tarefa.

LFW usa uma adaptação local de recuperação, não o protocolo oficial de pares.
Holidays verifica recuperação global com uma AP adaptada. Seus mAPs não devem
ser comparados entre si como uma competição ou como números oficiais.

## Perguntas prováveis da banca

1. **Qual é sua contribuição?** Uma comparação experimental delimitada,
   apoiada em software funcional e evidência rastreável; não um modelo novo.
2. **Por que estudar uma fusão que perdeu?** Uma hipótese plausível pode não
   se confirmar. O resultado informa em quais condições a combinação falhou
   em melhorar a média e evita uma recomendação sem evidência.
3. **Por que usar contexto?** Há trabalhos que exploram pistas além do rosto;
   a nossa soma de escores é mais simples e não replica todos esses métodos.
4. **Por que vinte consultas?** São as consultas elegíveis selecionadas no
   protocolo congelado. A seleção e a dependência limitam a generalização;
   não alegamos que bastem para uma população ampla.
5. **O que ficou fora da proposta?** FAISS ativo, comparação ampla de modelos,
   escala, latência controlada e estudo de usabilidade não foram realizados.
6. **É possível reproduzir tudo só com o pacote leve?** Não. Ele permite
   conferir agregados e procedência; dados/pesos têm condições próprias e os
   rankings completos não foram persistidos. AP individual não pode ser
   recalculada integralmente somente a partir do Top-10 salvo.
7. **Como foram protegidas as pessoas?** O PDF não contém fotos, recortes,
   embeddings ou rankings vinculáveis. Isso não substitui análise institucional
   das condições de uso e divulgação.

## Perguntas que você ainda precisa responder

- Você consegue explicar o cálculo de AP com suas próprias palavras?
- A descrição do sistema corresponde ao que você reconhece e consegue demonstrar?
- Você concorda com a interpretação dos objetivos parcialmente atendidos?
- Há alguma frase cuja autoria ou sentido você não está confortável em assumir?
- A orientação confirma título, recorte e formato final?
- Curso e coordenação confirmam o calendário e os procedimentos de entrega?
- Estão documentadas as condições de uso dos dados e o destino de um backup?

Nenhuma dessas respostas está presumida. Registre suas correções por capítulo
e página. O próximo passo mínimo é ler resumo, método e conclusão e levar as
pendências institucionais à orientação pelo seu próprio canal.
