# Síntese da revisão bibliográfica e posicionamento da monografia

Consolidação: **28/09/2026**. A base bibliográfica anterior permanece em
[bibliografia anotada](bibliografia_anotada.md). Esta atualização corrige o
posicionamento e as alegações; **não é uma nova revisão sistemática nem uma
declaração de leitura integral de todas as referências**.

## 1. Contribuição que a evidência permite defender

O trabalho avalia se uma soma ponderada de semelhança facial e conteúdo da
imagem inteira melhora a recuperação de fotografias de uma identidade
anotada, em um protocolo delimitado de fotos com múltiplas pessoas.

A contribuição é uma comparação experimental rastreável, apoiada em um
sistema implementado e testado. Não é um novo modelo de reconhecimento,
novo método de treinamento, primeira combinação de rosto e contexto,
solução de privacidade ou comprovação de escala.

A implementação modular e a interface ajudam a reproduzir e demonstrar
o método; não substituem a justificativa científica da pergunta e do protocolo.

## 2. Por que não alegar que ninguém fez algo parecido

Há trabalhos anteriores diretamente próximos. [Beyond Frontal Faces](https://arxiv.org/abs/1501.05703)
explora múltiplas pistas para reconhecer pessoas em álbuns; [Person Recognition
in Personal Photo Collections](https://arxiv.org/abs/1509.03502) analisa pistas
corporais e limitações de benchmarks. Os resumos e metadados primários desses
dois trabalhos foram reconsultados em 28/09/2026.

O catálogo também aponta recuperação de pessoas em coleções de consumidores,
reconhecimento assistido por contexto e modelos contextuais multinível.
Esses trabalhos devem ser retomados na escrita; não foram integralmente
reavaliados neste fechamento.

Portanto, retirar frases como “a literatura trata as partes separadamente”
ou “poucos trabalhos fazem isso” sem busca e comparação bibliográfica que
sustentem a quantificação. A diferença do presente protocolo deve ser explicada
comparando tarefa, unidade recuperada, consulta, representação, fusão e métricas,
não usando apenas a existência de uma arquitetura ou interface.

## 3. Escolhas metodológicas

- **ArcFace/InsightFace:** representam rostos por descritores numéricos e
  comparam semelhança; o projeto não treina uma classe nova para cada pessoa.
- **ResNet50 pré-treinada:** baseline global de imagem inteira. Não é uma
  representação exclusiva do fundo: também contém pessoas, objetos e cena.
- **Fusão tardia:** combina escores em vez de concatenar diretamente vetores
  de espaços diferentes. É uma escolha operacional simples, não prova de
  superioridade sobre concatenação ou outras estratégias.
- **Remapeamento do cosseno:** `(cos + 1) / 2` leva o intervalo a [0,1].
  Isso não é calibração estatística, probabilidade ou aprendizagem dos pesos.
- **Pesos fixos:** 1/0, 0/1, 0,9/0,1, 0,7/0,3 e 0,5/0,5. Não se apresenta
  a melhor configuração observada como otimização validada em dados independentes.
- **Retorno:** fotografias completas; sem confundir face detectada, identidade
  anotada e presença humana não anotada.

## 4. Como interpretar o resultado atual

O baseline facial teve o maior mAP neste protocolo. A fusão 0,9/0,1 apresentou
dois ganhos, nove empates e nove perdas, ficando abaixo na média.
A comparação completa está em [resultados experimentais](resultados_experimentais_congelados.md).

É defensável afirmar: “nas consultas e pesos avaliados, a incorporação do
descritor global não melhorou o mAP agregado”. Não é defensável afirmar:

- contexto é sempre inútil;
- a fusão falhou como implementação;
- o fundo causou os erros;
- o resultado é estatisticamente significativo ou generalizável;
- um resultado negativo, por si só, garante publicação.

Influência de fundo, coocorrência ou desequilíbrio dos escores pode ser discutida
como hipótese, não como mecanismo demonstrado. Não foram feitas ablação de
fundo, calibração, novo treinamento ou avaliação em outra coleção.

## 5. Papel das três bases e da aplicação

| Recurso | Papel no artigo | O que não demonstra |
|---|---|---|
| Gallagher | comparação principal, com anotações e exclusão da fonte | toda presença humana, domínio agro, população geral ou larga escala |
| LFW | verificação auxiliar de recuperação facial customizada | acurácia oficial de verificação ou independência do pré-treinamento |
| Holidays | verificação auxiliar global, AP adaptada | mAP diretamente comparável ao avaliador oficial |
| Álbum familiar | aceitação privada da aplicação | benchmark de identidade, consentimento público ou evidência principal |

A interface já possui escolha explícita do rosto e gestão de álbuns;
essas funções não devem aparecer como trabalhos futuros ainda não implementados.
O perfil Pesquisa é exploração interativa e não executa o benchmark automaticamente.

## 6. Estrutura para a escrita posterior

1. Problema de recuperação de fotografias e pergunta delimitada.
2. Trabalhos relacionados próximos, com comparação explícita das tarefas.
3. Representações, unidades, fusão, pesos, exclusões e tratamento de falhas.
4. Protocolo principal Gallagher e verificações auxiliares separadas.
5. Resultados agregados e por consulta, ligados à execução limpa de 28/09.
6. Discussão do resultado observado, dependência entre consultas e limites.
7. Reprodutibilidade permitida, dados/modelos, ética e uso real de IA.
8. Conclusão sem extrapolação e trabalhos futuros opcionais.

Não inserir fotos pessoais ou de terceiros sem elegibilidade de exibição.
Não anunciar disponibilidade pública de código/dados antes de existir.
Metadados e referências ainda precisarão da conferência humana específica
do manuscrito e das normas da revista.

## 7. Trabalhos futuros fora do fechamento

Outras representações, calibração, múltiplas consultas por identidade, outras
coleções autorizadas, busca aproximada e avaliação de escala podem ampliar
o estudo. Não são condições automáticas para encerrar este experimento delimitado.

Agrishow e WIDER FACE continuam históricos e fora do artigo definido. Não
reintroduzir domínio agro, novos modelos ou treinamento para tentar obter
um resultado positivo. A elegibilidade ética e institucional não é trabalho
futuro opcional: continua condição para a eventual submissão.


## 8. Consolidação para a monografia, posterior ao fechamento

A monografia em português é a prioridade; um artigo em inglês será derivado
posteriormente. A seleção atual, com versões, trechos consultados e lacunas,
está em [CONTROLE_FONTES.md](monografia/CONTROLE_FONTES.md). Os registros acima
sobre o fechamento não são apresentados como leitura integral realizada agora.

A redação retomou as introduções e seções pertinentes de Zhang, Oh e Li,
distinguindo reconhecimento contextual de pessoas e ranking de fotografias.
Costache foi utilizado apenas para o precedente descrito em seu resumo
institucional, com acesso incompleto declarado. A busca recente delimitada
incluiu Messina et al. (ECIR 2025; arXiv v2), como recuperação cruzada com
identidade, e não comparação quantitativa direta. Não se iniciou revisão sistemática.

Foram esclarecidos os modelos reais: SCRFD/ArcFaceONNX no pacote buffalo_l e
ResNet50 DEFAULT/IMAGENET1K_V2 em torchvision 0.26. FaceNet é apenas fundamento.
Manning sustenta a fórmula local de AP; a página Holidays informa o protocolo
oficial cuja integração difere da fórmula usada. Os clássicos pertinentes
foram preservados; rótulos antigos de obrigatoriedade foram retirados do catálogo.

A conclusão continua restrita: nenhuma fusão avaliada superou a busca facial
no mAP agregado atual. Deltas foram recalculados das linhas gravadas; não houve
nova execução científica. As três consultas candidatas descartadas na preparação
foram explicitadas como limite de seleção, sem confundi-las com troca por rosto
vizinho. Página principal LFW e página Gallagher seguem com acesso indisponível;
o relatório original LFW foi recuperado no domínio do autor.
