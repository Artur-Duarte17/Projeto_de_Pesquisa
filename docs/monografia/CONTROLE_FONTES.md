# Controle das fontes utilizadas

Conferência documental em 28/09/2026. Este registro separa metadados, leitura e
alcance da afirmação. Não declara leitura integral de todos os documentos.
A bibliografia anterior foi o ponto de partida; itens catalogados não se tornaram
citações obrigatórias. A chave abaixo corresponde a `referencias.bib`.

## Fontes efetivamente citadas

| Chave e versão | Metadados conferidos | Conteúdo consultado e função no texto | Limite |
|---|---|---|---|
| Smeulders2000 | Autores, título, TPAMI 22(12), 1349–1380, 2000; DOI 10.1109/34.895972, Crossref e PDF local. | Resumo, introdução e seção de lacuna semântica; cap. 1–2, definição de CBIR. | Não descreve o desempenho do sistema atual. |
| Dubey2022 | Autor, título, TCSVT 32(5), 2687–2704; DOI 10.1109/TCSVT.2021.3080920. | Resumo e introdução da cópia local; cap. 2, evolução de descritores profundos. | Publicação em volume de 2022, apesar do DOI/publicação antecipada em 2021. Não é revisão sistemática desta monografia. |
| Babenko2014 | Autores, título, ECCV 2014, 584–599; DOI 10.1007/978-3-319-10590-1_38. | Primeiras seções do preprint arXiv 1404.1777v2; cap. 2, representações de redes para recuperação. | Versão lida é preprint; não transferir ganhos numéricos ao presente sistema. |
| He2016 | Autores, título, CVPR 2016, 770–778; DOI 10.1109/CVPR.2016.90. | Resumo e introdução do PDF local; cap. 2, aprendizado residual. | Artigo não identifica sozinho o checkpoint torchvision utilizado. |
| Schroff2015 | Autores, título, CVPR 2015, 815–823; DOI 10.1109/CVPR.2015.7298682. | Resumo e introdução; cap. 2, espaço de representações faciais. | FaceNet não foi implementado no projeto. |
| Deng2019 | Autores, título, CVPR 2019; DOI 10.1109/CVPR.2019.00482. | Introdução e formulação inicial de margem angular; cap. 2–3. | Paginação 4690–4699 da versão CVF/PDF consultada, em vez de 4685–4694 retornada pelo registro Crossref. O artigo e o peso ONNX são objetos diferentes. |
| Guo2021 | Autores, título, arXiv 2105.04714v1, 2021; DOI de repositório 10.48550/arXiv.2105.04714. | Resumo primário; cap. 2, identificação e finalidade de SCRFD. | Não foram usadas alegações de superioridade ou resultados que exigissem leitura experimental integral. |
| Costache2008 | Autores, título, SPIE 6820, 682009; DOI 10.1117/12.766652. | Resumo no repositório institucional da University of Galway; cap. 2, precedente de recuperação de pessoas em coleções de consumidores. | Texto integral não acessado. Comparação explicitamente limitada ao resumo, sem descrever métricas ou resultados não lidos. |
| Zhang2015 | Autores, título, CVPR 2015, 4804–4813; DOI 10.1109/CVPR.2015.7299113. | Introdução e descrição inicial de PIPA/múltiplas pistas no PDF local; cap. 2 e 5. | Reconhecimento em álbuns não equivale ao ranking de fotografias deste protocolo. |
| Oh2015 | Autores, título, ICCV 2015, 3862–3870; DOI 10.1109/ICCV.2015.440. | Introdução, regiões e seção de partições/correlação de contextos; cap. 2, 4 e 5. | Utilizada a conferência de 2015, não o artigo de periódico posterior com título semelhante. |
| Li2016 | Autores, título, CVPR 2016, 1297–1305; DOI 10.1109/CVPR.2016.145. | Introdução e descrição dos níveis de contexto e inferência; cap. 2 e 5. | Modelo contextual estruturado não equivale à soma fixa de dois escores. |
| Atrey2010 | Autores, título, Multimedia Systems 16, 345–379; DOI 10.1007/s00530-010-0182-0. | Resumo, introdução e níveis de fusão; cap. 2. | Vocabulário de fusão; não evidência de ganho no presente experimento. |
| Manning2008 | Autores, título, Cambridge University Press, 2008; DOI 10.1017/CBO9780511809071. | Cópia local e seção 8.4 da edição online dos autores; cap. 2 e 4, AP/mAP e ranking. | A cópia online se identifica como rascunho de 2009; referência bibliográfica corresponde ao livro de 2008. |
| Huang2007 | Autores, título, UMass relatório 07-49, 2007; PDF e índice do autor. Sem DOI identificado. | Descrição da base e avaliação por pares; cap. 4. | Relatório acessível; página principal da coleção indisponível. Não confirma os termos atuais nem legitima equivalência do protocolo local. |
| Jegou2008 | Autores, título, ECCV 2008, 304–317; DOI 10.1007/978-3-540-88682-2_24. | Introdução, descrição da recuperação e página oficial da base; cap. 4. | Não importar métricas de referência como diretamente comparáveis à AP local. |
| Messina2025 | Autores, título, ECIR 2025, 437–452; DOI 10.1007/978-3-031-88708-6_28. | arXiv 2412.21009v2: resumo, introdução e descrição da tarefa/dataset; cap. 2. | Recuperação cruzada com identidade é tarefa adjacente, não concorrente diretamente avaliado aqui. |
| InsightFace | Instituição, título e documentação oficial consultados em 28/09/2026; sem DOI. | README oficial: código MIT, condições distintas de modelos; cap. 3–4. | Página dinâmica, não versão do ambiente científico. Manifestos e código local identificam versão 0.2.1 e pesos efetivos. |
| Torchvision | Instituição, página ResNet50 da documentação 0.26; consulta 28/09/2026; sem DOI. | DEFAULT = IMAGENET1K_V2, dimensões e transformações; cap. 2–3. | Não usar documentação “stable” sem indicar versão. |
| Holidays | Página institucional INRIA da coleção; consulta 28/09/2026; sem DOI. | 1.491 imagens, 500 consultas, avaliação e direitos; cap. 4. | Disponibilidade não implica redistribuição irrestrita. |
| Cockburn2005 | Autor, título, HaT Technical Report 2005.02 e página do autor; sem DOI. | Motivação e separação de portas/adaptadores; cap. 3. | Inspiração arquitetural não certifica aderência completa do software. |
| Brasil2018 | Identificação da Lei 13.709/2018 em fonte oficial, consulta 28/09/2026; sem DOI. | Referência normativa geral em cap. 4. | Sem parecer jurídico, enquadramento artigo por artigo, dispensa ou aprovação ética presumida. |

Os DOI acima foram confrontados com metadados Crossref e as cópias disponíveis,
com a exceção do identificador arXiv, conferido no próprio repositório. As fontes
sem DOI são registradas como tais; nenhum DOI foi inventado. Os campos completos
de autoria e veículo estão no `.bib`, sem duplicar aqui todas as listas de autores.

## Links primários de conteúdo e conferência

- [Costache: resumo institucional](https://research.universityofgalway.ie/en/publications/picture-management-using-person-retrieval-for-consumer-image-coll-11/).
- [SCRFD, versão consultada](https://arxiv.org/abs/2105.04714v1).
- [Messina, texto v2](https://arxiv.org/html/2412.21009v2).
- [Avaliação de rankings, seção 8.4](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).
- [LFW, relatório no domínio do autor](https://people.cs.umass.edu/~elm/papers/lfw.pdf).
- [Índice de publicações do autor, LFW 07-49](https://people.cs.umass.edu/~elm/papers_by_area.html).
- [Holidays e avaliação](https://thoth.inrialpes.fr/~jegou/data.php.html).
- [InsightFace: documentação e condições](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/README.md).
- [ResNet50, torchvision 0.26](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.resnet50.html).
- [Arquitetura hexagonal, autor](https://alistair.cockburn.us/hexagonal-architecture).
- [LGPD, fonte oficial](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm).

## Busca recente delimitada e reclassificação

Busca pontual por recuperação de pessoas/identidade em coleções de fotografias,
com termos de contexto e anos 2024–2026. Consultas incluíram “person retrieval
personal photo collections face context 2024 2025 2026”. Messina et al. foi
incorporado como tarefa adjacente, distinguindo consulta textual/multimodal.
Resultados sobre reidentificação exclusivamente texto–pedestre, PhotoBench e
AlbumFill não foram usados como evidência de desempenho nem tratados como
leitura integral. Esta busca não estabelece exaustividade ou novidade universal.

Os clássicos pertinentes foram mantidos. Datta, Wan, FAISS e fontes adicionais
do catálogo não são obrigatórios na versão atual apenas por já estarem listados.
Gender Shades não fundamenta afirmações sobre viés de recuperação de identidade
neste sistema. Estudos de viés não realizados permanecem limitações e trabalho
futuro, sem resultados presumidos.

## Fontes institucionais e lacunas separadas

O manual do modelo e a estrutura de Isaias foram consultados como material de
formatação, sem importar conteúdo, bibliografia ou documentos de aprovação.
A nota máxima relatada por Artur não foi verificada independentemente.
A proposta assinada foi lida como fonte dos objetivos; seu conteúdo não foi alterado.

A página [oficial do curso de Urutaí](https://www.ifgoiano.edu.br/home/index.php/cursos-superiores-urutai/302-sistema-da-informacao)
forneceu o PPC com nome de arquivo 2026-1, embora o rótulo do link indicava 2025.
Foi consultada sua seção 8.4 (páginas 29–30 do PDF). A aplicabilidade individual
foi informada por Artur e requer confirmação institucional. O calendário de
06/11 e 10/12 não foi revalidado. Detalhes em `PENDENCIAS.md` e `TEMPLATE.md`.

A página Gallagher em Cornell e a página principal LFW não responderam às
tentativas desta rodada. A lacuna diz respeito à confirmação atual das condições
de uso, não aos números locais conferidos. Não foram baixadas bases ou modelos.
Os PDFs e extrações de apoio permanecem fora do Git; a existência de metadados
no cache não equivale a aprovação de todas as fontes retornadas pela busca.
