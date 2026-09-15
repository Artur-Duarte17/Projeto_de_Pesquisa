# Texto extraido da proposta original

Fonte: `C:\OneDriePessoal\OneDrive\Documentos\Projeto IA\Projeto IA.docx`

<!-- p0000 -->
Recuperação de Imagens Fotográficas por Conteúdo e Reconhecimento Facial em Coleções de Grande Escala

<!-- p0001 -->
Resumo do Projeto

<!-- p0002 -->
A presente proposta visa desenvolver um sistema de busca de fotografias baseado em conteúdo (CBIR, Content-Based Image Retrieval) integrado com reconhecimento facial, voltado a atender demandas de fotógrafos profissionais e amadores que lidam com acervos de grande volume. O sistema permitirá localizar imagens por similaridade visual e por identificação de rostos específicos, empregando técnicas modernas de visão computacional e deep learning. A motivação está ancorada no cenário de Big Data de imagens – estima-se que em 2023 sejam capturadas 1,6 trilhão de fotos no mundo , com quase 10 trilhões armazenadas em dispositivos e nuvens – o que torna impraticável a gestão manual. Frente a esse volume, busca-se automatizar a organização e recuperação de fotos, solucionando dores do setor fotográfico como a dificuldade de encontrar imagens não etiquetadas ou agrupar fotos do mesmo cliente/pessoa. Espera-se, ao término, entregar um protótipo funcional que recupere imagens similares ou contendo pessoas-alvo em segundos, melhorando a produtividade, respeitando a privacidade (com processamento local) e contribuindo com avanços técnicos na área de recuperação de imagens e reconhecimento facial.

<!-- p0004 -->
Introdução

<!-- p0005 -->
Vivemos na era da explosão digital de imagens, na qual bilhões de fotografias são produzidas diariamente e armazenadas em meio físico ou na nuvem. Apenas no ano de 2023, projeta-se que 1,6 trilhão de fotos serão tiradas globalmente , um crescimento de 7,5% em relação a 2022, impulsionado pela popularização dos smartphones. Consequentemente, o número de fotos acumuladas cresce exponencialmente – cerca de 10 trilhões de imagens estarão guardadas mundialmente em 2023 . Mesmo usuários comuns lidam com milhares de fotos pessoais (em média,

<!-- p0006 -->
~2.100 fotos por pessoa apenas no smartphone ), enquanto fotógrafos profissionais podem gerir acervos com centenas de milhares de arquivos. Organizar e encontrar uma foto específica nesse mar de dados tornou-se um desafio de Big Data: métodos tradicionais baseados em álbuns manuais ou buscas por palavra-chave muitas vezes falham, pois exigem que as imagens tenham sido previamente

<!-- p0007 -->
descritas ou etiquetadas – tarefa onerosa e sujeita a omissões.

<!-- p0009 -->
No setor fotográfico, essa sobrecarga se traduz em “dores” concretas. Fotografar eventos como casamentos ou campeonatos gera dezenas de milhares de arquivos que precisam ser triados; clientes podem solicitar “todas as fotos onde eu apareço sorrindo” ou “imagens semelhantes àquela do pôr-do- sol”, e atender tais pedidos manualmente consome horas de trabalho. Há, portanto, uma demanda latente por sistemas inteligentes de busca de imagens que sejam capazes de explorar o conteúdo visual das fotos para efetuar consultas. A recuperação de imagem baseada em conteúdo (CBIR) surge como abordagem promissora ao permitir buscar por similaridade visual ao invés de texto: o usuário fornece uma imagem de exemplo (ou seleciona certos atributos visuais), e o sistema retorna fotos visualmente próximas na coleção . Além disso, integrar a dimensão de reconhecimento facial possibilita consultas por pessoa – por exemplo, encontrar todas as fotos contendo o rosto de um

<!-- p0010 -->
determinado indivíduo, independentemente de quando ou onde foram tiradas. Essa funcionalidade é especialmente útil para fotógrafos de eventos que precisam agrupar fotos de cada convidado ou para gestores de bancos de imagens pessoais/familiares que desejam organizar fotos por pessoa.

<!-- p0012 -->
Paralelamente, observa-se a evolução das técnicas de Visão Computacional e Inteligência Artificial aplicadas a imagens. Nos últimos anos, redes neurais profundas revolucionaram tanto o campo de recuperação de imagens quanto o de reconhecimento facial. Com o barateamento de hardware e avanços em algoritmos, soluções que antes eram restritas a gigantes da tecnologia tornaram-se acessíveis a pesquisadores independentes e desenvolvedores. Esse projeto se insere nesse contexto: propõe-se alavancar métodos de ponta de análise de imagens para atenuar o hiato entre a necessidade prática de fotógrafos e o estado atual da tecnologia. A motivação central combina, assim, a perspectiva técnica – investigar e aplicar algoritmos avançados de CBIR e face embedding em cenário de grande escala – e a perspectiva prática – resolver problemas reais de gerenciamento de fotos, aumentando produtividade e agregando valor comercial (por exemplo, oferecendo aos clientes buscas rápidas em seus álbuns digitais).

<!-- p0014 -->
Objetivos

<!-- p0015 -->
Objetivo Geral: Desenvolver um protótipo de sistema inteligente de filtragem e busca fotográfica que integre técnicas de detecção/identificação facial e recuperação de imagens por similaridade, capaz de identificar e reunir de maneira eficiente e precisa todas as fotografias que contenham uma pessoa específica ou sejam visualmente semelhantes a uma imagem de consulta em um grande acervo. Em suma, o sistema deverá permitir tanto buscas por rosto (retornando todas as fotos de um indivíduo-alvo) quanto buscas por similaridade (retornando fotos parecidas com uma fornecida), atendendo às necessidades de organização de fotógrafos em cenários de Big Data.

<!-- p0017 -->
Objetivos Específicos: Para alcançar o objetivo geral, definem-se os seguintes objetivos específicos:

<!-- p0019 -->
Coleta e preparo de dados: Coletar, organizar e rotular um conjunto significativo de imagens de teste, incluindo álbuns fotográficos diversificados (p.ex. eventos, retratos, paisagens), assegurando variedade de cenários, resoluções e condições de iluminação. Incluir bases públicas consagradas (e.g. Labeled Faces in the Wild, CelebA) complementadas por fotos de acervo particular obtidas com consentimento , totalizando inicialmente em torno de 10 mil imagens, com possibilidade de expansão para 100 mil na fase de testes de escalabilidade.

<!-- p0021 -->
Detecção facial: Comparar algoritmos modernos de detecção e alinhamento facial, identificando aqueles que oferecem melhor equilíbrio entre acurácia e desempenho computacional. Serão consideradas soluções baseadas em deep learning devido à robustez a variações de pose e iluminação , como por exemplo o detector RetinaFace ou MTCNN, definindo um componente confiável para localizar faces nas fotos do acervo.

<!-- p0023 -->
Geração de embeddings faciais: Avaliar e selecionar modelos de geração de face embeddings (representações vetoriais de rostos) de última geração, mensurando sua capacidade de produzir descritores discriminativos que diferenciem indivíduos mesmo sob variações de expressão, idade ou oclusões. Modelos como FaceNet, ArcFace, entre outros, serão analisados na literatura e eventualmente testados empiricamente, visando adotar um modelo pré-treinado capaz de mapear cada rosto para um vetor de características onde a distância euclidiana reflita similaridade .

<!-- p0024 -->
Extração de descritores visuais globais (CBIR): Implementar técnicas de CBIR para extrair descritores visuais de imagens completas (não focados apenas em rosto). Isso envolve selecionar algoritmos de características globais – podendo incluir descritores tradicionais (histogramas de cor, HOG, SIFT) e/ou deep features extraídas de redes neurais convolucionais treinadas em reconhecimento de imagens gerais. O objetivo é complementar o embedding facial com informações de cor, textura e forma da cena , de modo que mesmo se o rosto estiver parcialmente oculto ou ausente, a imagem possa ser recuperada pela similaridade visual global.

<!-- p0026 -->
Indexação eficiente dos vetores: Integrar os vetores gerados (embeddings faciais e descritores CBIR) em uma estrutura de indexação de alta performance, capaz de suportar busca por similaridade em grande volume de dados com latência baixa. Será utilizada a biblioteca FAISS (Facebook AI Similarity Search), reconhecida por viabilizar busca aproximada k-NN em milhões de vetores com tempos de resposta em nível de sub-segundo . Técnicas de indexação aproximada como product quantization e grafos HNSW (Hierarchical Navigable Small World) também serão consideradas para otimizar memória e acelerar consultas, se necessário. Em paralelo, explorar a possibilidade de uso do Elasticsearch (que desde versões recentes incorporou busca vetorial k-NN integrada) para gerenciamento distribuído dos índices e metadados, garantindo escalabilidade horizontal .

<!-- p0028 -->
Mecanismo de fusão de resultados: Desenvolver métodos de fusão de escores que combinem as similaridades obtidas pelo módulo facial e pelo módulo de descritores globais, aprimorando o ranqueamento final. Uma estratégia inicial será a soma ponderada das distâncias ou escores de similaridade (atribuindo pesos α e β para rosto e conteúdo, respectivamente) , seguida de um re-ranqueamento dos resultados para promover consistência. Inspirações serão buscadas em técnicas como o Accuracy Noise Reduction proposta por Vieira et al. (2023), que visa atenuar falsos-positivos intermediários na lista de resultados . De fato, Vieira et al. reportaram um aumento de 4–6% na assertividade da recuperação de imagens ao aplicar uma etapa de redução de ruído nos resultados do CBIR . Os parâmetros de fusão (como os pesos α/β) serão determinados empiricamente via validação cruzada, buscando maximizar métricas de precisão.

<!-- p0030 -->
Interface de consulta e visualização: Construir uma interface protótipo amigável que permita ao usuário final interagir com o sistema. Nesta interface, o usuário poderá: (i) selecionar/enviar uma imagem de consulta para busca por similaridade; (ii) selecionar um rosto de referência (dentre faces conhecidas previamente ou extraindo o rosto de uma foto enviada) para busca por pessoa; e então visualizar em tempo real todas as fotografias do acervo que correspondam à consulta, ordenadas por grau de similaridade. Planeja-se utilizar ferramentas open-source para agilizar o desenvolvimento, como por exemplo o framework Streamlit ou uma aplicação web em Flask, de forma a demonstrar concretamente o funcionamento do sistema.

<!-- p0032 -->
Avaliação de desempenho: Mensurar rigorosamente o desempenho do sistema resultante, tanto em termos de eficácia quanto eficiência. Para avaliar a eficácia na recuperação de imagens, serão empregadas métricas consagradas de Recuperação de Informação, em especial: Precision@K (precisão entre os K resultados retornados), Recall@K (taxa de recuperação de relevantes dentro dos K retornos) e a média de precisão média (mAP) sobre um conjunto de consultas de teste . Essas métricas permitirão comparar quantitativamente nosso método com abordagens existentes (por exemplo, baseline de usar somente reconhecimento facial isolado ou somente CBIR isolado). Além disso, serão medidos indicadores de eficiência e escalabilidade, como o tempo médio de resposta por consulta (latência) e o tamanho em memória do índice vetorial para diferentes volumes de imagens. Espera-se, por exemplo,

<!-- p0033 -->
latência de poucos segundos mesmo com 100 mil imagens indexadas, um patamar aceitável para uso prático.

<!-- p0035 -->
Em síntese, os objetivos específicos acima delineiam um plano de trabalho completo, desde a obtenção de dados, passando por escolha/implementação dos algoritmos núcleo, até a construção de uma solução integrada e sua avaliação comparativa. Cumpridas essas etapas, atingiremos o objetivo geral de entregar um protótipo funcional de busca inteligente em acervos fotográficos.

<!-- p0037 -->
Justificativa e Relevância

<!-- p0038 -->
A realização deste projeto se justifica por uma convergência de fatores sociais, técnicos e mercadológicos. Em primeiro lugar, do ponto de vista social e do usuário final, há um claro ganho de produtividade e conveniência: fotógrafos profissionais poderão localizar fotos de um cliente em instantes, otimizando fluxos de trabalho e possibilitando novos serviços (como entrega rápida de coleções personalizadas para cada pessoa fotografada). Usuários comuns, por sua vez, terão uma ferramenta para organizar memórias pessoais de forma inteligente, encontrando “aquele retrato perdido” sem esforço manual. Em termos de relevância social, o projeto também aborda aspectos de privacidade de dados: diferentemente de serviços comerciais existentes (Google Fotos, Amazon Rekognition, etc.), que exigem o envio das imagens para servidores na nuvem, a nossa proposta enfatiza uma solução local e open-source, onde o usuário mantém controle sobre seu acervo e dados biométricos. Isso é particularmente importante em vigência da Lei Geral de Proteção de Dados (LGPD, Lei 13.709/2018) no Brasil, a qual classifica informações biométricas (como faces) como dados sensíveis sujeitos a proteção legal específica. Atendendo a esse contexto legal, o sistema proposto armazenará apenas os embeddings (representações vetoriais) dos rostos, em formato cifrado, mantendo as imagens originais sob controle do usuário. Mecanismos de opt-out e eliminação segura de dados serão incorporados, de modo a respeitar a autonomia do usuário sobre suas informações. Também pretendemos conduzir testes para verificar viés e equidade do reconhecimento – por exemplo, avaliando se o desempenho se mantém consistente entre diferentes grupos demográficos (tons de pele, gêneros), alinhado às melhores práticas de IA responsável.

<!-- p0040 -->
Do ponto de vista técnico-científico, a relevância da proposta está em explorar e integrar fronteiras atuais da pesquisa em visão computacional: a união de CBIR com aprendizado profundo e reconhecimento facial de ponta em um único sistema coeso. Diversos estudos recentes de revisão (surveys) mapeiam os avanços em CBIR e destacam os desafios remanescentes, como a redução do semantic gap e o aumento da escalabilidade . Nosso projeto se apoia nesses levantamentos – por exemplo, Srivastava et al. (2023) fornecem um panorama abrangente sobre métodos de características locais/globais e parâmetros de avaliação em CBIR – e busca contribuir avaliando na prática a eficácia de combinar descritores faciais e globais. O estado da arte em reconhecimento facial já atingiu níveis de acurácia extremamente altos, com arquiteturas profundas (e.g. ArcFace) superando 99% de acerto em benchmarks públicos . Contudo, há lacunas a serem exploradas quando transpostos para cenários aplicados: como lidar com oclusões parciais (rostos parcialmente ocultos) ou consultas sem rosto? Como garantir o desempenho quando o volume de dados cresce de milhares para milhões de imagens (onde os tempos de busca podem degradar)? Nossa pesquisa é relevante por tentar responder a essas questões, experimentando soluções de indexação vetorial eficiente e fusão de evidências que mantenham a precisão sem comprometer a velocidade.

<!-- p0042 -->
Por fim, sob a ótica mercadológica e econômica, o projeto se mostra oportuno. O mercado de gerenciamento de ativos digitais (Digital Asset Management) e ferramentas de busca visual tem se aquecido com o crescimento das demandas por organização automática de fotos. Entretanto, as soluções disponíveis ou são muito limitadas (buscas por tags ou cores básicas em softwares

<!-- p0043 -->
domésticos) ou inacessíveis para fotógrafos independentes (serviços avançados cobrados em dólar e que operam apenas na nuvem). Existe, portanto, uma lacuna por soluções intermediárias: ferramentas acessíveis, que rodem localmente no computador do fotógrafo ou em servidores próprios, incorporando técnicas de IA de forma personalizada e integrada ao fluxo de trabalho existente. O presente projeto pretende justamente preencher essa lacuna, entregando um protótipo que poderá evoluir para uma ferramenta concreta de uso diário. Além disso, os conhecimentos gerados (avaliação de algoritmos de visão, engenharia de indexação de dados em larga escala, etc.) possuem valor acadêmico intrínseco, podendo resultar em publicações científicas e contribuir para a formação de recursos humanos especializados em IA e Big Data.

<!-- p0045 -->
Em suma, a justificativa do projeto repousa em (a) motivações práticas de resolver problemas reais de organização de fotos em massa de forma privada e eficiente; (b) motivações científicas de investigar a integração de técnicas de CBIR e reconhecimento facial sob desafios de grande escala; e (c) motivações estratégicas de democratizar tecnologias de IA avançada para públicos profissionais e amadores fora do circuito das big techs. Acreditamos que os benefícios potenciais – sociais (maior controle do usuário sobre seus dados e memórias), técnicos (avanços em metodologias de recuperação) e econômicos (redução de custos com serviços externos, novos produtos) – justificam amplamente o investimento na execução deste projeto.

<!-- p0047 -->
Fundamentação Teórica

<!-- p0048 -->
Nesta seção, apresentam-se os fundamentos teóricos que embasam o projeto, cobrindo os principais conceitos e trabalhos relacionados em: (1) Recuperação de Imagens por Conteúdo tradicional; (2) CBIR com técnicas de Deep Learning; (3) Reconhecimento Facial moderno; e (4) Indexação e busca de alta performance em Big Data.

<!-- p0050 -->
CBIR tradicional

<!-- p0051 -->
A Recuperação de Imagens Baseada em Conteúdo (CBIR) refere-se a sistemas de busca que permitem encontrar imagens em um banco de dados a partir de uma consulta visual, em vez de texto. Diferentemente da busca textual, em que metadados ou legendas são confrontados, no CBIR o próprio conteúdo intrínseco da imagem – suas cores, formas, texturas e composição – é analisado para determinar similaridades . Sistemas de CBIR clássicos, desenvolvidos a partir dos anos 1990, extraem das imagens descritores numéricos que representam características visuais; por exemplo: histogramas de cor capturam a distribuição de cores, momentos de forma descrevem contornos, e texturas podem ser quantificadas via filtros (como Gabor) ou padrões locais (LBP). Esses descritores são então comparados (usualmente via métricas de distância, como euclidiana ou coseno) para ranquear as imagens do banco conforme sua similaridade com a imagem de consulta.

<!-- p0053 -->
Um dos principais desafios históricos do CBIR é o chamado semantic gap (lacuna semântica): as máquinas podem medir facilmente características de baixo nível (pixels, cores, gradientes), porém nem sempre essas medidas se relacionam com os conceitos de alto nível percebidos pelos humanos (objetos presentes, cena retratada, emoção transmitida) . Por exemplo, duas fotos com fundo azul terão histogramas semelhantes de cor, embora uma possa ser “céu” e outra “oceano” – visualmente distintas para nós. Muitas pesquisas se dedicaram a mitigar essa lacuna, seja incorporando feedback do usuário (relevance feedback) ou usando descritores mais sofisticados. Técnicas tradicionais de CBIR incluem o uso de características locais (como pontos de interesse SIFT/ORB combinados em bag-of-visual-words), que tiveram bastante sucesso em determinados domínios (e.g. reconhecimento de cenas ou objetos específicos), e métodos de redução de dimensionalidade (PCA, LDA) para representar imagens em espaços mais compactos.

<!-- p0054 -->
Apesar de suas limitações, os sistemas clássicos de CBIR pavimentaram o caminho para aplicações importantes, incluindo busca por similaridade em bancos médicos, identificação de peças industriais por imagem, sistemas de recomendação de moda por semelhança visual, entre outros. Esses primeiros sistemas demonstraram a viabilidade de buscar “imagem por imagem” e evidenciaram a necessidade de técnicas mais robustas para vencer o gap semântico. A literatura recente reconhece que o CBIR evoluiu significativamente com a incorporação de métodos de aprendizado de máquina e, sobretudo, aprendizado profundo, conforme discutido a seguir .

<!-- p0056 -->
CBIR com Deep Learning

<!-- p0057 -->
O advento do deep learning catalisou avanços notáveis em CBIR na última década. Diferentemente dos descritores manuais fixos, as redes neurais convolucionais (CNNs) conseguem aprender representações de alto nível diretamente dos dados, ajustando-se ao conceito de similaridade relevante para a tarefa. Em vez de programar explicitamente o que constitui similaridade visual, alimentam-se milhares de imagens a uma CNN para que ela aprenda características discriminativas que melhor distinguem classes ou pares de imagens similares/não-similares. Essa abordagem permitiu reduzir o gap semântico, pois as características extraídas por redes profundas capturam estruturas hierárquicas da imagem – dos bordos e texturas simples em camadas iniciais às composições e objetos inteiros em camadas profundas . Consequentemente, sistemas CBIR modernos apresentam uma compreensão muito mais alinhada com a percepção humana de similaridade.

<!-- p0059 -->
Existem duas vertentes principais no uso de deep learning para CBIR: (a) Transferência de aprendizado – utilizar redes pré-treinadas (como VGG, ResNet, EfficientNet) treinadas em grande conjunto de imagens (p.ex., ImageNet) e extrair as ativações de uma camada intermediária como descritor da imagem. Essa técnica, simples e efetiva, provê um vetor de características profundas (deep features) sem necessidade de treinar um modelo do zero, e tem sido empregada em inúmeros sistemas com sucesso

<!-- p0060 -->
. (b) Aprendizado supervisionado para similaridade – treinar uma rede especificamente para

<!-- p0061 -->
medir similaridade, usando por exemplo arquiteturas siamesas ou triplet loss (como no FaceNet, originalmente aplicado a rostos). Nesse caso, a rede aprende um espaço de embedding em que imagens similares (conforme um critério definido no conjunto de treinamento) ficam próximas, enquanto diferentes ficam distantes. Esse treinamento pode exigir conjuntos rotulados (pares similares/diferentes) ou classes definidas.

<!-- p0063 -->
Nos últimos anos, pesquisas em CBIR com deep learning focaram também em arquiteturas especializadas (como redes para instância específica, p.ex. para reconhecimento de locais ou obras de arte), e em otimização de eficiência, visto que descritores profundos tendem a ser de alta dimensionalidade. Estratégias de compressão (quantização, hashing) e aproximação têm sido incorporadas para viabilizar a busca em larga escala sem perder muita acurácia. Ahmed & Ibraheem (2024), por exemplo, revisaram os avanços dos últimos 6 anos em CBIR com deep learning, demonstrando ganhos expressivos de desempenho em relação a métodos convencionais, especialmente quando combinados com mecanismos de feedback e aprendizado contínuo .

<!-- p0065 -->
Assim, pode-se afirmar que o CBIR atual, potenciado por deep learning, é uma tecnologia madura e em rápida evolução, capaz de indexar e pesquisar conjuntos massivos de imagens com precisão e rapidez muito superiores ao passado. No contexto deste projeto, aproveitaremos esse arcabouço: utilizaremos redes profundas tanto para extrair descritores globais das fotos (por exemplo, utilizando camadas de uma ResNet50 treinada em ImageNet como vetor de características) quanto para representar faces (via modelos de reconhecimento facial profundo). Isso permitirá que nosso sistema herde a capacidade de abstrair conceitos visuais das CNNs e realizar buscas mais semânticas (ex.: encontrar fotos com “ambiente parecido” e não apenas cor igual), minimizando a necessidade de anotações manuais.

<!-- p0066 -->
Reconhecimento Facial Moderno

<!-- p0067 -->
O Reconhecimento Facial automático é um problema clássico da visão computacional que ganhou enorme impulso com técnicas de aprendizado profundo. Historicamente, métodos de reconhecimento de faces passaram por gerações: desde abordagens baseadas em medidas geométricas do rosto e eigenfaces (década de 1990), passando por descritores locais treinados (LBP, Fisherfaces), até as atuais redes neurais profundas que dominam o estado da arte . A pipeline típica de um sistema de reconhecimento facial moderno inclui três etapas principais : (1) Detecção e alinhamento facial – localizar a posição do rosto na imagem e normalizá-lo (por exemplo, alinhando os olhos a posições fixas); (2) Extração de embedding – passar o rosto por uma rede neural profunda para obter um vetor fixo de características (geralmente de 128 a 512 dimensões) que representa aquela face de forma única;

<!-- p0068 -->
(3) Comparação de embeddings – dado dois vetores (ou um vetor de consulta vs. um banco de vetores), calcular uma métrica de distância (tipicamente distância euclidiana ou coseno) para inferir similaridade/identidade. Modelos como FaceNet (2015) foram pioneiros em treinar redes siamesas com triplet loss para diretamente otimizar a separabilidade de faces no espaço vetorial, alcançando então performance revolucionária (FaceNet atingiu ~99,6% de acerto no benchmark LFW já em 2015). Subsequentemente, métodos refinados de perda de classificação com margin angular, como ArcFace (2019), elevaram ainda mais o patamar, tornando possível obter acurácia acima de 99,8% em conjuntos de teste padronizados . Hoje, o reconhecimento facial profundo é uma tecnologia relativamente madura e amplamente empregada em aplicações como desbloqueio de smartphones, controle de fronteiras, vigilância, indexação de fotos pessoais (e.g. agrupamento de rostos no Google Fotos).

<!-- p0070 -->
No contexto deste projeto, o reconhecimento facial é um pilar fundamental: dele dependerá a capacidade do sistema de identificar “fotos da mesma pessoa”. Felizmente, podemos aproveitar décadas de pesquisa consolidada – optaremos por utilizar modelos já treinados disponíveis publicamente, evitando a necessidade de coletar e treinar em um gigantesco dataset de faces do zero. Por exemplo, poderemos usar as implementações open-source do ArcFace, cujos autores liberaram modelos pré-treinados com milhões de rostos, capazes de gerar embeddings com excelente discriminatividade. Com esses modelos, o sistema pode extrair o embedding facial de cada foto em nosso acervo e, na consulta, extrair o embedding da face de interesse; então a busca por rostos semelhantes se reduz a uma consulta de vizinhos mais próximos no espaço vetorial dos embeddings faciais. Essa arquitetura – detectar faces, extrair embeddings e comparar – alinha-se com as melhores práticas descritas na literatura .

<!-- p0072 -->
Um cuidado adicional será integrar o reconhecimento facial com o CBIR global de modo sinérgico. Em certos cenários, somente o embedding facial pode falhar: por exemplo, se o rosto estiver virado ou coberto parcialmente, ou se o usuário buscar “fotos parecidas” sem especificar face. Por isso, nosso sistema combinará ambos: o reconhecimento facial garantirá alta precisão quando a pessoa for o fator determinante, enquanto o CBIR global acrescentará robustez e contexto visual quando necessário . Essa complementaridade, conforme indicado em pesquisas recentes, tende a melhorar os resultados de busca em cenários desafiadores (oclusão, múltiplas pessoas, etc.), já que aproveita duas fontes de informação distintas da imagem.

<!-- p0074 -->
Indexação e Busca em Big Data de Imagens

<!-- p0075 -->
Um dos diferenciais e desafios deste projeto é lidar com grandes volumes de dados visuais, o que exige estratégias de indexação adequadas para garantir desempenho escalável. Indexar imagens para busca por conteúdo significa indexar vetores (embeddings) de alta dimensão. A busca ingênua (linear) comparando o vetor de consulta com todos os vetores do banco rapidamente se torna inviável conforme o número de imagens cresce para centenas de milhares ou milhões – as latências seriam de muitos segundos ou minutos, o que não atende a interatividade esperada. Surge então a necessidade

<!-- p0076 -->
de utilizar estruturas de dados e algoritmos de busca aproximada de vizinhos mais próximos

<!-- p0077 -->
(Approximate Nearest Neighbors – ANN) que aceleram a recuperação com mínima perda de acurácia.

<!-- p0079 -->
Dentre as soluções atuais, destaca-se a biblioteca FAISS (Facebook AI Similarity Search), lançada em 2017 e continuamente aprimorada . O FAISS implementa uma série de algoritmos de indexação vetorial (quantização vetorial, grafos de vizinhança, árvores, entre outros) otimizados para uso tanto em CPU quanto GPU, permitindo realizar buscas k-NN em bases com milhões ou até bilhões de vetores com tempos na ordem de milissegundos ou poucos segundos . Por exemplo, uma configuração comum é usar Product Quantization (PQ) para comprimir vetores e um IVF (Inverted File) para subdividir o espaço vetorial em clusters, fazendo com que a busca ocorra apenas em uma fração dos dados. Outra abordagem é o grafo HNSW (Hierarchical NSW) , que constrói conexões esparsas entre vetores para permitir saltos rápidos na busca aproximada; o HNSW é conhecido por alcançar balanço excelente entre velocidade e acurácia, sendo adotado inclusive em motores de busca de código aberto como NMSLIB e Annoy.

<!-- p0081 -->
O FAISS emergiu como referência na área – já acumulava mais de 30 mil estrelas no GitHub e 4 mil citações acadêmicas até 2023 – e será nossa escolha inicial para indexar tanto os embeddings faciais quanto os descritores CBIR. A configuração ótima (quantização, número de centroids, ligações no grafo etc.) será determinada experimentalmente, ponderando requisitos de memória vs. tempo. Além do FAISS, exploraremos integrações com o Elasticsearch, um motor de busca textual amplamente utilizado que recentemente incorporou funcionalidades de vector search no seu módulo k-NN. O Elastic permite indexar cada imagem como um documento JSON contendo campos vetoriais; consultas vetoriais são então executadas de forma distribuída, tirando proveito do sharding e replicação do cluster para escalar a vários nós. Essa combinação pode ser interessante caso queiramos suportar não apenas similaridade visual pura, mas consultas híbridas envolvendo texto e imagem (por ex., “fotos de Maria e com pôr-do-sol”). Ainda que esse aspecto híbrido não seja o foco principal aqui, mencionar o Elastic reforça a viabilidade de escalabilidade: empresas já demonstram ser possível realizar busca de vetores em coleções vastas com latência de poucos centenas de milissegundos . Em termos de arquitetura, podemos imaginar um fluxo onde o FAISS cuida da busca ANN local e o Elasticsearch gerencia metadados (tags, datas, etc.) e orquestra as consultas num ambiente distribuído.

<!-- p0083 -->
Outro componente teórico relevante é a avaliação e otimização de índices. Existem métricas próprias para algoritmos ANN, como o Recall@K (que porcentagem dos verdadeiros vizinhos mais próximos exatos estão entre os resultados aproximados) e a trade-off acurácia vs. tempo. Faremos uso desses conceitos para calibrar nosso sistema: por exemplo, configurar o FAISS para retornar 2× mais candidatos que o necessário e depois refinar ordenação com distância exata (estratégia narrowing), ou ajustar a profundidade de busca no grafo HNSW para atingir recall de 95% com tempo 10× menor que a busca exata. Essas decisões serão guiadas tanto por recomendações da literatura quanto por experimentação durante o projeto.

<!-- p0085 -->
Em suma, os fundamentos em indexação nos garantem que é possível manejar big data de imagens de forma eficiente. A combinação de embeddings significativos (graças ao deep learning) com estruturas de busca velozes (ANN/FAISS/Elastic) resulta numa solução capaz de entregar ao usuário resultados relevantes quase instantaneamente, mesmo varrendo um universo massivo de fotos. Este projeto aplicará esses conceitos de ponta para viabilizar a experiência de “buscar uma foto em meio a um milhão, como se estivesse em um álbum organizado”, concretizando assim a premissa de unir IA e Big Data para resolver problemas práticos.

<!-- p0086 -->
Metodologia

<!-- p0087 -->
Para atingir os objetivos propostos, o projeto será conduzido em etapas bem-definidas, conforme ilustra a Figura 1 (pipeline geral). As fases metodológicas incluem: preparação de dados, desenvolvimento dos módulos de processamento (detecção facial, extração de embeddings, CBIR), integração e indexação dos componentes, implementação da interface de busca, e por fim avaliação experimental. A seguir detalhamos cada etapa e as técnicas/ferramentas correspondentes.

<!-- p0089 -->
Figura 1: Visão geral da arquitetura proposta. O sistema consiste em (1) um módulo de detecção e alinhamento facial, seguido por (2) extração de embedding facial e de descritor visual global para cada imagem do acervo. Os vetores resultantes são indexados numa estrutura ANN (FAISS/Elastic). Na consulta, o usuário fornece uma imagem; (3) extraem-se seus vetores (facial/global) que são então comparados com os índices, retornando listas de imagens semelhantes por rosto e por conteúdo. Por fim, (4) um módulo de fusão e re-ranqueamento combina os resultados, priorizando imagens que satisfaçam ambos os critérios.

<!-- p0091 -->
Etapa 1 – Coleta e preparação dos dados: Inicialmente, serão reunidos os conjuntos de imagens para desenvolvimento e testes. Utilizaremos dois tipos de fontes: bases públicas e um acervo privado/ particular. Das bases públicas, destacamos a Labeled Faces in the Wild (LFW) – amplamente utilizada em pesquisas de reconhecimento facial, contendo mais de 13 mil fotos de rostos de celebridades em situação espontânea – e o conjunto CelebA, que oferece 200 mil imagens de rostos com variações de pose, iluminação e expressão. Essas bases fornecem diversidade e já vêm com labels (no caso, identidade da pessoa), úteis para avaliar acurácia do módulo facial. Complementarmente, incluiremos um álbum particular composto de fotos obtidas localmente (por exemplo, colaboradores do projeto ou voluntários fornecerão imagens pessoais com consentimento). Esse acervo privado, estimado inicialmente em ~5 mil fotos, servirá para testar o sistema em cenários próximos ao uso real de um fotógrafo (misturando retratos posados, fotos de eventos, ambientes diversos). Todo o conjunto de dados será particionado em pelo menos dois subconjuntos: um para treinamento/desenvolvimento (quando necessário treinar/ajustar algum modelo) e outro reservado para validação/teste dos resultados finais. Antes de alimentar as próximas etapas, as imagens passarão por um pré- processamento padronizado: correção de rotação (metadados EXIF), redimensionamento (diminuir resolução para agilizar processamento, mantendo qualidade suficiente), e alinhamento facial – nas fotos que contêm rosto, aplicar alinhamento geométrico (centralizar olhos/horizonte) para reduzir variações desnecessárias .

<!-- p0093 -->
Etapa 2 – Implementação do módulo de reconhecimento facial: Consiste em integrar um detector/ alinhador de faces e um gerador de embeddings faciais. Utilizaremos bibliotecas consolidadas como OpenCV (que possui detectores Haar Cascade e DNNs) ou diretamente implementações de estado da arte como MTCNN ou RetinaFace para detectar rostos com alta acurácia. Após detectar, cada rosto será alinhado e então passado a um modelo de embedding. Pretendemos usar o modelo ArcFace pré- treinado (disponível via framework InsightFace) ou similar, que produz embeddings de 512 dimensões. Esse modelo será carregado utilizando o PyTorch, permitindo execução acelerada por GPU. Durante esta etapa, faremos experimentos para validar a qualidade dos embeddings no nosso conjunto: selecionaremos algumas identidades conhecidas e verificaremos se fotos da mesma pessoa efetivamente resultam em vetores mais próximos entre si do que de outras pessoas (esperamos que sim, dado o desempenho reportado de ArcFace ). Caso necessário, poderemos refinar o modelo via fine-tuning em nosso conjunto (por exemplo, se quisermos incluir pessoas específicas não bem representadas no treino original), embora isso tenda a não ser necessário dado o uso de modelo amplo.

<!-- p0094 -->
Etapa 3 – Implementação do módulo CBIR global: Paralelamente, desenvolveremos o componente de extração de descritores visuais para a imagem inteira. Aqui investigaremos duas abordagens: (i) utilizar uma CNN pré-treinada genérica, como ResNet50 (ImageNet), extraindo o vetor da penúltima camada (de ~2048 dimensões) como descritor global; (ii) testar descritores clássicos ou híbridos, como histogramas de cor concatenados com algum descritor local agregado. A abordagem (i) é a mais provável, pois pesquisas indicam que deep features generalistas já superam em muito os descritores manuais em tarefas de similaridade . Implementaremos a extração no PyTorch ou TensorFlow, aproveitando modelos disponíveis (por exemplo, pode-se usar o modelo ResNet do torchvision e simplesmente remover a camada final). Cada imagem do acervo, mesmo sem rosto, terá então associado um vetor de características profundas. Assim, ao final das etapas 2 e 3, para cada imagem teremos até dois vetores: um facial (se houver face) e um global.

<!-- p0096 -->
Etapa 4 – Indexação dos vetores (FAISS/Elastic): Esta é uma etapa central para garantir que o sistema escale. Utilizaremos inicialmente o FAISS para indexar os embeddings. Provavelmente manteremos dois índices separados: um índice para os vetores faciais e outro para os vetores globais. Isso porque as distâncias nesses espaços podem não ser comparáveis diretamente (sem normalização) e porque nem toda imagem terá vetor facial (fotos sem pessoas). Com dois índices, podemos realizar buscas independentes e depois combinar resultados. Configuraremos o FAISS com método apropriado; por exemplo, poderemos usar IndexIVFFlat (inverted file with exact codes) durante desenvolvimento para simplicidade, e depois experimentar IndexIVFPQ (com quantização) ou IndexHNSW para lidar com coleções maiores com menos memória. O critério de comparação será principalmente a distância euclidiana (ou coseno, equivalentes após normalização) entre vetores. Uma vez construído o índice (fase de treino do índice, onde aplicável, e adição de todos os vetores), integraremos também consultas via ElasticSearch: utilizando seu plugin k-NN, podemos inserir os mesmos vetores em um índice Elastic (que internamente pode usar HNSW). Isso nos permitirá comparar desempenho FAISS vs. Elastic e também demonstrar a possibilidade de distribuição. Vale notar que, em ambiente de protótipo local, o FAISS supre bem; a adoção do Elastic seria mais relevante em uma futura fase de produto, com necessidade de clusterizar a solução.

<!-- p0098 -->
Etapa 5 – Mecanismo de busca e fusão: Com os índices preparados, implementaremos a lógica de busca do sistema. Quando o usuário fizer uma consulta fornecendo uma imagem Q, o sistema seguirá este fluxo: (i) detectar/alinhar rosto em Q (se existir); (ii) extrair embedding facial de Q usando o modelo definido; (iii) extrair descritor global de Q; (iv) consultar o índice facial com o vetor (ii) obtendo uma lista de imagens candidatas ranqueadas pela semelhança de rosto; (v) consultar o índice global com o vetor

<!-- p0099 -->
(iii) obtendo uma lista de imagens ranqueadas por similaridade visual; (vi) combinar os resultados – aqui aplicaremos o método de fusão, que inicialmente será a soma ponderada das distâncias rankeadas. Por exemplo, dado uma imagem X no acervo, podemos calcular um escore total:

<!-- p0100 -->
, onde d_face e d_global são as distâncias (normalizadas entre 0 e 1) de Q para X nos respectivos espaços; α e β controlam a importância de cada critério. Alternativamente, podemos combinar listas selecionando interseções: por exemplo, primeiro pegar top 100 por rosto e top 100 por global, depois ordenar aqueles que aparecem

<!-- p0101 -->
em ambas com preferência – mas a estratégia linear de combinação de escores tende a ser mais flexível. Depois de combinados, teremos um ranque final de imagens, do qual selecionaremos os Top K (por exemplo, K=20 para exibir). Caso haja múltiplas faces na imagem de consulta, poderemos pedir ao usuário para escolher qual rosto é o alvo principal (ou buscar por todos). Caso a imagem de consulta não contenha rosto, o sistema simplesmente se baseia no CBIR global. Implementaremos essa lógica em Python, possivelmente encapsulada em funções para facilitar manutenção (por ex., uma função que retorna a lista de resultados). Faremos também uso de técnicas de pós-

<!-- p0102 -->
processamento sugeridas em trabalhos como o de Vieira et al. (2023): por exemplo, o Accuracy Noise

<!-- p0103 -->
Reduction citado na fundamentação, que implicaria identificar resultados potencialmente ruidosos (falsos positivos) no meio da lista e ajustá-los . Poderemos implementar um simples filtro baseado

<!-- p0104 -->
em limiar: se a distância facial do resultado for muito alta (pouca confiança), removê-lo mesmo que a similaridade global seja alta, e vice-versa – garantindo que os resultados finais exibidos tenham coerência nos dois critérios.

<!-- p0106 -->
Etapa 6 – Desenvolvimento da interface e teste integrado: Em paralelo às etapas 4-5 (que podem ser desenvolvidas em ambiente de notebooks inicialmente), construiremos a interface de usuário do protótipo. Optamos por utilizar o Streamlit, uma biblioteca Python que permite criar dashboards web interativos de forma rápida. Com Streamlit, podemos montar uma página web local onde o usuário escolhe uma imagem (upload ou do disco), e o sistema roda a consulta em segundo plano, exibindo os resultados na própria página. A interface mostrará as imagens retornadas em forma de galeria ordenada, e poderá oferecer filtros adicionais (por exemplo, filtrar por intervalo de data ou por etiqueta se tais metadados estiverem disponíveis – embora isso seja um extra). Após integrar a interface com o backend de busca, teremos, de fato, um sistema funcional ponta-a-ponta rodando localmente. Realizaremos então uma bateria de testes qualitativos iniciais, simulando casos de uso: buscar rosto de uma pessoa conhecida e verificar se todas as fotos dela aparecem no topo; buscar uma imagem de paisagem e observar se retorna cenas parecidas; medir tempo de resposta para diferentes tamanhos de consulta (ex.: 1 mil vs 10 mil imagens no acervo) e otimizar configurações se necessário.

<!-- p0108 -->
Etapa 7 – Avaliação experimental: Finalmente, conduziremos uma avaliação quantitativa formal conforme delineado nos objetivos específicos. Montaremos um conjunto de consultas de teste padronizadas – por exemplo, selecionar 50 fotos-alvo, sendo 25 de pessoas e 25 de cenários – juntamente com um gabarito manual do que seriam resultados relevantes (para calcular métricas de precisão/recall). Com isso, executaremos o sistema nessas consultas e calcularemos Precision@K, Recall@K e mAP. Essas métricas serão computadas possivelmente com auxílio de bibliotecas de avaliação de IR ou manuseio de rankings. Também mediremos a latência média por busca (provavelmente via timestamps no código) e o uso de memória/disco do índice (via análise dos arquivos gerados pelo FAISS ou do index do Elastic). Os resultados serão compilados e comparados com benchmarks sempre que possível. Por exemplo, se encontrarmos algum trabalho relacionado similar, poderemos confrontar nossas métricas; caso não, usaremos as métricas absolutas para verificar se atendem a critérios aceitáveis (p.ex., precision@10 acima de 0.8 para buscas por pessoa; latência abaixo de 2 segundos em 100k imagens, etc., conforme metas estimadas). Essa etapa também servirá para validar as hipóteses: confirmaremos se a fusão realmente melhorou a qualidade em relação aos módulos isolados (podemos rodar a avaliação separadamente: só facial, só global, combinado, e comparar mAP). Adicionalmente, caso detectemos algum viés ou caso de falha (por exemplo, um grupo demográfico com desempenho inferior), documentaremos e, se houver tempo, proporemos ajustes (p.ex., calibrar limiares ou incluir dados mais diversificados).

<!-- p0110 -->
Etapa 8 – Documentação dos resultados: Concomitante à avaliação, iniciaremos a redação dos resultados e análises para compor o relatório final (TCC). Iremos documentar todo o processo, decisões de projeto, dificuldades encontradas e soluções adotadas. Gráficos ou tabelas relevantes (como curva de precisão por número de resultados, tempos de busca vs. tamanho do índice) serão produzidos para ilustrar os achados. Também registraremos cases de uso demonstrando o sistema em funcionamento (por exemplo, screenshots da interface retornando resultados corretamente). Essa documentação servirá tanto para avaliação acadêmica quanto para comunicar o valor do trabalho a terceiros interessados.

<!-- p0112 -->
Em termos de ferramentas e recursos para viabilizar todas as etapas acima, vale destacar que utilizaremos majoritariamente ferramentas open-source e infraestrutura acessível. A programação será em Python (pela riqueza de bibliotecas de IA disponíveis). Além das já mencionadas (PyTorch, OpenCV, FAISS, Streamlit, Elastic), usaremos ferramentas auxiliares como scikit-learn (eventualmente para validar alguma clusterização ou reduzir dimensionalidade), numpy/pandas para manipulação de

<!-- p0113 -->
dados e matplotlib/seaborn para gerar visualizações de resultados. Para execução de treinamentos ou inferências intensivas, contamos com acesso ao Google Colab Pro, que fornece instâncias de GPU (como Tesla T4 ou P100) a baixo custo, permitindo treinar modelos ou processar lotes de imagens mais rapidamente. Também temos disponibilidade de uma GPU local (por exemplo, uma Nvidia RTX 3060 no laboratório de IA) que será utilizada nas fases de experimentação pesada . Tarefas como indexação e busca ANN, que podem ser executadas em CPU de forma paralela, serão realizadas em estações de trabalho convencionais se possível, aproveitando múltiplos núcleos. A infraestrutura de armazenamento necessária não é muito grande – 100 mil imagens mesmo em alta resolução ocupam alguns dezenas de gigabytes, viáveis em um HD comum. Assim, não identificamos impedimentos técnicos ou de recursos à execução do projeto: pelo contrário, o ambiente atual de desenvolvimento em IA, aliado às plataformas de computação em nuvem e bibliotecas disponíveis, tornam este um momento oportuno para realizar a proposta com custo baixo e alta eficiência.

<!-- p0115 -->
Resultados Esperados e Impactos

<!-- p0116 -->
Ao término do projeto, espera-se obter resultados concretos tanto no aspecto do sistema desenvolvido quanto no conhecimento gerado:

<!-- p0118 -->
Do ponto de vista prático, o resultado principal será um protótipo funcional de software capaz de realizar buscas em um acervo fotográfico extenso por similaridade visual e por reconhecimento facial integrado. Esse protótipo, validado em nossos experimentos, deverá ser capaz de retornar, em questão de segundos, imagens relevantes à consulta do usuário, mesmo em um conjunto de dezenas de milhares de fotos. Idealmente, demonstrará alto desempenho de acurácia, por exemplo: para

<!-- p0119 -->
consultas por pessoa, recuperar >90% das fotos da pessoa entre os top 20 resultados (Recall@20 0.9),

<!-- p0120 -->
e para consultas por similaridade geral, apresentar na maioria resultados visualmente coerentes com a consulta (mAP global acima de 0.8 em cenários testados). Espera-se também que o sistema mantenha tempos de resposta interativos – meta de ~1 segundo para retornar resultados numa base de 10 mil imagens, e escalável a poucos segundos em 100 mil, o que seria uma ordem de grandeza mais rápido que uma busca manual tradicional. Esses indicadores de sucesso serão detalhados no relatório final.

<!-- p0122 -->
Em termos de impacto para o usuário final (fotógrafos e entusiastas), o sistema promete revolucionar a forma de gerenciar acervos fotográficos. Tarefas antes tediosas, como catalogar e separar fotos por cliente ou tema, poderão ser realizadas de forma automática e ágil. Isso libera tempo do profissional para atividades mais criativas (edição artística, atendimento ao cliente) em vez de trabalho braçal de organização. Além disso, a possibilidade de entregar ao cliente final um serviço diferenciado – por exemplo, um fotógrafo de eventos fornecendo um portal onde cada convidado pode entrar e encontrar todas suas fotos pelo rosto – agrega valor e potencialmente vantagem competitiva no mercado. No âmbito pessoal, o projeto pode resultar futuramente em uma ferramenta útil para qualquer pessoa organizar suas memórias digitais sem abrir mão da privacidade, diferentemente dos grandes serviços que monetizam dados.

<!-- p0124 -->
Do ponto de vista científico e tecnológico, os resultados esperados incluem a validação empírica de que a combinação de reconhecimento facial e CBIR traz benefícios mensuráveis. Se nossas hipóteses se confirmarem, teremos evidências de aumento de precisão ao usar a fusão de descritores, o que poderá ser publicado como um estudo de caso ou integrado a um artigo. Mesmo que alguns resultados não sejam superiores, documentaremos as lições aprendidas – por exemplo, se identificarmos cenários onde um método falha, isso indica oportunidades de pesquisa futura (como melhorar embeddings para determinadas condições). Espera-se também produzir um repositório de código bem documentado, que poderá ser disponibilizado como open-source para a comunidade, facilitando que outros reproduzam e construam em cima do nosso trabalho.

<!-- p0125 -->
Em termos de impactos sociais mais amplos, o projeto promove a democratização do acesso à IA. Ao visar uma solução local e open-source, difundimos conhecimento e ferramentas que podem ser adotadas por pequenos negócios, fotógrafos autônomos, pesquisadores iniciantes – grupos que muitas vezes ficam à margem das tecnologias de ponta por questões de custo ou complexidade. Adicionalmente, a aderência do sistema à LGPD e a preocupação com vieses contribuem para a pauta de IA ética e centrada no usuário, mostrando na prática como equilibrar inovação com responsabilidade.

<!-- p0127 -->
Podemos extrapolar ainda impactos potenciais em áreas correlatas: a tecnologia desenvolvida poderia ser adaptada, por exemplo, para sistemas de encontro de crianças perdidas (buscando por rostos de desaparecidos em bancos de imagens públicas), ou para auxiliar organizações jornalísticas a pesquisarem rapidamente seu acervo de fotos por pessoas e contextos (substituindo a necessidade de lembrar palavras-chave). Embora esses não sejam focos diretos agora, demonstram a versatilidade e relevância do núcleo tecnológico proposto.

<!-- p0129 -->
Em resumo, os resultados esperados englobam: um protótipo validado (com métricas de desempenho conhecidas), documentação completa do projeto, e possivelmente publicações ou apresentações em eventos acadêmicos. Os impactos positivos esperados incluem maior eficiência na gestão de grandes coleções de imagens, respeito à privacidade e autonomia de dados, fortalecimento do conhecimento local em IA aplicada, e abertura de caminhos para inovações futuras no cruzamento de Visão Computacional e Big Data.

<!-- p0131 -->
Cronograma

<!-- p0132 -->
O cronograma a seguir apresenta o planejamento das atividades ao longo de 12 meses, correspondentes ao desenvolvimento do trabalho de conclusão:

<!-- p0134 -->
Mês 1-2: Levantamento Bibliográfico e Planejamento Detalhado. Revisão de literatura sobre CBIR, reconhecimento facial e indexação (focando em surveys 2023-2024 ). Estudos de trabalhos relacionados e definição final das tecnologias a serem usadas. Refinamento do escopo e métricas-alvo. (Entrega: capítulo inicial de Referencial Teórico, conjunto de artigos-chave comentados).

<!-- p0136 -->
Mês 3: Coleta/Construção do Conjunto de Dados. Download e organização das bases públicas (LFW, CelebA, etc.). Aquisição de imagens para o acervo particular, incluindo termos de consentimento. Anotação/organização das identidades nas fotos, separação em pastas ou planilhas. Pré-processamentos iniciais (redimensionamento, limpeza de dados corrompidos). (Entrega: dataset consolidado, relatório com estatísticas do dataset).

<!-- p0138 -->
Mês 4: Desenvolvimento do Módulo de Reconhecimento Facial. Implementação de detecção e alinhamento de faces; integração do modelo de embedding selecionado (ArcFace ou similar) usando exemplos de código existentes. Testes unitários: para algumas faces conhecidas, verificar se o embedding distingue corretamente. (Entrega: código funcional de módulo facial, com documentação de uso e resultados de teste preliminar).

<!-- p0140 -->
Mês 5: Desenvolvimento do Módulo CBIR Visual. Implementação de extração de descritor global via CNN (e/ou outros métodos); geração de descritores para um subconjunto de imagens e inspeção manual da similaridade (verificar se imagens parecidas têm vetores próximos). Ajustes nos parâmetros se necessário (por ex., qual camada usar, necessidade de normalização).

<!-- p0141 -->
(Entrega: código do módulo CBIR, incluindo função para extrair descritor de uma imagem e armazenamento dos vetores).

<!-- p0143 -->
Mês 6: Integração e Indexação Inicial (FAISS). Inserção dos embeddings faciais e descritores globais das imagens de desenvolvimento em um índice FAISS. Experimentação com tipos de índice (Flat vs IVFFlat vs HNSW) e medição de tempos de busca em diferentes configurações. Seleção de uma configuração ótima para prosseguir. (Entrega: índice criado para dataset de desenvolvimento; relatório breve com tempos de busca vs. precisão aproximada para configurações testadas).

<!-- p0145 -->
Mês 7: Implementação do Mecanismo de Consulta e Fusão. Desenvolvimento da lógica de busca combinada (como descrito na metodologia). Testes manuais de consultas, ajustando pesos α/β para equilibrar resultados. Implementação opcional do módulo de re-ranqueamento (ANR) e comparação de listas antes/depois. (Entrega: função de busca unificada retornando resultados combinados; evidências de melhora com fusão, ex. lista de acertos sem fusão vs. com fusão).

<!-- p0147 -->
Mês 8: Desenvolvimento da Interface de Usuário. Construção da interface (Streamlit app) conectando com a função de busca. Design simples mas funcional, exibindo imagens retornadas e permitindo novas consultas de forma iterativa. Teste do sistema completo localmente com usuários simulados (colegas) para feedback de usabilidade. (Entrega: protótipo integrando front- end e back-end, rodando em ambiente local; coleta de feedback e identificação de eventuais bugs de integração).

<!-- p0149 -->
Mês 9: Avaliação Quantitativa e Ajustes Finais. Execução das consultas de teste definidas e cálculo das métricas (Precision@K, Recall@K, mAP, tempo médio). Compilação dos resultados e comparação com expectativas. Se alguma métrica ficar muito aquém, iterar ajustes: ex. se recall facial baixo, considerar mudar modelo ou limiar; se tempo alto, otimizar index ou usar quantização. Reavaliar após ajustes. (Entrega: tabela de resultados quantitativos finais; análise escrita explicando cada métrica e qualquer trade-off observado).

<!-- p0151 -->
Mês 10: Documentação do Projeto (Relatório Parcial). Iniciar escrita dos capítulos finais do TCC: Metodologia (descrevendo pipeline, já em boa parte elaborado), Resultados (apresentando métricas e discussões), e Conclusão/Trabalhos Futuros. Gerar figuras (diagramas, gráficos) necessários para ilustrar arquitetura e desempenho. (Entrega: rascunho avançado do relatório, cobrindo do introdução até resultados, a ser revisado com orientador).

<!-- p0153 -->
Mês 11: Revisão, Formatação e Preparação da Defesa. Revisão detalhada do texto segundo normas ABNT, ajustes de linguagem e clareza. Formatação final de referências, sumário, etc. Preparação de slides ou demonstração para eventual apresentação do projeto (banca). (Entrega: versão quase final do TCC; slides de apresentação e protótipo pronto para demonstração ao vivo).

<!-- p0155 -->
Mês 12: Entrega Final e Divulgação. Entrega oficial do TCC escrito. Apresentação para banca examinadora (defesa) incluindo demonstração prática do sistema funcionando com casos de exemplo. Coleta de sugestões da banca e público. Pós-defesa, considerar divulgação do código em repositório público (GitHub) e redação de um artigo resumindo os achados para submissão em conferência ou workshop (se viável). (Entrega: TCC final aprovado; repositório público lançado; artigo em preparação – este último como extra acadêmico).

<!-- p0156 -->
Esse cronograma prevê folgas para eventuais atrasos e iterações, distribuindo as atividades de forma equilibrada ao longo do ano. As fases críticas são as de desenvolvimento de algoritmos (meses 4-7), mas mitigamos riscos ao usar componentes pré-existentes de IA. A avaliação foi alocada com tempo suficiente (mês 9) para refinamentos. Assim, acreditamos que o planejamento é factível e leva em conta tanto a complexidade das tarefas quanto a necessidade de documentação contínua do trabalho.

<!-- p0158 -->
Viabilidade

<!-- p0159 -->
A viabilidade do projeto é analisada sob os aspectos de recursos computacionais, acesso a tecnologias e competências necessárias:

<!-- p0161 -->
Recursos Computacionais: Como mencionado, dispomos de infraestrutura suficiente para conduzir os experimentos. O uso do Google Colab Pro garante acesso a GPUs potentes por longos períodos a um custo baixo (cerca de US$10 mensais), o que é viável dentro do orçamento estudantil. Adicionalmente, o laboratório de IA da instituição oferece uma estação com GPU dedicada que pode ser utilizada para treinos ou inferências locais. O armazenamento necessário (até ~50 GB considerando imagens e índices) pode ser alocado em HDs externos ou no próprio Google Drive vinculado ao Colab. Não haverá necessidade de aquisição de hardware extra. Em caso de necessidade de processamento massivo eventual (por exemplo, indexar milhões de vetores para testar escalabilidade extrapolada), poderíamos utilizar créditos em nuvens

<!-- p0162 -->
acadêmicas ou reduzir a amostra – de toda forma, o escopo definido (100k imagens) está bem dentro do que as ferramentas disponíveis suportam.

<!-- p0164 -->
Tecnologias e Ferramentas: O projeto apoia-se fortemente em ferramentas open-source e gratuitas, eliminando barreiras de licenciamento. Bibliotecas como PyTorch, OpenCV, FAISS, etc. são todas de uso livre. O ElasticSearch tem versão gratuita open-source que pode ser usada localmente. Assim, não há impedimentos de acesso às tecnologias de software. Além disso, muitas das tarefas (detecção facial, geração de embeddings, indexação ANN) possuem implementações maduras e documentadas na comunidade – por exemplo, a biblioteca facenet- pytorch ou insightface fornece modelos de face recognition de imediato; o faiss-gpu instala facilmente via pip. Isso acelera o desenvolvimento e garante confiabilidade, já que utilizaremos componentes testados em produção por grandes empresas (como Facebook/Meta no caso do FAISS ). Em resumo, do ponto de vista tecnológico, o projeto é altamente viável pois se alinha ao estado da arte disponível publicamente.

<!-- p0166 -->
Competências e Equipe: O aluno proponente possui familiaridade com programação Python e fundamentos de aprendizado de máquina, tendo cursado disciplinas de visão computacional e mineração de dados. O orientador (Prof. Gabriel da Silva Vieira) tem expertise comprovada na área, inclusive com publicações recentes relevantes , o que será valioso para direcionar as soluções. Caso surjam desafios específicos (por exemplo, dificuldades em ajustar o FAISS ou em melhorar a acurácia de detecção), a vasta literatura e fóruns online (StackOverflow, comunidades de IA) servirão de apoio. Além disso, planeja-se colaboração eventual com colegas de laboratório para brainstorming de ideias. O cronograma prevê bastante tempo de pesquisa inicial justamente para assegurar que as abordagens escolhidas sejam as mais adequadas e evitar retrabalho por escolhas errôneas.

<!-- p0168 -->
Riscos e Mitigações: Avaliamos que os principais riscos estão ligados à integração dos componentes e ao desempenho real obtido. Por exemplo, pode ocorrer de a fusão de resultados não trazer ganho substancial, ou de a latência ficar um pouco acima do desejado em determinado hardware. Esses riscos, contudo, não inviabilizam o projeto – no pior caso, o

<!-- p0169 -->
sistema funcionará com desempenho subótimo, mas ainda funcionará e poderá ser avaliado e melhorado. Adotamos um design modular justamente para isolar problemas: se o módulo X falhar, podemos substituí-lo ou ajustá-lo sem refazer tudo. A flexibilidade no uso do ElasticSearch é outro fator de segurança: se o FAISS apresentar limitações, o Elastic (com HNSW interno) pode suprir e vice-versa. Quanto a prazos, a elaboração antecipada do protótipo básico (busca facial isolada e CBIR isolado) até metade do projeto garante que o resto do tempo sirva para aprimoramentos e avaliações, diminuindo a chance de não entrega.

<!-- p0171 -->
Em conclusão, a execução do projeto é perfeitamente viável dentro das condições atuais. Os recursos de computação estão assegurados com alternativas (local e cloud), as ferramentas algorítmicas estão ao nosso alcance gratuitamente, e a equipe detém (ou está apta a adquirir rapidamente) o conhecimento necessário para lidar com os desafios técnicos. O retorno esperado – um sistema inovador e funcional – justifica plenamente empreender o esforço, e as probabilidades de sucesso são altas dado o planejamento cuidadoso e apoio institucional disponíveis.

<!-- p0173 -->
Referências

<!-- p0174 -->
AHMED, A. S.; IBRAHEEM, I. N. Recent advances in content-based image retrieval using deep learning techniques: A survey. In: AIP Conference Proceedings, v. 3219, n. 1, 2024. Disponível em: .

<!-- p0176 -->
BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais. Diário Oficial da União, Brasília, DF, 15 ago. 2018.

<!-- p0178 -->
DENG, J. et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), p. 4690–4699, 2019. Disponível em: .

<!-- p0180 -->
LACAILLE, M. Rise Above Research estimates that 1.6 trillion photos will be taken in 2023. Mediaclip Blog, 25 jan. 2023. Disponível em: . Acesso em: 07 maio 2025.

<!-- p0182 -->
ONDAS, R.; SUHM, B. Overview of image similarity search in Elasticsearch. Elastic Blog, 28 fev. 2023. Disponível em: . Acesso em: 07 maio 2025.

<!-- p0184 -->
SRIVASTAVA, D. et al. Content-based image retrieval: a survey on local and global features selection, extraction, representation, and evaluation parameters. IEEE Access, v. 11, p. 95410–95431, 2023.

<!-- p0186 -->
VIEIRA, G. S.; FONSECA, A. U.; SOARES, F. CBIR-ANR: A content-based image retrieval with accuracy noise reduction. Software Impacts, v. 15, p. 100486, 2023. DOI: 10.1016/j.simpa.2023.100486.

<!-- p0188 -->
WANG, X. et al. A Survey of Face Recognition. arXiv preprint arXiv:2212.13038, 2023 .

<!-- p0189 -->
1.6 Trillion Photos Expected in 2023, says Rise Above Research

<!-- p0190 -->
https://www.mediaclip.ca/en/blog/rise-above-research-estimates-that-1-6-trillion-photos-will-be-taken-in-2023/2023/

<!-- p0192 -->
Content-based Image Retrieval: Innovations in Feature Extraction and User-centric Design | Journal of Advancement in Electronics Signal Processing https://matjournals.net/engineering/index.php/JoAESP/article/view/1387

<!-- p0193 -->
Projeto_Artur_Duarte.docx

<!-- p0194 -->
file://file-FmyLjV6V7j3jwMyFT7meGt

<!-- p0196 -->
arxiv.org

<!-- p0197 -->
https://arxiv.org/pdf/2401.08281

<!-- p0199 -->
Overview of image similarity search | Elastic.co | Elastic Blog

<!-- p0200 -->
https://www.elastic.co/blog/overview-image-similarity-search-in-elastic

<!-- p0202 -->
A novel content-based image retrieval system with feature descriptor ...

<!-- p0203 -->
https://www.sciencedirect.com/science/article/abs/pii/S0957417423012769

<!-- p0205 -->
[2312.10089] Advancements in Content-Based Image Retrieval: A Comprehensive Survey of Relevance Feedback Techniques

<!-- p0206 -->
https://arxiv.org/abs/2312.10089

<!-- p0208 -->
ArcFace: Additive Angular Margin Loss for Deep Face Recognition https://openaccess.thecvf.com/content_CVPR_2019/papers/ Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf

<!-- p0209 -->
[2212.13038] A Survey of Face Recognition

<!-- p0210 -->
https://arxiv.org/abs/2212.13038

<!-- p0212 -->
Large scale similarity search across digital reconstructions of neural ...

<!-- p0213 -->
https://pmc.ncbi.nlm.nih.gov/articles/PMC9960175/

<!-- p0215 -->
FAISS (Facebook AI Similarity Search) | Continuum Labs

<!-- p0216 -->
https://training.continuumlabs.ai/knowledge/vector-databases/faiss-facebook-ai-similarity-search
