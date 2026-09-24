# Arquitetura de software da versão ativa

## O que foi adotado

O projeto é um **monólito modular**: interface, comandos e avaliações usam módulos Python do mesmo repositório e executam localmente, sem serviços independentes. A organização interna segue uma **arquitetura hexagonal leve** (portas e adaptadores). As regras de seleção, pontuação e ranking ficam em funções compartilhadas; o acesso a fotografias, índices e modelos fica nas bordas. Dentro da indexação e da avaliação, há **pipelines** com etapas em ordem definida pelo protocolo.

Uma *porta* é a operação que uma parte do sistema oferece ou espera, com entradas e saídas conhecidas. Um *adaptador* liga essa operação a uma tecnologia concreta. Por exemplo, a busca recebe descritores e metadados; o adaptador facial usa InsightFace para produzir um descritor, e o adaptador de arquivos encontra as fotografias a excluir da consulta. Não é necessário criar uma classe ou interface para cada função: nesta implementação, funções públicas e formatos de dados cumprem esses contratos.

Fluxo simplificado:

```text
Streamlit / comandos / avaliações
               |
               v
retrieval/adapters (modelos, arquivos, índices, busca e manifestos)
               |
               v
retrieval/application (rankings)
               |
               v
retrieval/domain (seleção, similaridade, fusão e métricas)
```

Esse desenho indica responsabilidades, não processos separados nem uma exigência de que toda chamada atravesse todas as caixas. A interface continua usando as mesmas funções de busca que os comandos e os protocolos de avaliação.

## Como a escolha foi feita

O critério partiu do código e das tarefas do projeto: há mais de uma entrada (Streamlit, comandos, testes e avaliações), duas formas de extrair descritores (face e imagem inteira), índices em arquivos e regras científicas que precisam continuar reproduzíveis. Foram comparados padrões de interface, de organização interna, de processamento em etapas e de implantação. A [descrição original de Alistair Cockburn](https://alistair.cockburn.us/hexagonal-architecture) foi a referência decisiva: ela trata explicitamente da mesma aplicação acionada por interface, programas, testes ou rotinas em lote, com tecnologias externas conectadas por adaptadores. A decisão de manter tudo em um monólito modular é uma inferência sobre as necessidades deste repositório; o padrão hexagonal não exige uma implantação específica.

| Abordagem pesquisada | Por que ela não foi escolhida como estrutura principal aqui |
|---|---|
| [Camadas tradicionais / N-tier](https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/n-tier) | Separa apresentação, lógica e dados de modo simples, mas descreve menos diretamente as várias entradas e saídas deste projeto. **Camada lógica** e *tier* físico são coisas distintas; não precisamos distribuir o sistema em máquinas. |
| [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) | Compartilha a ideia de dependências voltadas para as regras centrais. Seus círculos de entidades, casos de uso e interfaces poderiam organizar o projeto; portas e adaptadores dão um vocabulário mais direto às fronteiras que já existem. |
| [Onion Architecture](https://jeffreypalermo.com/2008/07/the-onion-architecture-part-1/) | Também mantém infraestrutura fora do núcleo e dependências voltadas para dentro. Foi descrita para aplicações duradouras com comportamento complexo; suas camadas de modelo de domínio acrescentariam estrutura sem uma necessidade clara nesta etapa. |
| [Pipes and Filters](https://learn.microsoft.com/en-us/azure/architecture/patterns/pipes-and-filters) | É útil para descrever operações em sequência, como ler, detectar, extrair e indexar imagens. Não organiza por si só a interação entre interface, busca e armazenamento. A ordem de etapas científicas permanece fixa onde o protocolo exige. |
| [MVVM](https://learn.microsoft.com/en-us/windows/uwp/data-binding/data-binding-and-mvvm) | Separa apresentação e lógica da interface, sobretudo em plataformas com *data binding*. Ajuda a pensar na tela, mas não cobre os comandos, índices e protocolos; [Streamlit reexecuta o script a cada interação](https://docs.streamlit.io/develop/concepts/architecture/session-state), o que pede cuidado com estado da interface. |
| [Microsserviços](https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/microservices) | Oferecem implantação e escala independentes. Aqui não há requisito demonstrado para isso; comunicação pela rede, consistência e operação de vários serviços aumentariam o custo do experimento. |

Esses padrões não são mutuamente exclusivos. O próprio autor da Clean Architecture aponta a semelhança entre Clean, Onion e Hexagonal. Usamos a arquitetura hexagonal para nomear as fronteiras do sistema e a ideia de pipeline apenas onde existe processamento sequencial.

## Onde cada responsabilidade está

| Local | Responsabilidade atual |
|---|---|
| `scripts/retrieval/domain/similarity.py` | Normalização de vetores e cálculo de similaridade por cosseno. |
| `scripts/retrieval/domain/face_policy.py` | Ordem e escolha de rostos, inclusive seleção pelo ponto anotado, extração do descritor do rosto e formato da caixa facial. |
| `scripts/retrieval/domain/fusion.py` | Conversão e combinação de escores, classificação de variação de métricas e união dos resultados facial e global. |
| `scripts/retrieval/domain/metrics.py` | Relevância, exclusões e métricas de recuperação, incluindo precisão, revocação, AP e mAP. |
| `scripts/retrieval/application/search.py` | Montagem dos rankings facial, global e combinado a partir de descritores, índices, metadados e IDs já excluídos. |
| `scripts/retrieval/adapters/face_model.py` e `global_model.py` | Uso concreto de InsightFace, PyTorch e ResNet50 para detectar rostos e extrair descritores. |
| `scripts/retrieval/adapters/files.py` e `index_store.py` | Arquivos de imagem, inventários, caminhos, hashes, exclusão da imagem consultada, carregamento dos índices `.npy`/`.csv` e visualização de resultados. |
| `scripts/retrieval/adapters/search_gateway.py` | Resolve exclusões por caminho/hash para cada índice e chama os casos de uso de ranking. É a entrada comum das buscas na interface e nos comandos. |
| `scripts/retrieval/adapters/manifest.py` | Proveniência da execução, estado do Git e hashes dos artefatos. |
| `scripts/retrieval/adapters/run_paths.py` | Valida o destino isolado da reexecução científica para preservar a referência anterior. |

Os pontos de entrada continuam nos locais conhecidos: `scripts/app/streamlit_app.py` para a interface; `scripts/face/`, `scripts/global/` e `scripts/fusion/` para indexação, busca e avaliações; `scripts/prepare_collection.py` para uma coleção própria; e `scripts/run_final_validation.py` e `scripts/run_final_gallagher.py` para orquestrar a reprodução científica. Os arquivos em `scripts/data/` baixam as bases, sem integrar a execução final. Essas pastas de comandos não duplicam o núcleo em `scripts/retrieval/`. Os cinco arquivos de compatibilidade que antes ficavam diretamente em `scripts/` foram retirados após a migração das importações.

Na prática, uma busca recebe uma foto na interface ou em um comando; o adaptador de modelo extrai seu descritor; o adaptador de arquivos identifica a própria foto e cópias a excluir; a aplicação calcula o ranking; a fusão pode combinar os dois rankings; o ponto de entrada apresenta ou grava o resultado. Essa divisão evita que a regra de ranking dependa diretamente de `cv2.imread`, de um modelo específico ou do formato do caminho da consulta.

## Fronteiras científicas e limites

Os protocolos **LFW, INRIA Holidays e Gallagher** preservam suas regras específicas nos scripts de preparação e avaliação em `scripts/face/`, `scripts/global/` e `scripts/fusion/`. A separação arquitetural compartilha cálculos e operações gerais, mas não transforma os três protocolos em um protocolo genérico. Isso reduz o risco de alterar sem querer as consultas, a relevância, as exclusões, a ordem do processamento ou as métricas publicadas. Uma mudança futura nessas regras exige justificativa metodológica e nova comparação com as evidências anteriores.

Esta é uma aplicação **leve e parcial** do padrão. O domínio e a aplicação ainda usam `numpy` e `pandas`; a indexação e a avaliação ainda têm orquestração nos pontos de entrada; não há uma interface formal para todo adaptador. Essa escolha mantém contratos concretos e evita abstrações que o projeto ainda não precisa.

Artur executou a suíte automatizada da estrutura atual após a adição de `--run-root` e a remoção das APIs de compatibilidade: **37 testes passaram em 24/09/2026**. Esses testes verificam as regras e os contratos cobertos, mas **não demonstram equivalência de resultados** com a execução científica anterior. Antes de usar novos números no artigo, ainda será necessária uma execução limpa e a comparação dos manifestos, métricas e hashes pertinentes. A verificação com álbum privado avalia a utilidade da aplicação para uma pessoa; ela responde a uma pergunta diferente da reprodução das métricas acadêmicas.
