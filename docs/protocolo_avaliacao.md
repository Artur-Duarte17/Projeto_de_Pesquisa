# Protocolo vigente de avaliação

Data de referência: **28 de setembro de 2026**.

## 1. Pergunta científica

Em coleções fotográficas com múltiplas pessoas, acrescentar contexto visual global por fusão tardia melhora a recuperação baseada em identidade facial?

Gallagher é a base principal porque contém fotografias com múltiplos rostos e anotações de identidade. LFW e INRIA Holidays verificam separadamente os componentes facial e global.

## 2. Unidade recuperada

O sistema retorna fotografias completas. O índice facial pode conter vários rostos para a mesma fotografia; durante a busca, somente o maior score facial de cada fotografia é mantido.

## 3. Métodos comparados

As configurações Gallagher são predefinidas:

- face/contexto `1,0/0,0`;
- face/contexto `0,0/1,0`;
- face/contexto `0,9/0,1`;
- face/contexto `0,7/0,3`;
- face/contexto `0,5/0,5`.

Os cossenos são convertidos para `[0,1]` antes da soma ponderada. Uma fotografia sem rosto detectado recebe contribuição facial zero, mas continua elegível pelo componente global.

Esse remapeamento não é calibração estatística nem probabilidade. O descritor global representa a imagem inteira, incluindo pessoas e objetos, não somente seu fundo. Os pesos são fixos nesta comparação, sem escolha otimizada em um conjunto independente.

## 4. Consultas Gallagher

O protocolo possui 20 identidades escolhidas por frequência decrescente e uma consulta por identidade. A seleção dentro da identidade é determinística por nome da imagem e índice anotado da face.

O recorte de consulta é ligado à pessoa-alvo pela caixa detectada que contém o ponto médio dos olhos anotados. Quando mais de uma caixa contém o ponto, o desempate usa distância ao centro, menor área e coordenadas. Se nenhuma caixa contiver o ponto, a consulta falha explicitamente; não é permitido escolher o maior rosto ou um vizinho.

## 5. Gabarito de relevância

A relevância é construída pelas anotações oficiais cruzadas com as 589 fotografias da galeria global. Ela não depende de a face ter sido detectada no índice facial.

A fotografia-fonte da consulta é removida tanto do ranking quanto do conjunto relevante. Restam 588 candidatas por consulta e 884 relações relevantes no conjunto completo.

## 6. Métricas

- **Precision@K:** fração das primeiras K posições que é relevante, sempre usando K como denominador.
- **Recall@K:** fração das fotografias relevantes recuperada nas primeiras K posições.
- **Average Precision:** média das precisões nas posições relevantes do ranking integral.
- **mAP:** média da Average Precision entre consultas.

O Top-10 salvo serve para inspeção; AP e mAP usam todas as candidatas elegíveis.

## 7. Integridade da execução

Cada execução final deve registrar:

- commit Git e indicação de árvore limpa ou suja;
- versões de Python e dependências críticas;
- dispositivo solicitado e provedor efetivamente usado;
- configuração completa;
- hashes de entradas e saídas;
- quantidade de imagens, rostos, falhas e consultas;
- arquivos de métricas agregadas e por consulta.

Saídas existentes não podem ser sobrescritas sem `--overwrite`. A execução final deve usar diretórios novos e identificados.

## 8. Evidência vigente

- EX-033: produz consultas, recortes e relevância corrigidos.
- EX-034: emparelha o recorte facial com a fotografia-fonte completa.
- EX-035: avalia as cinco configurações no mesmo universo de candidatas.

Os manifests e métricas seguros estão em `docs/evidencias/gallagher_a01_a02_20260922_v2/`.

Esse pacote documenta a correção de 22/09, como referência preservada. Os números atuais vêm de `outputs/validation_runs/fechamento_20260928_cuda/`, commit limpo `91a2a28`, após ativação efetiva de CUDA nos modelos faciais. O registro canônico é `docs/resultados_experimentais_congelados.md`; não misturar as execuções.

## 9. Validação da versão final

A aprovação técnica exigirá três camadas:

1. **Testes automatizados:** regras isoladas, sem inferência pesada.
2. **Validação científica completa:** recriação dos índices e avaliações LFW e Holidays, seguida da criação dos índices Gallagher, protocolo corrigido e avaliação pareada a partir das entradas atuais.
3. **Teste privado de aceitação:** uso da aplicação com uma pasta externa de fotografias familiares, sem incorporar essas imagens à evidência científica.

As três camadas foram executadas em 28/09 pelo assistente, com autorização de Artur: 93 testes aprovados, reprodução científica completa e aceitação privada aprovadas. A conferência verificou códigos de saída, contagens, hashes, manifestos e consistência com o protocolo. A validação científica final do texto pelos autores humanos não é substituída por essa execução assistida.

## 10. Teste privado com álbum familiar

O álbum será indicado por caminho externo. Os indexadores percorrerão recursivamente formatos de imagem suportados e produzirão índices em `outputs/collections/`.

A aplicação permite que o usuário escolha explicitamente a pessoa quando a consulta possui múltiplos rostos. Usar automaticamente o maior rosto não é aceitável para esse cenário.

As fotografias, recortes e embeddings não entram no Git nem no artigo. A fotografia consultada deve ser excluída dos resultados por caminho ou SHA-256. Falhas serão registradas, não substituídas silenciosamente.

O teste realizado utilizou 171 imagens (59 HEIC e 112 JPEG), com escolha da pessoa, três modos de busca, paginação, atualização no mesmo destino e remoção do álbum exclusivo de validação. Os 207 arquivos originais e os dois álbuns existentes ficaram intactos. O relatório de aceitação não é um gabarito de identidade familiar nem estima sua acurácia.

## 11. Critérios de saída

A versão estará pronta para a redação final somente quando:

- os scripts ativos estiverem separados do histórico;
- o lock de dependências estiver sincronizado;
- todos os testes automatizados ativos passarem;
- a validação LFW, Holidays e Gallagher terminar sem falhas de integridade;
- as métricas novas forem explicadas e comparadas às vigentes;
- a seleção de rosto da aplicação funcionar em consulta multi-rosto;
- o teste privado de aceitação demonstrar o fluxo completo de uso;
- README e documentos canônicos corresponderem ao comportamento real.

Esses critérios técnicos foram atendidos no fechamento de 28/09. Veja `docs/fechamento_tecnico_2026.md` para cobertura e limites. Continuam separados os critérios humanos, éticos, institucionais e editoriais para submissão.

## 12. Limites dos protocolos auxiliares

LFW usa recuperação customizada condicionada à detecção e não o protocolo oficial de verificação em pares. Não há garantia de ausência de sobreposição com o pré-treinamento do modelo.

Holidays permanece auxiliar. A AP local é a média das precisões nas posições relevantes, dividida pelo total de relevantes; o avaliador oficial integra trapézios. Não se apresentam os dois valores como equivalentes. A cópia local tem três pares byte a byte idênticos em grupos diferentes. O avaliador exclui a fonte por ID/caminho, enquanto a busca interativa exclui também cópias por SHA-256: não há alegação de equivalência universal entre essas rotas.

As ordenações auxiliares são vetorizadas pelo PyTorch; não se promete desempate idêntico em todo equipamento. A regra explícita score decrescente/ID crescente descrita para Gallagher não deve ser estendida automaticamente a esses avaliadores.

## 13. Limites da conclusão principal

As vinte consultas representam identidades selecionadas por frequência em um único álbum, com dezenove fotos-fonte. Duas consultas compartilham uma foto, e todas compartilham galeria e contexto. A população-alvo é a identidade anotada segundo os critérios originais, não qualquer pessoa visível.

A análise pareada é descritiva, completa para essas consultas e sem alegação de independência, significância estatística ou generalização populacional. O estudo não identifica a causa da diferença entre métodos, não calibra pesos e não mede larga escala. A interface é demonstração/exploração; seus filtros e modos de consulta não substituem o protocolo controlado.
