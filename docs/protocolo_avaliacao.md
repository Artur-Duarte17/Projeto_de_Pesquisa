# Protocolo vigente de avaliação

Data de referência: **24 de setembro de 2026**.

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

## 9. Validação da versão final

A aprovação técnica exigirá três camadas:

1. **Testes automatizados:** regras isoladas, sem inferência pesada.
2. **Validação científica completa:** recriação dos índices e avaliações LFW e Holidays, seguida da criação dos índices Gallagher, protocolo corrigido e avaliação pareada a partir das entradas atuais.
3. **Teste privado de aceitação:** uso da aplicação com uma pasta externa de fotografias familiares, sem incorporar essas imagens à evidência científica.

O autor executará os comandos. A revisão confirmará códigos de saída, contagens, hashes, manifestos e consistência com o protocolo.

## 10. Teste privado com álbum familiar

O álbum será indicado por caminho externo. Os indexadores percorrerão recursivamente formatos de imagem suportados e produzirão índices em `outputs/collections/`.

A aplicação permite que o usuário escolha explicitamente a pessoa quando a consulta possui múltiplos rostos. Usar automaticamente o maior rosto não é aceitável para esse cenário.

As fotografias, recortes e embeddings não entram no Git nem no artigo. A fotografia consultada deve ser excluída dos resultados por caminho ou SHA-256. Falhas serão registradas, não substituídas silenciosamente.

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
