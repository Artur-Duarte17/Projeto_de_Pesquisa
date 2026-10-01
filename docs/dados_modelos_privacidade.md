# Dados, modelos, privacidade e uso de IA

Registro técnico de 28/09/2026. Este documento não é um parecer jurídico,
aprovação ética ou autorização de publicação. Fontes e restrições devem ser
revistas pelos autores antes de qualquer disponibilização pública.

## O que o projeto faz com as imagens

O código ativo usa modelos previamente treinados. Detecta rostos, extrai
descritores numéricos, constrói índices e compara rankings. Não treina uma nova
rede, não ajusta seus pesos e não realiza nova coleta de imagens neste fechamento.
Portanto, chamar todo esse processamento de "apenas treino" não descreve o uso real.

Fotografias, recortes, descritores faciais e rankings que permitem vincular pessoas
permanecem locais e ignorados pelo Git. Não se presume anonimização apenas
porque os nomes foram omitidos. Os índices dependem das pastas originais:
preparar um álbum não faz cópia de segurança das fotografias.

## Origem e condições documentadas

| Recurso | Origem e papel | Condição e limite de evidência |
|---|---|---|
| Gallagher | [Página do responsável](http://chenlab.ece.cornell.edu/people/Andy/GallagherDataset.html); URLs e anotações locais. Comparação científica principal. | A auditoria de 21/09 registrou pesquisa não comercial e proibição de redistribuição. A página não respondeu à revalidação de 28/09; não há confirmação nova nem autorização de exibição de pessoas. |
| LFW | [Projeto original UMass](http://vis-www.cs.umass.edu/lfw/); coleção local já existente. Verificação auxiliar facial. | Página original indisponível na consulta de 28/09. Não declarar licença aberta irrestrita nem usar o número local como acurácia oficial LFW. Os termos precisam de registro institucional/editorial adequado antes da publicação. |
| INRIA Holidays | [Página oficial INRIA](https://thoth.inrialpes.fr/~jegou/data.php.html), verificada em 28/09; origem dos arquivos locais deve acompanhar o manifesto. Verificação auxiliar global. | A página atribui copyright ao INRIA e pede citação. Isso não é licença irrestrita de redistribuição. A métrica local é adaptada; não comparar diretamente ao avaliador oficial. |
| Álbum familiar | Pasta externa indicada por Artur. Teste privado de aceitação da aplicação. | Sem criação de dataset público, uso nas conclusões científicas, figuras no artigo ou arquivos no Git. Há verificação de integridade de originais, não autorização individual de publicação. |
| Agrishow / WIDER FACE | Material histórico, fora da versão ativa e do artigo definido. | Não recolocar automaticamente no projeto nem apresentar como evidência desta finalização. |

Não foram baixadas novas bases, coleções de imagens ou checkpoints durante este
fechamento. Os testes reais utilizam apenas coleções e pesos previamente existentes.
Falha de acesso a uma página não demonstra autorização nem proibição: demonstra
somente uma lacuna de verificação, explicitamente registrada aqui.

## Código e pesos são recursos diferentes

A [política oficial InsightFace](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/README.md),
consultada em 28/09/2026, distingue o código MIT dos modelos pré-treinados destinados
a pesquisa não comercial. O [README principal](https://raw.githubusercontent.com/deepinsight/insightface/master/README.md)
também menciona licenciamento separado para o pacote `buffalo_l`.

O perfil **Uso pessoal** é uma simplificação da interface deste protótipo acadêmico;
não modifica a licença dos modelos nem autoriza um produto comercial ou uso geral
irrestrito. Nenhum peso será incluído em um pacote público deste projeto sem
permissão específica. O projeto continua fixado em InsightFace 0.2.1: consultar a
documentação atual não significa adotar a implementação mais recente.

O extrator global utiliza `ResNet50_Weights.DEFAULT` do torchvision 0.26.0,
com o checkpoint ImageNet já existente. A [documentação do fornecedor](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
identifica as configurações de pesos; licença do código, condições dos pesos e
direitos da base de pré-treinamento não devem ser confundidos. As impressões
digitais dos arquivos utilizados são registradas nos novos manifestos de indexação.

## Fronteira público/privado

Um eventual pacote público poderá conter código, configuração, protocolo,
instruções e métricas agregadas previamente revisadas. Não deverá conter fotos,
recortes, embeddings, pesos, credenciais, caminhos pessoais ou rankings vinculáveis.
Arquivos presentes apenas em `outputs/` não estão publicamente disponíveis.

Esta rodada não faz push, submissão, publicação, contato externo nem promessa
de disponibilização. Questões de apreciação/dispensa ética e base de tratamento
dos dados continuam sendo decisões institucionais; o teste computacional não as resolve.

## Uso de IA no fechamento técnico anterior

OpenAI Codex auxiliou a inspeção e modificação do código, organizou documentos,
executou testes locais e a reprodução computacional autorizada, verificou
artefatos e preparou explicações e rascunhos de documentação.

Resultados numéricos devem vir dos arquivos efetivamente produzidos, não de
texto gerado. O assistente não é autor nem autoridade de aprovação científica,
ética ou institucional. A interpretação e a versão final do artigo ainda precisam
de revisão e aprovação pelos autores humanos. Não afirmar antecipadamente que
essa revisão já ocorreu. Uma futura declaração no artigo deve refletir esse uso
real, inclusive a execução assistida, e a política da revista escolhida.

## Rodada documental da monografia — 28/09/2026

Na preparação em `docs/monografia`, OpenAI Codex auxiliou a redação dos seis
capítulos, organização das referências, pesquisa bibliográfica delimitada,
leitura pontual de fontes e código, conferência de métricas e produção de
tabelas, gráficos, diagrama e auxiliares documentais. Os gráficos foram derivados
de resultados já existentes. Não houve nova inferência, execução dos 93 testes
ou alteração do código científico nesta rodada. O uso autorizado de recursos
LaTeX se limita à construção documental e não muda o ambiente experimental.

O material não inclui fotos familiares ou de terceiros, recortes, embeddings,
pesos, credenciais ou rankings vinculáveis. A distribuição de deltas é exibida
sem identificadores de consulta; isso não constitui prova de anonimização.
A monografia também depende de revisão humana e de elegibilidade para depósito.
Artur é o autor; o assistente não é autor, e a aprovação de Artur ou do orientador
não está presumida. O uso de IA deve ser declarado conforme a regra institucional
aplicável, ainda pendente. O artigo permanece uma entrega futura.
