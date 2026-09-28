# Estado atual do projeto

Data de referência: **28 de setembro de 2026**.

## 1. O que é este projeto

É uma **pesquisa experimental com uma aplicação local de demonstração**.
A aplicação mostra que o método pode ser usado; a pesquisa mede e compara
os métodos sob regras controladas. Não é um serviço publicado na internet,
um produto comercial pronto ou um artigo já aceito.

O fechamento técnico e documental está concluído no escopo testado. A etapa
seguinte é escrever e validar o artigo em inglês, com as decisões institucionais
e de uso dos dados separadas da aprovação dos testes.

## 2. O que a aplicação faz

1. Prepara os índices a partir da pasta de fotografias e suas subpastas, sem
   copiar nem modificar os originais.
2. Recebe uma foto de referência e permite escolher qual pessoa procurar.
3. Retorna fotografias completas por semelhança facial ou visual.
4. Permite atualizar o mesmo álbum, navegar pelos resultados e remover seus
   índices mediante confirmação, sem apagar as fotografias.
5. Oferece um perfil de Pesquisa, ativado explicitamente, com fusão experimental
   e índices científicos/personalizados.

O padrão é Uso pessoal, sem Gallagher automático. HEIC/HEIF são suportados;
JPEGs excepcionalmente grandes usam leitura global reduzida e controlada.
Não existe teto de 50 resultados: a galeria tem 24 fotos por página.
Existem limites reais de memória, tempo e dimensões de imagem.

O sistema usa modelos já treinados. Não aprende novas classes de pessoas nem
treina uma rede neste projeto: detecta rostos, extrai descritores e compara vetores.

## 3. Pergunta e resultado científico

> Em coleções com múltiplas pessoas, acrescentar contexto visual global por
> fusão tardia melhora a recuperação baseada em identidade facial?

Gallagher é a avaliação principal. LFW verifica o componente facial em um
protocolo customizado; Holidays verifica o componente global com AP adaptada.
As métricas dessas tarefas não são diretamente comparáveis.

Na execução atual, a busca facial obteve mAP **0,952261**. As três fusões
obtiveram 0,946274, 0,901460 e 0,813297. Nenhuma superou a média facial.
A fusão 0,9/0,1 melhorou duas consultas, empatou nove e piorou nove:
o resultado não significa que contexto nunca possa ajudar.

A conclusão vale somente para as vinte consultas do protocolo, não para
qualquer álbum, todas as pessoas, o domínio agropecuário ou larga escala.
Os valores e todas as comparações estão em [resultados experimentais](resultados_experimentais_congelados.md).

## 4. O que a auditoria encontrou e o que foi resolvido

A auditoria anterior encontrou fotos relevantes retiradas do gabarito por
falha de detecção e uma consulta usando o rosto vizinho. As correções A01/A02
separaram relevância de detecção e vincularam o rosto ao ponto dos olhos anotados.
Ambas foram novamente verificadas na reprodução atual.

A aceitação desta rodada encontrou outro defeito: selecionar CUDA não ativava
a GPU nos modelos faciais. O adaptador foi corrigido e agora confirma os
provedores reais. Isso alterou alguns números; os antigos ficaram preservados
como referência, sem serem misturados aos atuais.

As diferenças de AP Holidays, duplicatas, dependência entre consultas,
limites do gabarito e ausência de calibração da fusão estão documentadas,
não escondidas nem transformadas em alegações de generalização.

## 5. O que realmente foi executado

Com autorização de Artur, o assistente executou:

- 93 testes automatizados, todos aprovados.
- Preparação, busca, seleção da pessoa, paginação, atualização e remoção de
  um álbum exclusivo de validação, com modelos e fotos reais.
- Aceitação com 171 imagens familiares, incluindo 59 HEIC e um JPEG de
  aproximadamente 200 megapixels, sem falhas de leitura.
- Os três comandos de busca com os índices científicos novos.
- Reprodução completa LFW/Holidays/Gallagher em destino novo, no commit limpo
  `91a2a28360ded41fd86abf3371c989d571bf95bd`.
- Conferência dos 13 manifestos, hashes, contagens, médias e resultados por consulta.

Os 207 arquivos originais da coleção e os dois álbuns existentes ficaram intactos.
Somente os índices dos álbuns exclusivos de teste foram removidos; são regeneráveis.
As fotos familiares não entraram no Git nem nas tabelas científicas.
O teste não mediu acurácia de identidade familiar.

A execução válida está em `outputs/validation_runs/fechamento_20260928_cuda/`.
As referências anteriores continuam preservadas. O [fechamento completo](fechamento_tecnico_2026.md)
explica verificações, tentativas interrompidas e limites de cobertura.

## 6. Organização e pendências

O código compartilhado mantém a [arquitetura hexagonal leve](arquitetura_software.md),
com apresentação, casos de uso, regras de recuperação e adaptadores separados.
Não houve outra mudança de arquitetura. Materiais substituídos permanecem em
`C:/Projeto_de_Pesquisa_arquivo_local/2026-09-24_pre_finalizacao/`.

Não foi comprovada ausência de qualquer defeito ou funcionamento em toda máquina.
Não houve benchmark de carga, nova instalação do zero, avaliação demográfica
ou aprovação jurídica/ética. Dados e modelos continuam com restrições próprias,
descritas em [dados, modelos e privacidade](dados_modelos_privacidade.md).

Antes da submissão, continuam necessários: redação e validação humana do artigo;
confirmações de TC, aceite, entrega e calendário; decisão institucional sobre
dados/biometria; aprovação da autoria; e conformidade com a revista e o depósito
institucional. Não houve push, contato externo, publicação ou submissão.
