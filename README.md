# Recuperação de fotografias por identidade facial e contexto visual

Sistema experimental para localizar fotografias de uma pessoa em coleções com múltiplos rostos. A versão ativa combina:

- detecção e descritores faciais com InsightFace/ArcFace;
- descritores globais de imagem com ResNet50 pré-treinada no ImageNet;
- busca facial, busca visual global e fusão tardia de escores;
- métricas de recuperação e manifestos de execução;
- interface local em Streamlit.

Antes de usar resultados ou executar experimentos, leia [LEIA_PRIMEIRO.md](LEIA_PRIMEIRO.md).

## Estado da versão

O protocolo Gallagher corrigido foi reproduzido em 28/09/2026 com CUDA efetivamente ativado nos modelos faciais. Entre as cinco configurações avaliadas, a busca somente facial obteve o maior mAP (`0,952261`). Nenhuma fusão testada superou esse baseline.

O fechamento técnico está concluído no escopo testado: 93 testes aprovados, aceitação integrada com modelos e fotos reais e reprodução científica completa no commit limpo `91a2a28`. A execução foi assistida e autorizada por Artur. Veja [o fechamento e seus limites](docs/fechamento_tecnico_2026.md). Isso não equivale à aprovação ética, institucional ou editorial para publicação.

## Estrutura ativa

- `scripts/retrieval/`: regras compartilhadas, busca e adaptadores; veja [arquitetura de software](docs/arquitetura_software.md).
- `scripts/face/`: indexação, busca, avaliação facial e protocolo Gallagher.
- `scripts/global/`: indexação, busca e avaliação por imagem inteira.
- `scripts/fusion/`: busca combinada e avaliação pareada Gallagher.
- `scripts/data/`: aquisição das bases Gallagher e INRIA Holidays.
- `scripts/app/`: interface local em Streamlit.
- `tests/`: testes metodológicos do sistema ativo.
- `requirements/`: dependências diretas e único lock do ambiente reproduzível.
- `docs/`: protocolo, resultados, revisão bibliográfica e evidências seguras.
- `data/`: entradas locais; fotografias e arquivos privados ficam fora do Git.
- `outputs/`: índices, métricas e manifestos gerados; fica fora do Git.

Código, documentos e experimentos encerrados são preservados fora desta árvore em `C:\Projeto_de_Pesquisa_arquivo_local`.
As saídas exploratórias anteriores à execução final também foram movidas para esse arquivo histórico; `outputs/final/` e `outputs/experiments/` continuam no projeto por conterem a referência e suas evidências.

## Ambiente de referência

O ambiente aprovado usa Python 3.10, PyTorch com CUDA e ONNX Runtime GPU.

```powershell
uv venv laboratorio/cibir_gpu --python 3.10
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe -r requirements/experiment-gpu.lock.txt
laboratorio/cibir_gpu/Scripts/python.exe scripts/validate_environment.py --require-cuda
```

O arquivo `requirements/experiment-gpu.in` contém as dependências diretas. O lock atual inclui 82 versões fixadas e duas rodas CUDA por URL (PyTorch e torchvision), todas conferidas com o ambiente em 28/09. `uv pip check` aprovou os 88 pacotes instalados; quatro pacotes antigos de aquisição permanecem no ambiente, mas não são dependências da versão ativa. Nada foi instalado ou removido neste fechamento. Os antigos cinco apontadores idênticos para esse lock foram removidos na organização anterior.

O adaptador facial configura explicitamente os provedores ONNX em detecção e reconhecimento. CUDA indisponível ou não ativado gera erro com orientação para escolher CPU; não há mais execução facial silenciosamente rotulada como CUDA. Os modelos continuam os mesmos, sem treinamento adicional.

## Testes automatizados

```powershell
laboratorio/cibir_gpu/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Os testes verificam regras de ranking, exclusão da imagem-fonte, métricas, seleção da pessoa-alvo e integridade do protocolo. Eles não substituem a execução com modelos e fotografias reais.

Em 28/09/2026, **93 testes passaram**, incluindo arquitetura, ranking, protocolo Gallagher, álbuns, estado da interface, HEIC, JPEGs grandes e seleção/proveniência dos provedores. A execução com modelos e dados reais foi realizada separadamente. Os três comandos de busca também foram executados com os índices novos, retornando dez fotografias distintas e excluindo a fonte.

## Reprodução científica final

O assistente executou o comando abaixo em 28/09/2026, com autorização de Artur e árvore Git limpa no commit `91a2a28360ded41fd86abf3371c989d571bf95bd`:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/run_final_validation.py --run-root outputs/validation_runs/fechamento_20260928_cuda
```

O orquestrador valida o ambiente, reexecuta a avaliação facial LFW, a avaliação global INRIA Holidays e o protocolo Gallagher corrigido, e grava um manifesto que liga todas as saídas. A aquisição dos datasets não é refeita: os baixadores são utilitários de preparação, não etapas da reprodução final. O fluxo recusa por padrão uma árvore Git suja e saídas finais preexistentes.

As referências em `outputs/final/` e `outputs/validation_runs/arquitetura_v1/` permanecem intactas. Os 13 manifestos novos, 92 referências a arquivos e 15.265 imagens-fonte indexadas tiveram hashes conferidos. Os descritores globais foram idênticos à referência; os faciais e algumas métricas mudaram após a correção de CUDA. As diferenças e os resultados por consulta estão [documentados](docs/resultados_experimentais_congelados.md). Os rankings completos não são persistidos para comparação direta linha a linha. **Não repita o comando acima no mesmo destino**: escolha uma pasta nova sob `outputs/validation_runs/`.

O subfluxo Gallagher continua disponível isoladamente em `scripts/run_final_gallagher.py`, mas o comando acima é o critério de liberação da versão completa.

## Usar uma coleção própria

Não é necessário copiar as fotos para o projeto nem informar caminhos de índices
no uso normal. O álbum fica na pasta original; a aplicação prepara e guarda seus
índices locais automaticamente.

### 1. Abrir a aplicação

```powershell
laboratorio/cibir_gpu/Scripts/python.exe -m streamlit run scripts/app/streamlit_app.py
```

### 2. Preparar ou atualizar pela interface

1. Abra **Meus álbuns** e clique em **Adicionar álbum**; se ainda não houver álbuns, o formulário inicial já estará disponível.
2. Dê um nome e cole o endereço da pasta com as fotos.
3. Clique em **Preparar álbum** e aguarde o término. As subpastas também são lidas.
4. Em **Buscar fotos**, selecione o álbum, envie a foto de consulta, escolha a pessoa e clique em **Buscar fotos no álbum**.

Prepare somente na primeira vez e quando adicionar, remover ou alterar as fotos.
Para atualizar, selecione o álbum já cadastrado e clique em **Atualizar álbum**;
isso substitui os índices no mesmo destino, sem criar uma pasta `v2` e sem alterar
as fotografias. Depois da preparação, o cache dos índices é recarregado
automaticamente. Não feche a aplicação enquanto ela prepara o álbum.

Álbuns já preparados pelo terminal sob `outputs/collections/`, como `familia`,
aparecem automaticamente. Se uma preparação pela interface falhar, ela marca
esse álbum como incompleto e permite tentar novamente; índices incompletos não
entram na seleção normal de busca. Fotos originais movidas, excluídas ou limpas
por outro programa deixam de estar disponíveis: informe a pasta atual e prepare
novamente. A aplicação não oferece hospedagem ou cópia de segurança das fotos.

### 3. Remover um álbum

Abra **Meus álbuns**, selecione **Remover**, escolha o álbum, marque a confirmação e clique em
**Remover álbum**. A aplicação apaga somente os índices e registros gerados em
`outputs/collections/` para aquele álbum, nunca as fotos originais. A remoção
também está disponível para álbuns cuja preparação falhou. Para usar o álbum
removido novamente, prepare-o outra vez. Pastas externas, vínculos de arquivos
ou pastas e conteúdos não reconhecidos como gerados bloqueiam a remoção.

### Quantidade de fotos e resultados

A preparação pela interface não impõe um número máximo de fotos: analisa todas
as imagens compatíveis da pasta e de suas subpastas. O limite antigo de 50 era
de resultados exibidos por busca, não de tamanho do álbum, e foi removido.
**Mostrar todos os resultados** não corta o ranking solicitado; a galeria mostra
24 fotos por página para não carregar todas as prévias ao mesmo tempo. Na busca
facial, continuam valendo o filtro de semelhança e a exclusão da foto de consulta.
Também é possível desmarcar essa opção e informar um máximo de resultados,
inclusive acima de 50.

Essa mudança não comprova desempenho em larga escala. A busca atual carrega os
índices em memória e compara com o acervo inteiro; capacidade e tempo dependem
do computador e do tamanho da coleção. Não há teto artificial de quantidade
na interface, mas existem limites práticos de memória, armazenamento e tempo.

O perfil inicial é **Uso pessoal**, com busca nos álbuns cadastrados. Se não houver
álbum pronto, a aplicação oferece a preparação do primeiro, sem escolher Gallagher
ou outro acervo científico como alternativa automática. A navegação separa
**Buscar fotos** de **Meus álbuns**, onde há cartões para buscar, atualizar e remover.

**Pesquisa** é um perfil ativado explicitamente: permite escolher Gallagher,
índices personalizados, fusão experimental e detalhes técnicos. Mesmo nesse perfil,
a fonte inicial é **Meu álbum**, não Gallagher. Nenhum protocolo de avaliação ou
nova medição científica é executado automaticamente ao abrir a interface.

CPU/CUDA fica em **Preferências de processamento**. Filtro facial e quantidade de
resultados ficam em **Ajustes da busca**; no perfil Pesquisa, há parâmetros e
configurações experimentais identificados separadamente. A aplicação recebe uma
imagem de consulta, mostra os rostos detectados com numeração, permite escolher
a pessoa e exclui a mesma fotografia quando reconhecida por caminho ou SHA-256.
No uso pessoal, pontuações e caminhos técnicos ficam ocultos por padrão.

A imagem de referência é armazenada localmente com nome derivado do conteúdo,
evitando sobrescrita entre fotos diferentes com o mesmo nome. A análise facial
reutiliza um cache limitado de prévias pequenas e descritores, não de fotografias
inteiras em resolução original. Resultados da última busca ficam na sessão:
ajustes de exibição não recalculam o ranking; alterar foto, pessoa, álbum,
parâmetros ou os arquivos do índice exige uma nova busca. A paginação atualiza
apenas a galeria. Os caches de índices são limitados e invalidados quando seus
arquivos mudam ou quando um álbum é preparado ou removido pela interface.

O tema fica em `.streamlit/config.toml`, com componentes nativos, cartões e uma
paleta de contraste claro. A aplicação continua local: não foi publicada no
Streamlit Community Cloud nem usa o aplicativo de referência como serviço.

### Preparação pelo terminal (opcional)

Os indexadores também continuam aceitando uma pasta pelo argumento `--input-dir`:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/prepare_collection.py `
  --input-dir "C:\caminho\do\album" `
  --output-dir outputs/collections/minha_colecao `
  --device cuda
```

O preparador não copia as fotografias. Ele cria os índices `face` e `global`, omite nomes derivados das pastas e grava um manifesto da coleção.
Aceita JPG, JPEG, PNG, BMP, WebP, HEIC e HEIF em subpastas. Para HEIC/HEIF, instale
`pillow-heif==1.8.0` no ambiente do projeto antes de preparar a coleção. Vídeos não
são indexados. Se já existir uma coleção preparada antes desse suporte, atualize
o álbum pela interface para incluir os HEIC; índices antigos não se atualizam
sozinhos. Pelo terminal, use o mesmo `--output-dir` com `--overwrite`.

O ambiente local foi criado com `uv` e não inclui `pip`. Para adicionar apenas o
leitor HEIC, sem recriar o ambiente:

```powershell
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe pillow-heif==1.8.0
```

### Fotografias JPEG muito grandes

O limite global do Pillow continua ativo. Quando ele bloqueia um JPEG, o leitor
aceita no máximo 256 megapixels de origem e solicita decodificação em resolução
reduzida (1/8 por dimensão), limitada a 8 megapixels decodificados. Os arquivos
originais não são modificados. Essa leitura atende à extração global e às prévias
da interface; o leitor facial OpenCV dos JPEGs permanece inalterado. Outros
formatos acima do limite não ganham uma exceção automática.

O manifesto do índice global registra quais imagens receberam essa redução e as
dimensões usadas. O CSV de metadados mantém as dimensões originais. Os JPEGs comuns
das bases científicas continuam com a leitura anterior; resultados existentes
não são recalculados por esta alteração. Uma coleção que tinha falhas de leitura
precisa ser atualizada pela interface ou preparada novamente com `--overwrite`.

## Bases da avaliação científica

- **Gallagher:** comparação central entre face, contexto visual global e fusão, em fotografias com múltiplas pessoas.
- **LFW:** validação auxiliar do componente facial.
- **INRIA Holidays:** validação auxiliar do componente global.

As tarefas não são equivalentes e suas métricas não devem ser comparadas diretamente entre bases.

## Dados e privacidade

Este é um protótipo acadêmico. O perfil **Uso pessoal** simplifica a interface,
mas não altera as restrições dos modelos pré-treinados. Consulte
[dados, modelos e privacidade](docs/dados_modelos_privacidade.md) antes de qualquer
uso público, redistribuição ou disponibilização comercial.

Não versione fotografias, recortes faciais, embeddings, modelos baixados ou resultados que exponham pessoas. `data/raw/`, `data/query/`, `outputs/`, `.local/` e `laboratorio/` permanecem ignorados pelo Git.

O álbum familiar foi usado somente em teste local de aceitação. Ele não integra a evidência principal do artigo e não será redistribuído. As fotografias originais ficaram intactas.

## Documentos canônicos

- [Estado e decisões atuais](LEIA_PRIMEIRO.md)
- [Mapa da documentação](docs/README.md)
- [Protocolo vigente](docs/protocolo_avaliacao.md)
- [Resultados oficiais atuais](docs/resultados_experimentais_congelados.md)
- [Síntese bibliográfica](docs/sintese_revisao_bibliografica.md)
- [Evidências seguras Gallagher](docs/evidencias/gallagher_a01_a02_20260922_v2/README.md)
- [Fechamento técnico e científico](docs/fechamento_tecnico_2026.md)
- [Dados, modelos, privacidade e uso de IA](docs/dados_modelos_privacidade.md)
