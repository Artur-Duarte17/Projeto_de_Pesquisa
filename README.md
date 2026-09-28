# Recuperação de fotografias por identidade facial e contexto visual

Sistema experimental para localizar fotografias de uma pessoa em coleções com múltiplos rostos. A versão ativa combina:

- detecção e descritores faciais com InsightFace/ArcFace;
- descritores globais de imagem com ResNet50 pré-treinada no ImageNet;
- busca facial, busca visual global e fusão tardia de escores;
- métricas de recuperação e manifestos de execução;
- interface local em Streamlit.

Antes de usar resultados ou executar experimentos, leia [LEIA_PRIMEIRO.md](LEIA_PRIMEIRO.md).

## Estado da versão

O protocolo Gallagher foi corrigido e reavaliado nas execuções EX-033, EX-034 e EX-035. Entre as cinco configurações avaliadas, a busca somente facial obteve o maior mAP (`0,951446`). Nenhuma fusão testada superou esse baseline.

Esses resultados continuam válidos como evidência já produzida. Artur reexecutou o fluxo científico completo após a reorganização, no commit limpo `954d749`, e os resultados científicos verificados coincidiram com os da referência anterior. Ainda falta o teste privado de aceitação com um álbum real. A reprodução não equivale, por si só, à aprovação para publicação.

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

O arquivo `requirements/experiment-gpu.in` contém as dependências diretas. Artur regenerou o lock em 24/09/2026; a resolução terminou com 83 pacotes e retirou `gdown` e suas dependências exclusivas. Os antigos cinco arquivos `app.txt`, `face.txt`, `fusion.txt`, `global.txt` e `prep.txt` eram apenas apontadores idênticos para esse mesmo lock e foram removidos.

## Testes automatizados

```powershell
laboratorio/cibir_gpu/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Os testes verificam regras de ranking, exclusão da imagem-fonte, métricas, seleção da pessoa-alvo e integridade do protocolo. Eles não substituem a execução com modelos e fotografias reais.

Artur executou a suíte da estrutura atual em 24/09/2026: **37 testes passaram**, incluindo fronteiras da arquitetura, busca, protocolo Gallagher e isolamento do destino da validação. A execução científica completa foi realizada separadamente, com modelos e datasets reais, e comparada com a referência.

## Reprodução científica final

Artur executou o comando abaixo em 24/09/2026, com a árvore Git limpa no commit `954d749`:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/run_final_validation.py --run-root outputs/validation_runs/arquitetura_v1
```

O orquestrador valida o ambiente, reexecuta a avaliação facial LFW, a avaliação global INRIA Holidays e o protocolo Gallagher corrigido, e grava um manifesto que liga todas as saídas. A aquisição dos datasets não é refeita: os baixadores são utilitários de preparação, não etapas da reprodução final. O fluxo recusa por padrão uma árvore Git suja e saídas finais preexistentes.

As saídas anteriores em `outputs/final/` permanecem intactas. A nova execução está em `outputs/validation_runs/arquitetura_v1/`: os quatro índices e metadados, os gabaritos LFW e Gallagher, as métricas científicas e os Top-10 salvos foram conferidos com a referência. Tempos e caminhos dos recortes variaram; os rankings completos não são persistidos para comparação direta linha a linha. **Não repita o comando acima no mesmo destino**, pois ele exige uma pasta nova ou vazia. Para outra execução, escolha um novo nome sob `outputs/validation_runs/`.

O subfluxo Gallagher continua disponível isoladamente em `scripts/run_final_gallagher.py`, mas o comando acima é o critério de liberação da versão completa.

## Usar uma coleção própria

Os indexadores recebem qualquer pasta de fotografias pelo argumento `--input-dir`; não é necessário copiar o álbum para o projeto.

### 1. Preparar os dois índices

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/prepare_collection.py `
  --input-dir "C:\caminho\do\album" `
  --output-dir outputs/collections/minha_colecao `
  --device cuda
```

O preparador não copia as fotografias. Ele cria os índices `face` e `global`, omite nomes derivados das pastas e grava um manifesto da coleção.
Aceita JPG, JPEG, PNG, BMP, WebP, HEIC e HEIF em subpastas. Para HEIC/HEIF, instale
`pillow-heif==1.8.0` no ambiente do projeto antes de preparar a coleção. Vídeos não
são indexados. Se já existir uma coleção preparada antes desse suporte, use um novo
`--output-dir` para incluir os HEIC; índices antigos não se atualizam sozinhos.

O ambiente local foi criado com `uv` e não inclui `pip`. Para adicionar apenas o
leitor HEIC, sem recriar o ambiente:

```powershell
uv pip install --python laboratorio/cibir_gpu/Scripts/python.exe pillow-heif==1.8.0
```

### 2. Abrir a aplicação

```powershell
laboratorio/cibir_gpu/Scripts/python.exe -m streamlit run scripts/app/streamlit_app.py
```

Na opção `Personalizado`, informe os diretórios `face` e `global` criados acima. A aplicação recebe uma imagem de consulta, mostra os rostos detectados com numeração, exige a escolha da pessoa que deve ser procurada, exclui do ranking a mesma fotografia quando reconhecida por caminho ou SHA-256 e mostra as imagens recuperadas.

## Bases da avaliação científica

- **Gallagher:** comparação central entre face, contexto visual global e fusão, em fotografias com múltiplas pessoas.
- **LFW:** validação auxiliar do componente facial.
- **INRIA Holidays:** validação auxiliar do componente global.

As tarefas não são equivalentes e suas métricas não devem ser comparadas diretamente entre bases.

## Dados e privacidade

Não versione fotografias, recortes faciais, embeddings, modelos baixados ou resultados que exponham pessoas. `data/raw/`, `data/query/`, `outputs/`, `.local/` e `laboratorio/` permanecem ignorados pelo Git.

O álbum familiar será usado somente em teste local de aceitação. Ele não integra a evidência principal do artigo e não será redistribuído.

## Documentos canônicos

- [Estado e decisões atuais](LEIA_PRIMEIRO.md)
- [Mapa da documentação](docs/README.md)
- [Protocolo vigente](docs/protocolo_avaliacao.md)
- [Resultados oficiais atuais](docs/resultados_experimentais_congelados.md)
- [Síntese bibliográfica](docs/sintese_revisao_bibliografica.md)
- [Evidências seguras Gallagher](docs/evidencias/gallagher_a01_a02_20260922_v2/README.md)
