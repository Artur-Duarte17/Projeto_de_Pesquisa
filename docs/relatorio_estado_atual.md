# Estado atual do projeto

Data de referência: **24 de setembro de 2026**.

## 1. O que o sistema faz

O sistema procura fotografias dentro de uma coleção por três caminhos:

1. **Busca facial:** usa o rosto escolhido na consulta para localizar outras fotos da mesma pessoa.
2. **Busca global:** compara a aparência da imagem inteira, sem identificar uma pessoa específica.
3. **Fusão tardia:** combina as pontuações da busca facial e da busca global.

Os modelos já vêm pré-treinados. O projeto não treina uma rede neural nova; ele detecta rostos, extrai vetores numéricos, cria índices e avalia rankings de recuperação.

## 2. Pergunta científica

> Em coleções fotográficas com múltiplas pessoas, acrescentar contexto visual global por fusão tardia melhora a recuperação baseada em identidade facial?

O Gallagher é a base central para responder essa pergunta. O LFW verifica o componente facial e o INRIA Holidays verifica o componente de imagem inteira. Essas tarefas são diferentes e suas métricas não devem ser comparadas diretamente.

## 3. Resultado vigente

A auditoria encontrou dois problemas no protocolo Gallagher anterior:

- fotos relevantes eram excluídas do gabarito quando o detector não encontrava um rosto;
- uma consulta usava o rosto de uma pessoa vizinha, e não o rosto da pessoa-alvo.

As correções A01 e A02 eliminaram esses problemas. A reavaliação EX-035 utilizou 20 consultas, 588 candidatas por ranking e cinco configurações predefinidas. O maior mAP foi obtido pela busca somente facial (`0,951446`). Nenhuma fusão testada superou esse baseline.

Isso é um resultado científico válido: dentro do protocolo avaliado, adicionar o contexto global da maneira testada não melhorou o resultado agregado.

## 4. O que integra a versão final

- indexação e recuperação facial;
- indexação e recuperação global;
- fusão tardia;
- protocolo Gallagher corrigido;
- verificações auxiliares LFW e INRIA Holidays;
- interface local em Streamlit;
- seleção explícita da pessoa em consultas com vários rostos;
- testes metodológicos e manifestos com hashes;
- preparação de uma coleção indicada pelo usuário, sem copiar as fotografias.

Materiais encerrados, privados ou substituídos foram retirados da árvore ativa e preservados em `C:\Projeto_de_Pesquisa_arquivo_local\2026-09-24_pre_finalizacao`.

## 5. Como a versão final será validada

O autor executará, depois da revisão e do commit, um único orquestrador:

```powershell
laboratorio/cibir_gpu/Scripts/python.exe scripts/run_final_validation.py
```

Esse fluxo:

1. confere o ambiente e a GPU;
2. recria o índice facial e a avaliação LFW;
3. recria o índice global e a avaliação Holidays;
4. recria os índices e o protocolo Gallagher corrigido;
5. avalia as cinco configurações Gallagher;
6. registra manifestos, hashes, contagens e métricas.

Os scripts de download não fazem parte dessa execução. Eles servem apenas para adquirir dados ausentes; as bases já existentes serão reutilizadas.

## 6. Teste com o álbum familiar

O álbum de 139 fotografias será um teste privado de aceitação, não uma nova base científica. Ele responderá a uma pergunta prática: uma pessoa consegue indicar sua própria coleção, escolher alguém em uma foto e receber resultados úteis?

O álbum pode permanecer na pasta escolhida pelo autor. `scripts/prepare_collection.py` recebe esse caminho, cria índices em `outputs/collections/` e não copia as fotografias. Na aplicação, o usuário escolhe explicitamente o rosto que deseja procurar quando há várias pessoas na consulta.

As fotografias, embeddings e resultados do álbum não serão versionados nem usados no artigo como evidência científica.

## 7. O que ainda não foi comprovado

As mudanças de organização e os novos orquestradores ainda não foram executados. Portanto, neste momento:

- Artur executou a suíte automatizada em 24/09/2026: 27 testes passaram, sem falhas;
- a reprodução completa LFW, Holidays e Gallagher ainda precisa ser rodada;
- o álbum familiar ainda precisa ser preparado e testado;
- as novas métricas precisam ser comparadas com os resultados congelados;
- o projeto ainda não está liberado para iniciar a versão final do artigo.

## 8. Critério de encerramento técnico

A versão poderá ser considerada tecnicamente consolidada quando:

1. o lock de dependências estiver regenerado;
2. todos os testes automatizados passarem;
3. o orquestrador final terminar sem erro em uma árvore Git limpa;
4. manifests, hashes, contagens e métricas forem conferidos;
5. divergências em relação às execuções válidas anteriores forem explicadas;
6. o teste privado com o álbum funcionar nos três modos relevantes;
7. nenhuma fotografia ou dado biométrico privado aparecer no Git.

Somente depois desse fechamento a redação final em inglês deverá usar os números da nova execução.
