# Proposta de comparação facial - 30/09/2026

**Estado: proposta para aprovação; não implementada nem executada.** Os resultados congelados e a monografia atual não foram alterados. Não foram baixados pesos nem datasets, instaladas dependências ou iniciados treinamentos.

## 1. O que estamos tentando responder

A proposta assinada inclui comparar algoritmos de detecção facial e geração de embeddings. São duas decisões diferentes: encontrar o rosto na fotografia e representar esse rosto numericamente para buscar a mesma pessoa. A comparação abaixo amplia a evidência da pesquisa sem transformar o TCC em um projeto de treinamento de redes.

Recomendo **avaliar modelos pré-treinados**, não treiná-los do zero nem ajustar seus pesos no Gallagher. Indexar fotos não é treinamento. Gallagher será avaliação, não material para ensinar o modelo e depois medir nele mesmo. Treinamento exigiria outro protocolo, dados de treinamento/validação/teste separados, orçamento computacional e revisão das condições de uso.

Perguntas adicionais delimitadas:

- Com o reconhecedor fixo, trocar o detector altera a recuperação de fotografias relevantes?
- Com o detector fixo, trocar o reconhecedor altera essa recuperação?
- O comportamento da fusão com contexto global se mantém nas combinações testadas?

## 2. Modelos recomendados e procedência

| Etapa | Referência atual | Alternativa recomendada | Justificativa |
|---|---|---|---|
| Detecção e pontos faciais | SCRFD-10G, `det_10g.onnx`, pacote buffalo_l | RetinaFace-R50, pesos de detecção publicados pelo InsightFace | RetinaFace já é exemplo na proposta e fornece caixas e cinco pontos faciais; permite trocar a detecção mantendo o restante controlado. |
| Embeddings | ArcFace, `w600k_r50.onnx`, buffalo_l | AdaFace IR-50, checkpoint oficial MS1MV2 | Alternativa documentada pelos autores, voltada à variação de qualidade; arquitetura de porte semelhante e inferência explicitamente documentada. Não pressupomos melhora no Gallagher. |

**Escolha de checkpoint AdaFace:** IR-50/MS1MV2, exatamente a variante utilizada pelo exemplo oficial `inference.py`, não selecionar a variante que obtiver melhor resultado no Gallagher. O README também oferece outras bases, mas não faremos uma busca entre todas elas. Link de pesos indicado pelos autores: https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view . Disponibilidade efetiva, termos e hash ainda precisam ser verificados antes do download e uso.

**RetinaFace:** https://github.com/deepinsight/insightface/tree/master/detection/retinaface . O checkpoint de detecção R50 não deve ser confundido com os pesos ImageNet usados para inicializar o treinamento. O código original depende de MXNet; a integração no ambiente atual ainda não está demonstrada. Usar ambiente isolado ou conversão validada, sem alterar o ambiente funcional atual. Se não houver uma execução rastreável viável, voltar à decisão de implementação; não substituir por um pacote de terceiros silenciosamente.

Esta é uma comparação de **sistemas com pesos pré-treinados**, não uma experiência isolando apenas a função de perda ArcFace versus AdaFace. Bases de treinamento, receitas e detalhes de arquitetura diferem. Mesmo ambos tendo 50 camadas, não são modelos idênticos fora da perda. Não se poderá concluir superioridade universal de uma perda.

### Por que não escolher FaceNet ou CosFace automaticamente?

As referências já existentes de FaceNet (2015), CosFace (2018) e ArcFace (2019) continuam importantes. Entretanto, um artigo não identifica sozinho o checkpoint a executar.

- O repositório `davidsandberg/facenet` oferece modelos Inception-ResNet-v1 treinados com softmax. Isso não é reprodução direta do FaceNet original treinado com triplet loss. Pode ser comparado como implementação explicitamente identificada, mas não com a alegação de comparar as receitas originais.
- `yule-li/CosFace` descreve uma reimplementação, diferenças de resultados e variantes de dimensão/alinhamento. Não foi demonstrado que seja o código dos autores do artigo. Não escolher esses pesos somente pelo nome do repositório.
- AdaFace acrescenta uma referência, mas oferece correspondência explícita entre artigo, código dos autores, checkpoint e inferência. A motivação de qualidade é pertinente; não é evidência de que vencerá neste acervo.

Não recomendo acrescentar um terceiro reconhecedor só para aumentar a contagem. Dois detectores e dois reconhecedores já permitem uma comparação delimitada. Se quisermos comparar **perdas** em condições iguais de treinamento, este plano não basta e deverá ser substituído por outro de maior escopo.

## 3. Configuração proposta

Valores marcados como escolha experimental são decisões nossas, não parâmetros obrigatórios dos artigos.

| Componente | Configuração |
|---|---|
| SCRFD | Preservar checkpoint atual; entrada 640 × 640; limiar de detecção 0,5; NMS 0,4. Conferir e registrar valores efetivos do runtime. |
| RetinaFace-R50 | Proposta: mesmo orçamento de imagem de entrada 640 × 640, proporção preservada e padding documentado; escala única, sem flip; limiar 0,5 e NMS 0,4 como ponto operacional predefinido. Validar adapter e coordenadas antes da avaliação. |
| ArcFace | Preservar pipeline atual: alinhamento por cinco pontos, 112 × 112, RGB no tensor, 512 dimensões, normalização L2. Manter a normalização que o loader detecta no ONNX, registrando média/desvio efetivos; não aplicar normalização duas vezes. |
| AdaFace | IR-50/MS1MV2; 112 × 112, BGR, pixel dividido por 255 seguido de `(x - 0,5) / 0,5`; 512 dimensões normalizadas; modo avaliação, sem gradientes, FP32. |
| Alinhamento | Por detector, reutilizar os mesmos cinco pontos e a mesma transformação geométrica para ambos os reconhecedores, após conferir compatibilidade com os templates originais. Não executar o MTCNN da demonstração AdaFace por trás do adapter. |
| Similaridade | Cosseno entre vetores normalizados do mesmo modelo. Para uma fotografia com vários rostos, máximo dos escores faciais, como no protocolo atual. Nunca comparar embedding ArcFace com AdaFace. |
| Contexto global | ResNet50 atual, pesos e descritores preservados quando hashes e entradas coincidirem. |
| Fusão | Mesma fórmula atual e pesos faciais 0,9; 0,7; 0,5, além de face isolada e global isolado. Não otimizar pesos no conjunto de avaliação. |
| Execução | Mesmo hardware; lote inicial de embeddings 32, reduzir apenas por memória e registrar; detector uma imagem por vez; sem mixed precision nesta comparação inicial. Lote é escolha operacional, não resultado científico. |

O mesmo limiar numérico de detecção não representa a mesma confiança ou taxa de erro nos dois modelos. O estudo comparará esses pontos operacionais, não curvas completas de detectores calibrados. Não copiar os números publicados do RetinaFace com múltiplas escalas como expectativa para nossa avaliação de escala única.

Learning rate, épocas, otimizador e margem de treinamento **não se aplicam à execução proposta**: os pesos estarão congelados. Se forem descritos, serão atributos históricos do treinamento dos autores, não configurações executadas por nós.

## 4. Quatro combinações, sem misturar alterações

| Execução | Detector | Reconhecedor | O que permite observar |
|---|---|---|---|
| A | SCRFD | ArcFace | Referência atual |
| B | RetinaFace | ArcFace | Efeito da troca do detector e seus pontos faciais |
| C | SCRFD | AdaFace | Efeito da troca do reconhecedor, com detecção fixa |
| D | RetinaFace | AdaFace | Combinação alternativa e possível interação |

Para cada combinação: face isolada e três fusões; global isolado é compartilhado. São 17 condições de ranking, **não 17 treinamentos**. Resultados anteriores ficam preservados; a nova rodada terá identificador próprio.

## 5. Cuidados obrigatórios com o protocolo

1. Congelar fontes, hashes dos modelos, código, dependências e configuração antes dos resultados.
2. Usar a mesma galeria, relevância, exclusão da fotografia de origem e regras de desempate em todos os métodos. Não retirar foto relevante porque um detector falhou.
3. As 20 consultas atuais passaram por validação com SCRFD. Mantê-las serve para continuidade, mas a seleção favorece consultas viáveis para a referência. Declarar isso e não apresentar a comparação como detector-agnóstica.
4. Para a comparação principal nova, definir painel de consultas por anotações e regra determinística **independente da detecção**, antes de observar resultados; manter as mesmas identidades e regra de frequência quando possível. Fotografias/rostos ambíguos recebem revisão de anotação, não escolha pelo desempenho. Executar as quatro combinações nesse painel. As 20 consultas antigas ficam como análise de continuidade, sem misturar suas médias com as novas.
5. Consulta sem rosto-alvo detectado: registrar falha e não substituir por vizinho ou excluir a consulta. Proposta: recuperação facial retorna vazio, AP/P@K/Recall@K zero; fusão aplica política de face ausente e contexto disponível, documentada antes da execução. Relatar cobertura separadamente. Na galeria, manter candidatos sem face no universo e declarar a pontuação/desempate usados.
6. Embeddings armazenados em índices separados por checkpoint/detector/alinhamento. Validar dimensão, cores, normalização e integridade antes de inferência completa.
7. Gallagher possui anotações de identidades/olhos que não constituem anotação exaustiva de todas as caixas. Não calcular precisão do detector dividindo contagens brutas de faces por anotações, nem tratar toda detecção sem anotação como falso positivo.
8. A comparação proposta mede sobretudo o efeito do detector **na recuperação**. Para AP oficial de detecção e uma comparação específica de SCRFD/RetinaFace, é necessário protocolo WIDER FACE com anotações adequadas. Isso será uma etapa adicional a decidir, não uma conclusão que Gallagher sozinho permite. Verificar dados existentes e condições de uso antes de solicitar qualquer download.
9. Não ajustar escolhas procurando melhorar o número final. Diferenças pequenas em poucas identidades devem ser tratadas com cautela; consultas/fotos relacionadas não são observações independentes irrestritas.

## 6. Métricas e gráficos

- Principal: mAP do ranking completo e AP por consulta; complementar: Precision@5/10 e Recall@5/10.
- Separar falhas de leitura, de detecção do alvo, de geração de embedding e erros de ordenação.
- Gráfico de mAP das quatro combinações, diferenças por consulta e Recall@K com K predefinidos (1, 5, 10, 20, 50).
- Relatar tempo de indexação e consulta com aquecimento, sincronização da GPU, repetições e separação entre leitura/detecção/embedding/busca; não comparar tempos de máquinas diferentes como efeito do modelo.
- Matriz binária de relevância em Top-10 pode ser complementar: relevante/não relevante versus retornada/não retornada. Não é matriz de identidade nem de detecção; fotos com várias pessoas e ranking não são classificação comum de uma classe por imagem. A matriz não substitui mAP.
- LFW/Holidays continuam auxiliares. Não é necessário retreinar ou repetir o ramo global inalterado para cada combinação. Não usar resultados de LFW como prova de ausência de sobreposição com pré-treinamento.

## 7. Base documental e leitura local

Biblioteca: `docs/Referencias/`, já ignorada pelo Git. PDFs novos têm título/autoria inicial, número de páginas e SHA-256 registrados no manifesto local. Isso valida o arquivo, não significa leitura crítica integral de cada referência baixada.

Fontes centrais:

- ArcFace e comparação de perdas: PDF local `09_Deng_2019_ArcFace.pdf`, especialmente experimentos. https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- FaceNet/CosFace: PDFs locais 07 e 08; usados como fundamentos e alternativas, não prova de procedência de qualquer checkpoint encontrado na internet.
- SCRFD: https://arxiv.org/abs/2105.04714 e versão ICLR 2022. Acesso local às versões ainda depende do download validado; não confundir os anos/versões.
- RetinaFace: PDF local `24_Deng_2020_RetinaFace.pdf`, seção 4.2 (treinamento/teste); documentação oficial local `RetinaFace_README.md` para o checkpoint disponibilizado. https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- AdaFace: PDF local `25_Kim_2022_AdaFace.pdf`, seção 4.1; documentação `AdaFace_README.md` e `AdaFace_inference.py.txt`. https://github.com/mk-minchul/AdaFace
- Jain et al.: `30_Jain_2005_Score_Normalization.pdf` como leitura complementar sobre normalização/fusão. Não implementamos uma normalização nova nesta etapa.

As duas novas referências essenciais são RetinaFace e AdaFace; o restante reaproveita a bibliografia existente. As páginas de código salvas são snapshots consultados nesta data; a implementação futura deve fixar commit e pesos, não depender de `master` mutável. Licença do código não equivale automaticamente à licença dos pesos ou dados.

## 8. Próxima etapa e critérios de saída

Após aprovação dos modelos e deste escopo: confirmar termos/disponibilidade dos dois checkpoints, viabilidade do RetinaFace, templates de alinhamento e painel de consultas independente. Só depois implementar adapters e testes, rodar piloto técnico sem selecionar parâmetros por resultados e congelar a execução completa.

Um bloqueio de checkpoint, licença ou runtime será informado antes de substituir modelos. O trabalho fecha com configurações identificadas, quatro combinações verificadas, todas as falhas contabilizadas e tabelas/gráficos rastreáveis. Não é necessário que a alternativa ou a fusão vença. Atualizar a monografia somente depois de interpretar os resultados válidos; até lá, ela continua descrevendo a rodada anterior.
