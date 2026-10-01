# Modelo e compilação

Origem: `monografia2024/monografia2024` fornecido por Artur, modelo de Cristiane
de Fátima dos Santos Cardoso. Manual de janeiro de 2013; fonte principal com
cabeçalho de março de 2019. São materiais históricos de formatação, não prova
de conformidade vigente. O manual, p. 2, diz que o uso é recomendado e condicionado
ao orientador. Nenhuma meta arbitrária de páginas foi adotada.

Preservados: classe report 12 pt A4, margens superior/esquerda de 30 mm e
inferior/direita de 20 mm, fonte Times, abertura de capítulos Glenn, capa com
identificação institucional, recuo de 1,25 cm, corpo com espaçamento 1,5 e
bibliografia autor-data. Logo institucional é o arquivo do template, sem assinatura.
Estilo limpo em `formatacao.sty`, sem pacotes duplicados, capítulos de exemplo,
bibliografia alheia, aprovações ou auxiliares antigos. Siglas são lista estática
curada, evitando geração de índice desnecessária. Resumo e abstract incluídos;
o exemplo deixava o abstract comentado, portanto sua obrigatoriedade atual
não foi inferida desse exemplo.

Fontes de Isaias consultadas apenas para estrutura. Arquivos ausentes e estilos
bibliográficos conflitantes não foram herdados. A nota máxima é informação
de Artur, não verificada e não utilizada como evidência acadêmica.

## Compilação verificada

Tectonic 0.17.0+20260731 já fornecido pelo aplicativo. Teste inicial restrito ao
cache falhou por ausência de report.cls. Artur autorizou especificamente baixar
recursos LaTeX necessários pelo Tectonic nesta conversa. A compilação mínima
de `minimo.tex` passou após esse download. Nenhum compilador ou dependência
experimental foi instalado. Projeto modular: usa compilação em disco; o
compilador nativo de documento avulso não suporta arquivos adicionais.

Comando no PowerShell, na raiz do repositório, com o executável já disponível:

```powershell
& $env:CODEX_TECTONIC_PATH --keep-logs --keep-intermediates --outdir docs/monografia/build docs/monografia/main.tex
```

Os auxiliares e logs ficam em `build/`, ignorado. O PDF conferido foi copiado
para `monografia_artur_revisao.pdf`. `referencias.bib` é a única bibliografia
mestre; o pacote abntex2cite seleciona um único estilo. A conformidade final
com o manual vigente e a edição aplicável da NBR 6023 permanece pendente.


Compilação final concluída em 29/09/2026, 40 páginas. A bibliografia usa corpo
menor (aproximadamente 11 pt), espaçamento simples e alinhamento à esquerda
para acomodar URLs. Os DOI estão também em notas porque o estilo histórico
não imprime o campo DOI diretamente. `hyperref` precede `abntex2cite`, conforme
[exemplo oficial](https://github.com/abntex/abntex2/blob/master/doc/latex/abntex2/abntex2cite-alf.tex).
As páginas pré-textuais são contadas, com numeração visível a partir do corpo.
Detalhes da conferência e limites em VERIFICACAO_ENTREGA.md.
