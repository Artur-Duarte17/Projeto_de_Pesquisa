# Monografia de Artur Duarte Monteiro

**Fonte revisada em 01/10/2026.** A limpeza editorial retirou marcas de rascunho
e referências à assistência automatizada sem alterar protocolo, métricas ou conclusões
científicas. O PDF versionado ainda corresponde à compilação anterior e deve ser
recompilado e conferido antes de substituir a cópia de entrega.

- [PDF para leitura](monografia_artur_revisao.pdf).
- [Fonte principal](main.tex), com seis capítulos modulares e um apêndice.
- [Guia de compreensão e revisão](GUIA_DE_REVISAO_ARTUR.md).
- [Checkpoint](ESTADO_EXECUCAO.md) e [pendências humanas/institucionais](PENDENCIAS.md).
- [Verificação da entrega](VERIFICACAO_ENTREGA.md) e [manifesto de arquivos](MANIFESTO_ENTREGA.json).
- [Bibliografia mestre](referencias.bib), [controle das fontes](CONTROLE_FONTES.md)
  e [mapa entre objetivos e evidências](MAPA_EVIDENCIAS.md).

## Organização e reconstrução documental

`formatacao.sty` contém o estilo; `pretextuais.tex` contém capa, folha de rosto,
resumo/abstract e listas. `capitulos/` guarda o corpo textual,
`figuras/` e `tabelas/` guardam elementos usados no PDF. O pacote leve de métricas
fica em `../evidencias/fechamento_20260928_cuda/`.

O procedimento de compilação e a origem do template estão em [TEMPLATE.md](TEMPLATE.md).
Para editar o texto, manter `main.tex` e seus módulos; não editar o PDF diretamente.
Depois de qualquer alteração, recompilar e revisar as páginas afetadas antes de
substituir o PDF de entrega e atualizar seu manifesto.

Os auxiliares são documentais: `preparar_evidencias.py` confere preservação e
copia agregados; `gerar_resultados.py` deriva tabelas/gráficos; `conferir_documento.py`
confere referências, hashes e renderiza páginas. Não executam modelos.
`conferir_metadados.py` consulta registros Crossref por DOI; o retorno não substitui
leitura ou aprovação de uma referência. Não é necessário executá-los para ler
ou compilar as fontes já prontas. A geração de gráficos usa matplotlib e NumPy
já existentes; a renderização usa pypdf, pypdfium2 e Pillow do runtime documental.

`build/`, `cache/` e `local/` estão ignorados no Git. Contêm logs, renderizações,
leituras de apoio e rastreabilidade restrita. Não devem ser incluídos automaticamente
em um pacote de divulgação. A cópia independente dos artefatos volumosos continua
pendente; o pacote leve não a substitui.
