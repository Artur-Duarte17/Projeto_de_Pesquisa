# Resultados experimentais oficiais

Data de referência: **24 de setembro de 2026**.

## Resultado principal — Gallagher corrigido

A EX-035 usa o protocolo corrigido EX-033/EX-034. As cinco configurações compartilham 20 consultas, 589 fotografias na galeria, 588 candidatas após excluir a fonte e 884 relações de relevância.

| Método | P@5 | P@10 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| Somente face | 0,900000 | 0,680000 | 0,577119 | 0,687856 | **0,951446** |
| Somente contexto global | 0,260000 | 0,195000 | 0,203512 | 0,234712 | 0,280864 |
| Face/contexto 0,9/0,1 | 0,900000 | 0,680000 | 0,577119 | 0,687856 | 0,946297 |
| Face/contexto 0,7/0,3 | 0,840000 | 0,675000 | 0,540571 | 0,682856 | 0,901525 |
| Face/contexto 0,5/0,5 | 0,770000 | 0,630000 | 0,481033 | 0,634905 | 0,813358 |

Nenhuma fusão superou o baseline facial no mAP agregado.

## Correções incorporadas

- Duas fotografias da identidade 2 sem face detectada continuam no gabarito e contam como relevantes.
- A consulta `gallagher_id6_q1` utiliza o rosto que contém o ponto médio dos olhos anotados da pessoa-alvo.
- A fotografia-fonte é excluída de todos os métodos.
- O ranking é integral; Top-10 limita somente as linhas preservadas como exemplo.
- Empates são resolvidos por score decrescente e `image_id` crescente.

## Resultados auxiliares

| Componente | Base | Consultas | mAP | Papel |
|---|---|---:|---:|---|
| Busca facial | LFW | 1.672 | 0,965135 | validar o componente facial em fotografias organizadas por identidade |
| Busca global | INRIA Holidays | 500 | 0,842612 | validar o descritor de imagem inteira |

Esses números representam tarefas diferentes e não são comparações diretas com Gallagher.

## Proveniência

| Execução | Função | Commit registrado |
|---|---|---|
| EX-008/009 | índice e avaliação facial LFW | manifests locais da execução |
| EX-013/014 | índice e avaliação global Holidays | manifests locais da execução |
| EX-010 | índice facial Gallagher usado pela correção | `66d41bc` |
| EX-015 | índice global Gallagher usado pela correção | `d300a6e` |
| EX-033 | consultas e relevância corrigidas | `b0abf4c` |
| EX-034 | protocolo pareado corrigido | `938911b` |
| EX-035 | comparação final das cinco configurações | `938911b` |

As evidências leves da correção estão em `docs/evidencias/gallagher_a01_a02_20260922_v2/`.

## Limitações obrigatórias

- Gallagher usa 20 identidades entre as mais frequentes e uma consulta por identidade.
- Fotografias do mesmo álbum podem ser correlacionadas.
- A diferença entre face (`0,951446`) e fusão 0,9/0,1 (`0,946297`) é pequena; a análise estatística pareada ainda será consolidada.
- Os resultados não demonstram desempenho em qualquer domínio profissional específico.
- O teste futuro com álbum familiar será privado e terá caráter de aceitação do sistema, não de benchmark científico.

## Regra para substituir estes números

Uma nova execução só poderá substituir a tabela se usar o protocolo corrigido, entradas congeladas, árvore Git limpa, hashes verificáveis, manifestos completos e auditoria equivalente. Até essa conferência, os valores acima permanecem os resultados oficiais vigentes.
