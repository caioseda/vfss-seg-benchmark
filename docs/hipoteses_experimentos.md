# Hipóteses e experimentos: origem, ordem original e reordenação

**VFSS — segmentação das vértebras C2 e C4 sob supervisão esparsa**

Este documento substitui a versão anterior. Ele responde a três perguntas: de onde veio cada hipótese, de onde veio cada experimento, e por que a ordem de execução que eu proponho difere da ordem proposta pelo professor.

---

## Parte I — Vocabulário

Esta seção existe para que nenhum termo do documento fique sem definição. Ela é curta, mas cada item aparece depois em critérios de decisão, e uma leitura ambígua aqui produz uma decisão errada adiante.

**Anotação de referência.** A máscara traçada por um humano. É a resposta considerada correta na avaliação. Existem cerca de 1.000 delas, aproximadamente 5 por vídeo, em posições aleatórias dentro de cada exame.

**Pseudo-rótulo.** Uma máscara produzida pelo próprio modelo para um frame que não tem anotação de referência, e que é usada como se fosse a resposta correta durante o treino. É o objeto central de qualquer método semi-supervisionado: a qualidade do treino depende da qualidade desses rótulos inventados.

**Perturbação fraca e perturbação forte.** Alterações aplicadas à imagem de entrada antes de o modelo prever a máscara. Fraca significa alteração leve (pequeno deslocamento, pequena mudança de brilho); forte significa alteração agressiva (ruído, recorte, apagamento de regiões). O DiffRect usa esse par para criar duas versões da mesma predição com qualidades diferentes: presume-se que a versão prevista a partir da imagem pouco alterada seja melhor que a prevista a partir da imagem muito alterada.

**Transformação rígida no plano da imagem.** Um conjunto de quatro números — deslocamento horizontal, deslocamento vertical, ângulo de rotação e fator de escala — que descreve como reposicionar uma máscara. Não deforma a máscara: apenas move, gira e redimensiona.

**Transporte de máscara.** Aplicar essa transformação à máscara de um frame para obter uma estimativa da máscara de outro frame. "Transportar a máscara do frame A para o frame B" significa: reposicionar a máscara de A de modo que ela caia onde a estrutura está em B.

**Registro de imagem.** O procedimento que estima essa transformação a partir de duas imagens, procurando os quatro parâmetros que maximizam a semelhança entre elas.

**Resíduo após alinhamento.** O que sobra de diferença entre duas máscaras depois de aplicar a melhor transformação rígida possível entre elas. Se o resíduo é pequeno, a diferença entre as duas máscaras era essencialmente uma questão de posição. Se é grande, a forma projetada mudou, e nenhum reposicionamento resolve.

**Condição-teto.** Um cálculo feito com informação que o método real **não terá** quando estiver funcionando, cujo resultado serve como limite superior do que qualquer versão realizável pode alcançar.

> Exemplo concreto, porque este termo aparece muitas vezes adiante. Quero saber se transportar a máscara de um frame para outro é útil. Na prática, a transformação teria de ser estimada a partir das imagens, e essa estimativa tem erro. Mas eu tenho, em pares de frames que ambos possuem anotação de referência, a possibilidade de calcular a transformação usando as duas máscaras corretas — isto é, usando a resposta que o método não teria acesso. Essa transformação é a melhor possível. Se o transporte feito com ela já não for útil, então nenhuma estimativa realizável será útil, porque toda estimativa realizável é pior. Essa é a lógica: **a condição-teto serve para descartar barato, não para prometer resultado.** Um resultado bom na condição-teto não garante nada; um resultado ruim na condição-teto encerra a questão.
>
> Na literatura em inglês isso costuma ser chamado de *oracle*. Uso "condição-teto" neste documento.

**Comparador.** A quantidade contra a qual um resultado é medido para decidir se ele significa alguma coisa. Uma hipótese sem comparador declarado não é testável: "a informação temporal ajuda" não diz ajuda em relação a quê.

**Critério de descarte.** O resultado numérico que, se observado, encerra uma linha de investigação. Deve ser escrito **antes** de rodar o experimento, senão qualquer número vira confirmação.

**Janela e passo temporal.** Janela é o conjunto de frames vizinhos considerados em torno de um frame alvo. Passo é a distância, em número de frames, entre o frame alvo e o vizinho escolhido. Neste documento, essa distância é chamada `g`.

**Retificação latente do DiffRect.** O módulo do DiffRect que aprende a corrigir pseudo-rótulos. Conforme já levantado neste projeto, ele opera sobre a representação comprimida do **rótulo**, sem olhar para a imagem. Essa propriedade importa em duas hipóteses adiante, porque significa que a correção não pode usar evidência visual para desfazer um erro de posição ou de identidade.

**Escalar de calibração do DiffRect.** O DiffRect calcula uma medida de quanto as duas versões perturbadas do pseudo-rótulo divergem entre si, e usa isso para calibrar a correção. Isso importa porque a ideia de "usar a semelhança entre frames vizinhos como indicador de confiabilidade" é estruturalmente parecida com essa medida — o que levanta a pergunta de se o sinal temporal acrescenta algo além do que o método já computa.

---

## Parte II — Os documentos em circulação e a lógica interna de cada um

Existem quatro fontes de hipóteses e experimentos. Elas não são versões sucessivas do mesmo plano: são três lógicas distintas, com ordenações distintas e critérios distintos do que deve vir primeiro.

### Fonte A — Plano V2 (professor), `Plano_VFSS_rev2.md`

**Descrição.** O plano parte de uma observação sobre os dados: a vértebra é um osso, não deforma, e entre dois frames próximos do mesmo exame o que muda é onde ela está na imagem, não a forma dela. Disso decorre uma proposta de usar o movimento real do exame no lugar das perturbações sintéticas do DiffRect, em três pontos de integração:

- **Integração A** — o frame vizinho, com sua máscara transportada de volta para o referencial do frame alvo, substitui (ou acompanha) a perturbação forte sintética. O par "versão ruim → versão boa" que o DiffRect precisa passa a ser "máscara transportada do vizinho → máscara prevista do alvo".
- **Integração B** — em vez de um vizinho, uma janela deles (por exemplo, de `t−3` a `t+3`). Prevê-se a máscara de cada um, transportam-se todas para o referencial do alvo e combinam-se por média ou voto. O argumento é que o borrão de movimento e a oclusão mudam de frame para frame, enquanto o osso não muda; combinar cancela parte do erro. O pseudo-rótulo consolidado é melhor que o de frame único, e esse par alimenta o mecanismo de correção.
- **Integração C** — um termo adicional na função de custo que penaliza apenas a parte da diferença entre máscaras de frames vizinhos que não é explicável por deslocamento, giro e escala. Marcado como opcional e primeiro item a cair.

**Lógica de ordenação.** O plano é organizado por **custo crescente e por dependência de infraestrutura**. As medições baratas que testam a premissa vêm primeiro (semana 2), o cache de transformações é identificado como caminho crítico (semana 3), e as integrações vêm depois, da menos invasiva (B, que é pré-processamento) para a mais invasiva (A, que mexe no laço de treino; C, que exige estimar transformação dentro do treino).

**Ordem original.**

| Semana | Item |
|---|---|
| 1 | E0 — fundação: mapeamento paciente/vídeo, divisões, higienização das anotações |
| 2 | E1 — quanto muda entre frames; E2 — até onde a aproximação rígida vale; E2b — escolha do estimador de transformação; E3 — quanto a compressão latente perde |
| 3 | Cache de transformações para todos os vídeos; início dos baselines |
| 3–5 | E4a — baselines por orçamento de anotação: supervisionado, Mean Teacher, FixMatch, DiffRect, supervisionado com todos os rótulos |
| 5–6 | Integração B |
| 6–7 | Integração A |
| 7 | E4b — variantes propostas nos mesmos orçamentos |
| 8–9 | E5 — ganho estratificado por movimento |
| 9–10 | Integração C (opcional) |
| 10–11 | E6 — ablações; E7 — qualidade sem anotação (exploratório) |
| 12–13 | Consolidação |

**O que o plano assume sem testar.** Que a informação transportada acrescenta algo em relação ao que o modelo já produz sozinho. Essa suposição não aparece como hipótese em nenhum lugar do documento; ela está embutida na decisão de construir as três integrações.

---

### Fonte B — Sua crítica falada, registrada em `Crítica-ao-plano-adaptado.md` (26/08)

**Descrição.** São três críticas distintas mais uma proposta nova. Registro separadamente porque a atribuição importa e porque as três têm status epistêmico diferente.

1. **Mistura de distribuições.** O plano coloca, na mesma hierarquia de qualidade, uma perturbação sintética (artificial, controlada) e a diferença real entre dois frames (movimento, mudança de projeção, borrão, oclusão, mudança do que está visível). Sua objeção é que essas duas coisas não representam o mesmo fenômeno e talvez não devam ocupar a mesma escala de qualidade. **Esta crítica é correta e é testável** — virou a hipótese H4 e o experimento X6.

2. **A estimação da transformação pode virar o gargalo.** A cadeia é longa — pseudo-rótulo, região de interesse, correspondência, transformação, transporte, novo pseudo-rótulo — e cada elo introduz erro antes de chegar ao objeto de interesse. Sua conclusão foi que a linha é válida mas frágil, e que precisa ser validada separadamente. **O plano já concorda parcialmente** (é o E2b), mas o E2b decide qual estimador é melhor, não se algum deles é bom o bastante. Virou a hipótese H2 e o experimento X5 ampliado.

3. **A adaptação ao DiffRect é pequena demais.** O plano reaproveita quase toda a formulação e altera apenas o mecanismo de geração de um dos níveis de qualidade. Você observou que, ao introduzir movimento, a unidade do problema deixa de ser o frame e passa a ser o par de frames, e que passam a existir três tipos de par: ambos anotados, ambos não anotados, e um de cada. **Esta observação é estrutural e correta**, mas ela não é, por si só, uma hipótese: é um argumento de que a formulação está incompleta. Ela se torna testável apenas através de H4.

4. **Proposta nova: condicionamento temporal.** Entre dois frames anotados separados por dezenas de frames existe uma sequência visual completa. Sua ideia é que essa sequência poderia condicionar a geração ou correção do pseudo-rótulo, sem que seja necessário estimar explicitamente uma transformação. Você declarou explicitamente que não considera essa formulação melhor que A/B, e que quer compará-las. Virou H10; recomendo mantê-la fora de escopo, com regra de reentrada (Parte VII).

---

### Fonte C — Reformulação do ChatGPT (26/08), no mesmo arquivo

**Descrição.** Reorganizou sua fala em cinco hipóteses encadeadas (H1 a H5) mais uma opcional, e propôs cinco experimentos em quatro fases. A contribuição real dessa fonte é o princípio de sequenciamento: **validar o fenômeno antes de escolher a arquitetura**, e definir critérios de parada antes de começar.

**Ordem original proposta.**

| Fase | Item |
|---|---|
| 0 | Estabelecer baseline: DiffRect funcionando, métricas confiáveis de segmentação e de pseudo-rótulo |
| 1 | Experimento 1 — propagar o rótulo entre pares de frames anotados, escondendo o segundo, e medir contra ele |
| 2 | Experimento 2 — transformação explícita contra condicionamento temporal; Experimento 3 — ablação temporal (só extremos contra extremos + intermediários) |
| 3 | Experimento 4 — DiffRect original contra DiffRect com mecanismo temporal |
| 4 | Experimento 5 — distância entre máscaras como indicador de confiabilidade; demais refinamentos |

**Onde essa fonte erra.** O Experimento 1, como formulado, mede o Dice da máscara transportada contra a anotação de referência. Isso responde "a propagação é precisa?", que é a mesma pergunta do E2 do professor — e o E2 responde melhor, porque usa a melhor transformação possível em vez de uma estimada. O Experimento 1 não tem comparador contra o modelo, e por isso não responde a pergunta que ele diz responder. Detalho em H3.

**Uma segunda observação sobre essa fonte.** A ordenação proposta coloca o Experimento 2 (explícito contra implícito) na Fase 2, logo depois do Experimento 1. Isso é caro e não decide nada: as duas alternativas diferem em arquitetura, número de parâmetros, entrada e regime de treino ao mesmo tempo, de modo que uma diferença de resultado não pode ser atribuída ao mecanismo temporal.

---

### Fonte D — `Explorar-métodos-alternativos.md` (27/08)

**Descrição.** Não propõe experimentos. Propõe uma mudança de enquadramento: as hipóteses não são sobre DiffRect, são sobre um problema mais geral de aproveitar informação temporal com anotação esparsa, e existem muitas famílias de método que atacam esse problema (fluxo óptico, registro, rastreamento, redes temporais, propagação de rótulo, teacher-student, difusão). A recomendação é montar um mapa hipótese × solução existente × lacuna antes de escolher a implementação.

**Status.** O argumento é correto como princípio. Como plano de ação, tem um risco: uma revisão orientada por cinco hipóteses abstratas, sem limite de escopo, consome semanas e não produz decisão. Minha recomendação (Parte VIII) é reduzir isso a duas perguntas específicas de novidade, com prazo fechado, rodando em paralelo com os experimentos baratos.

---

### Fonte E — Minha reordenação

**Descrição.** Não é um quarto plano. É uma reordenação das fontes A–C mais quatro experimentos novos, guiada por um único princípio: **um experimento só vem antes de outro se puder tornar o outro desnecessário.** As quatro adições existem porque identifiquei quatro perguntas que nenhuma das fontes faz e cujas respostas mudam o que vale a pena construir.

---

## Parte III — Proveniência das hipóteses

A coluna "forma original" mostra como a hipótese apareceu na fonte; a coluna "forma corrigida" mostra a formulação que eu proponho, com comparador explícito. Onde as duas coincidem, a hipótese já estava bem formulada.

| # | Fonte | Forma original na fonte | Forma corrigida (com comparador) | Por que mudou |
|---|---|---|---|---|
| **H0** | Plano V2, seção do E4 e "Sobre saturação" | Existe implicitamente: o plano prevê que a tarefa pode estar saturada e propõe trocar o eixo de escassez se isso ocorrer | Existe margem para semi-supervisão? Comparador: supervisionado com todos os rótulos contra supervisionado com 1 rótulo por vídeo, medido contra a variação entre sementes de treino | Estava presente como ressalva dentro de um experimento tardio; promovo a hipótese de primeira ordem |
| **H0b** | **Minha** | — | O erro residual do modelo é dominado por confusão de identidade entre vértebras e por precisão de contorno, não por área. Comparador: decomposição do erro do baseline em quatro classes | Nenhuma fonte pergunta *qual erro* precisa ser corrigido, e isso determina se um método temporal tem mecanismo plausível |
| **H1** | Plano V2, seções 2.2 e 2.3, medida pelo E2 | A aproximação rígida vale localmente e o alcance dela é mensurável | Idem, com duas correções: transformações separadas para C2 e C4, e curva parametrizada por deslocamento medido além de por distância temporal | A coluna cervical é uma cadeia articulada, e a distância temporal é um proxy ruim de movimento |
| **H2** | Plano V2 (E2b) + sua crítica nº 2 | Qual estimador de transformação reconstrói melhor a máscara | A transformação é estimável, sem usar anotação, com erro menor que o resíduo de H1 — e a cauda de falhas é pequena o bastante para aplicar a 40.000 frames | O E2b compara estimadores; não estabelece se algum é suficiente, nem quantifica a fração de falhas graves |
| **H3** | **Minha**, motivada pelo defeito do Experimento 1 do ChatGPT | ChatGPT: "a informação temporal é útil" (sem comparador) | A máscara transportada acrescenta informação **além** do que o modelo já produz no frame alvo. Comparador: a predição do próprio modelo naquele frame | Sem esse comparador, um resultado positivo não distingue "propagação boa" de "propagação inútil porque o modelo já é melhor" |
| **H4** | **Sua crítica nº 1**, formalizada por mim | "Estamos misturando duas distribuições diferentes na mesma hierarquia" | A máscara transportada ocupa posição ordenada e estável na hierarquia de qualidade do DiffRect. Comparador: as distribuições de qualidade das perturbações fraca e forte | Sua objeção era conceitual e não tinha número; a versão com número é falseável |
| **H5** | Sua fala (26/08) sobre os frames intermediários; ChatGPT, Experimento 3 | Os intermediários contêm informação sobre como a estrutura passou de um estado a outro | Os frames intermediários acrescentam informação de trajetória além dos extremos. Comparador: registro direto de A para B em um passo | A versão original só era testável com arquitetura nova; proponho uma versão geométrica barata |
| **H6** | Sua fala sobre distância entre rótulos como proxy de movimento; ChatGPT, hipótese opcional e Experimento 5 | A distância entre frames/máscaras informa a confiabilidade da propagação | Idem, com comparador: a distância temporal `g` sozinha. E com a pergunta adicional de se acrescenta algo ao escalar de calibração que o DiffRect já computa | Sem comparador, qualquer correlação positiva parece resultado; o que interessa é o ganho sobre o preditor trivial |
| **H7** | Plano V2 (Integrações A e B, avaliadas em E4b); ChatGPT, Experimento 4 | O mecanismo temporal melhora o desempenho final | O mecanismo temporal melhora **os pseudo-rótulos**, medido separadamente do desempenho final | O plano mede só o desfecho final, o que confunde "gerar melhor" com "aproveitar melhor" |
| **H8** | **Plano V2 (E5)** | O ganho cresce com o movimento, o que confirma o mecanismo alegado | Idem, sem alteração conceitual; muda apenas o instrumento de medida do movimento | O desenho é bom; o instrumento (E1) mede a coisa errada |
| **H9** | Plano V2 (E7 e contribuição C4), como item exploratório | O resíduo entre frames consecutivos pode indicar qualidade sem anotação | O ganho aparece na estabilidade temporal da saída e no escalar clínico C2–C4, não apenas em Dice por frame. Comparador: baseline por frame, medido nos 40.000 frames | A previsão mais nítida de um método temporal é sobre estabilidade; medi-la nos 40.000 frames dá muito mais poder estatístico que 1.000 frames anotados |
| **H10** | **Sua proposta nova (26/08)** | A dinâmica temporal pode ser aprendida por condicionamento, sem estimar transformação | Mantida como está | Não recomendo testá-la nesta dissertação; ver Parte VII |

**Observação sobre atribuição.** As hipóteses H1, H8 e H9 são do professor. As críticas que geraram H2 e H4 são suas. H3, H0b e a versão barata de H5 são minhas. H0, H6 e H7 existiam de forma difusa em mais de uma fonte e foram apenas reformuladas com comparador.

---

## Parte IV — Proveniência dos experimentos e movimentação na ordem

| ID | Nome | Origem | Corresponde a | Posição original | Nova posição | Natureza da mudança |
|---|---|---|---|---|---|---|
| **X1** | Gate de margem para semi-supervisão | Plano V2 (E4), reduzido | E4a parcial | Semanas 3–5 | **1º** | Antecipado e reduzido: 1 fold, 4 orçamentos, supervisionado + FixMatch apenas |
| **X2** | Decomposição do erro residual | **Minha** | — | — | **2º** | Novo |
| **X3** | Validade da aproximação rígida | Plano V2 (E2) | E2 | Semana 2 | **3º** | Mantido, com três correções internas |
| **X4** | Utilidade marginal do transporte | **Minha** | Substitui o Experimento 1 do ChatGPT | (Fase 1 do ChatGPT, mal formulado) | **4º** | Novo; é o gate principal |
| **X5** | Estimabilidade e confiabilidade da transformação | Plano V2 (E2b) + sua crítica nº 2 + ChatGPT Exp. 5 | E2b ampliado | Semana 2 (E2b) e Fase 4 (Exp. 5) | **5º** | Fundidos: escolha do estimador e estudo de confiabilidade são o mesmo conjunto de pares |
| **X6** | Auditoria de ordenação de qualidade | **Minha**, formalizando sua crítica nº 1 | — | — | **6º** | Novo |
| **X7** | Teto da compressão latente | Plano V2 (E3) | E3 | Semana 2 | **imediato** | Inalterado; custa uma hora, faça na semana 1 |
| **X8** | Trajetória: registro encadeado contra direto | **Minha**, versão barata da sua H5 e do Exp. 3 do ChatGPT | Substitui Exp. 3 | Fase 2 do ChatGPT | **7º** | Novo na forma; a pergunta é sua |
| **X9** | Integração B — consolidação temporal | Plano V2 | Integração B | Semanas 5–6 | **8º** | Mantido na mesma posição relativa; medição desdobrada em dois níveis |
| **X10** | Integração A — vizinho como perturbação | Plano V2 | Integração A | Semanas 6–7 | **9º** | Mantido; condicionado ao resultado de X6 |
| **X11** | Atribuição de mecanismo | Plano V2 (E5) | E5 | Semanas 8–9 | **10º** | Mantido; troca do instrumento de medida de movimento |
| **X12** | Estabilidade temporal como eixo de avaliação | Plano V2 (E7), promovido | E7 | Semanas 10–11, "se houver tempo" | **eixo contínuo** | Promovido de exploratório a métrica reportada em todos os experimentos de treino |
| **X13** | Ablações | Plano V2 (E6) | E6 | Semanas 10–11 | **11º** | Inalterado |
| — | E1 — quanto muda entre frames | Plano V2 | E1 | Semana 2 | **removido** | Instrumento inadequado; a variável passa a vir de X3 |
| — | Integração C | Plano V2 | Integração C | Semanas 9–10 | **removida do escopo planejado** | O próprio plano já a marca como primeira a cair |
| — | Experimento 1 do ChatGPT | ChatGPT | — | Fase 1 | **absorvido** | Sua parte válida já está em X3; sua parte faltante virou X4 |
| — | Experimento 2 do ChatGPT | ChatGPT | — | Fase 2 | **fora de escopo** | Comparação não controlada; ver Parte VII |

---

## Parte V — As três ordens lado a lado

A tabela mostra o que estaria acontecendo em cada momento sob cada plano. O eixo é a posição na sequência, não a semana, porque as durações diferem.

| Posição | Plano V2 (professor) | Sequência do ChatGPT | Reordenação proposta |
|---|---|---|---|
| 1 | Fundação: mapeamento, divisões, higienização | Baseline do DiffRect com métricas confiáveis | **X1 — existe margem para semi-supervisão?** + X7 (uma hora) |
| 2 | E1, E2, E2b, E3 | Exp. 1 — propagação em pares anotados | **X2 — qual erro precisa ser corrigido?** |
| 3 | Cache de transformações (caminho crítico) | Exp. 2 — explícito contra implícito | **X3 — a aproximação rígida vale?** |
| 4 | E4a — baselines por orçamento | Exp. 3 — ablação temporal | **X5 — a transformação é estimável e quando?** |
| 5 | Integração B | Exp. 4 — DiffRect contra DiffRect temporal | **X4 — o transporte acrescenta algo?** |
| 6 | Integração A | Exp. 5 — confiabilidade; refinamentos | **X6 — o transporte cabe na hierarquia?** |
| 7 | E4b — variantes nos mesmos orçamentos | — | **X8 — os intermediários acrescentam?** |
| 8 | E5 — ganho por estrato de movimento | — | Cache de transformações |
| 9 | Integração C | — | X9 — Integração B |
| 10 | E6 e E7 | — | X10 — Integração A |
| 11 | Consolidação | — | X11 — atribuição de mecanismo |
| 12 | — | — | X13 — ablações; consolidação |

### As cinco divergências e o motivo de cada uma

**1. O gate de margem sobe da posição 4 para a posição 1.**
No Plano V2, os baselines por orçamento de anotação (E4a) rodam nas semanas 3–5, depois do cache de transformações. O plano reconhece o risco de saturação — a seção "Sobre saturação" descreve exatamente o que fazer se o supervisionado com 1 rótulo por vídeo já empatar com o supervisionado com todos. Mas o experimento que detecta isso está posicionado depois da construção do cache, que é o item que o próprio plano identifica como caminho crítico e que custa horas de CPU sobre 40.000 frames.

A consequência prática: se a tarefa estiver saturada, o cache terá sido construído para nada e três semanas terão sido gastas. A versão reduzida do E4a — um fold, quatro orçamentos, dois métodos — responde a mesma pergunta a uma fração do custo e pode ser executada na GPU enquanto os experimentos de CPU rodam em paralelo. Não há motivo de dependência para ela vir depois; a ordenação original parece decorrer de o E4 estar categorizado como "resultado principal" em vez de "gate".

**2. A decomposição do erro (X2) entra na posição 2 e não existia.**
Nenhuma das fontes pergunta qual erro o modelo comete. O Plano V2 lista métricas (Dice, IoU, HD95, ASSD, taxa de saída inválida) mas não decompõe o erro em modos de falha. Isso importa porque um método temporal tem mecanismo forte para corrigir um tipo de erro específico — confusão de identidade entre vértebras adjacentes, que são estruturas quase idênticas em aparência — e mecanismo fraco para corrigir erro de área. Se a maior parte do erro residual for de contorno, a hipótese temporal está mirando onde não há alvo.

**3. A utilidade marginal (X4) entra na posição 5 e não existia em nenhum plano.**
Esta é a divergência mais importante. O Plano V2 mede, no E2, se a diferença entre duas máscaras anotadas é explicável por reposicionamento. Isso responde se o transporte é **geometricamente possível**. Não responde se ele é **útil**. As três integrações do plano são construídas sobre a suposição de que uma máscara transportada de um vizinho carrega informação que o modelo não tem para o frame alvo, e essa suposição nunca é testada.

O ChatGPT chega perto no Experimento 1, mas erra o comparador: mede o transporte contra a anotação de referência, não contra a predição do modelo. Um transporte com Dice 0,90 parece bom em absoluto e é inútil se o modelo sozinho entrega 0,93 naquele mesmo frame.

O experimento correto custa cerca de um dia de CPU, reaproveita o modelo de X1 e as transformações de X3, e pode encerrar toda a linha temporal antes do cache. Por relação entre custo e capacidade de decisão, é o melhor experimento do conjunto — e é o único que faltava.

**4. A auditoria de ordenação (X6) entra na posição 6 e formaliza a sua crítica.**
Sua objeção sobre misturar distribuições permaneceu conceitual nas três fontes. Ela é convertível em medida, e a medida é barata. O resultado tem consequência direta sobre como a Integração A deve ser implementada — não sobre se ela deve existir.

**5. O E1 é removido e a Integração C sai do escopo planejado.**
O E1 mede mudança de intensidade numa região dilatada em torno da vértebra. Numa deglutição, essa região contém o bolo de contraste passando, a mandíbula e a coluna de ar, que se movem independentemente da vértebra. Um frame com bolo denso e vértebra parada aparece como alto movimento. Como o E5 estratifica os resultados por essa variável, o experimento de atribuição de mecanismo — que é o melhor experimento do plano — corre o risco de estratificar por passagem do bolo. A variável correta é a magnitude da transformação ajustada, que o X3 já produz para pares anotados e o cache produz para o resto. Isso remove um experimento em vez de acrescentar.

A Integração C já está marcada no plano como opcional e como primeira a cair. Concordo, e proponho tratá-la como fora do escopo planejado desde já, para não ocupar espaço nas decisões intermediárias.

---

## Parte VI — Os experimentos em detalhe

### X1 — Gate de margem para semi-supervisão

**Origem:** Plano V2, E4a, reduzido e antecipado.
**Hipótese:** H0.

**Pergunta.** Neste dataset, com este regime de anotação, existe diferença de desempenho entre treinar com poucos rótulos e treinar com todos? Se não existir, não há espaço para nenhum método semi-supervisionado atuar, e todas as hipóteses temporais ficam sem objeto.

**Procedimento detalhado.**

1. Usar um único fold da validação cruzada. A versão de três folds fica para o E4 final; aqui o objetivo é detectar presença ou ausência de margem, não estimar o valor com precisão.
2. Definir quatro orçamentos de anotação: 1, 2 e 5 rótulos revelados por vídeo, e todos os rótulos disponíveis. Os rótulos não revelados **não são descartados**: os frames correspondentes entram no treino como frames sem anotação, com o rótulo simplesmente ignorado. Isso mantém constante o número de imagens vistas e isola o efeito da supervisão.
3. Treinar o segmentador supervisionado nos quatro orçamentos.
4. Treinar o FixMatch — o método semi-supervisionado mais simples do conjunto de baselines — no orçamento de 1 rótulo por vídeo. Isso mede quanto a semi-supervisão rende sem nenhuma engenharia temporal, e estabelece a referência que qualquer contribuição temporal precisa superar.
5. Repetir o orçamento de 1 rótulo por vídeo com três sementes de inicialização diferentes. Isso produz a **variação entre sementes**, que é a régua usada em quase todos os critérios de decisão deste documento, já que não há uma medida separada do ruído de anotação disponível (ver Parte IX).
6. Avaliar tudo no mesmo conjunto de teste, reportando Dice, IoU, ASSD e HD95, **separadamente para C2 e para C4**, com intervalo de confiança por reamostragem de pacientes.

**Saída.** Uma curva de quatro pontos por métrica, com banda de variação; o ganho isolado do FixMatch sobre o supervisionado no mesmo orçamento.

**Critérios de descarte, escritos antes de rodar.**

- Se a diferença entre "todos os rótulos" e "1 rótulo por vídeo" for menor que a variação entre sementes **em todas as quatro métricas**, o eixo "rótulos por vídeo" está saturado. A consequência é trocar o eixo de escassez para número de pacientes, exatamente como o Plano V2 já prevê: treinar com 10, 25 e 50 pacientes usando todos os rótulos de cada um. Escassez de diversidade anatômica é uma restrição diferente de escassez temporal, e provavelmente mais severa.
- Se o ganho do FixMatch sobre o supervisionado no mesmo orçamento for menor que cerca de 3 pontos de Dice — limiar de trabalho já adotado neste projeto, que é uma convenção e não uma derivação —, o eixo semi-supervisionado é fraco neste dataset e a contribuição precisa migrar para o que o X2 revelar.
- Se o supervisionado com todos os rótulos já atingir Dice muito alto com ASSD na ordem da resolução do pixel, a dissertação não pode ser sobre ganho de Dice. Isso não encerra o trabalho; muda o alvo, e o X2 define o novo alvo.

**Custo.** Seis a oito treinos curtos, três a quatro dias de GPU. Roda em paralelo com X2, X3 e X5, que são de CPU.

---

### X2 — Decomposição do erro residual

**Origem:** minha.
**Hipótese:** H0b.

**Pergunta.** Dos erros que o modelo comete, quanto é erro de área, quanto é erro de contorno e quanto é erro de identidade? A resposta define o alvo da contribuição, porque um método temporal tem mecanismo plausível para um desses tipos e não para os outros.

**Por que o erro de identidade merece tratamento separado.** C2, C3 e C4 são estruturas quase idênticas em aparência num exame de fluoroscopia lateral. Um modelo que segmenta cada frame isoladamente não tem nada que o impeça de atribuir a rótulo de C2 ao corpo vertebral de C3 num frame e acertar no frame seguinte. A identidade, porém, é a propriedade mais estável que existe ao longo de um vídeo: a vértebra que era C2 no frame 30 continua sendo C2 no frame 31. Ou seja: **este é o modo de falha para o qual a informação temporal tem o mecanismo de correção mais direto, e é o modo de falha ao qual o Dice é quase cego** — trocar os rótulos de duas vértebras de tamanho parecido altera pouco o Dice médio entre classes.

**Procedimento detalhado.** Usando o baseline supervisionado do X1:

1. Nos frames anotados do conjunto de teste, classificar cada frame em um ou mais dos seguintes modos:
   - **Erro de identidade:** um componente conexo predito como C2 sobrepõe majoritariamente a anotação de C4, ou vice-versa.
   - **Erro de detecção:** vértebra presente na anotação sem componente predito correspondente, ou componente predito sem contraparte anotada.
   - **Erro de contorno:** sobreposição correta em identidade, mas distância média entre superfícies acima de um valor que será fixado a partir da distribuição observada.
   - **Saída inválida:** predição com fragmentos desconexos atribuídos à mesma classe.
2. Reportar a fração de frames afetada por cada modo, e quanto cada modo custa no Dice médio. Uma forma direta de fazer isso: recalcular o Dice depois de corrigir manualmente cada modo isoladamente, e observar quanto ele sobe.
3. Rodar o mesmo modelo sobre os 40.000 frames e medir, **sem usar nenhuma anotação**, a taxa de troca temporal de identidade: rastrear cada componente conexo entre frames consecutivos por sobreposição espacial, e contar as transições em que o rótulo de classe atribuído ao mesmo componente rastreado muda. Isso mede instabilidade de identidade diretamente, e não depende de haver anotação.
4. Medir também a variação frame a frame da distância C2–C4, com um extrator de pontos fixo e documentado.

**Sobre a exclusão dos pontos anatômicos no Plano V2.** A seção 6 do plano descarta pontos anatômicos e a medida clínica derivada com o argumento de que extrair pontos de máscaras previstas é impreciso e a métrica acabaria medindo o extrator. O argumento está correto para usá-los como **métrica primária de qualidade de segmentação**. Mas excluí-los inteiramente tem dois custos: remove a única quantidade que interessa clinicamente ao INCA, e remove a métrica em que um método temporal faz a previsão mais nítida. A proposta é usá-los como métrica secundária, com extrator congelado, e reportar principalmente a **variação entre frames vizinhos**, não o valor absoluto — nessa forma, o erro do extrator é aproximadamente constante e se cancela na comparação entre métodos.

**Critérios de decisão.**

- Se a taxa de erro de identidade nos frames anotados e a taxa de troca temporal nos 40.000 frames forem materiais, o alvo da dissertação passa a ser identidade e estabilidade. Isso **reforça** a hipótese temporal e reformula a contribuição de "melhorar a segmentação" para "corrigir o erro que modelos por frame cometem por não terem memória".
- Se ambas forem desprezíveis e o erro for quase todo de contorno próximo à resolução do pixel, a tarefa está essencialmente resolvida e o trabalho precisa de outra pergunta. É melhor saber isso agora.

**Custo.** Cerca de um dia de CPU depois do X1. Nenhum treino adicional.

---

### X3 — Validade e alcance da aproximação rígida

**Origem:** Plano V2, E2. É o experimento conceitualmente mais importante do plano do professor e o desenho está correto.
**Hipótese:** H1.

**Pergunta.** Duas máscaras anotadas do mesmo vídeo, separadas por `g` frames: quanto da diferença entre elas é explicável por deslocamento, giro e mudança de escala, e quanto sobra?

**Procedimento base, como no plano.** Para cada par de frames anotados dentro do mesmo vídeo: registrar `g`; encontrar os quatro parâmetros da transformação que melhor sobrepõem a máscara A à máscara B, inicializando com a diferença entre centroides e a razão entre áreas; aplicar a transformação a A; medir Dice e distância média entre superfícies residuais em relação a B; plotar o resíduo contra `g`.

**Verificação obrigatória do procedimento.** Aplicar a uma máscara uma transformação conhecida — por exemplo, giro de 5 graus, deslocamento de 8 pixels, escala 1,03 — e verificar que o alinhador recupera esses parâmetros. Sem isso, um resíduo alto é ambíguo entre "a suposição é falsa" e "o alinhador não convergiu". Esta verificação já está no plano e deve ser mantida.

**Correção 1 — parametrizar também por deslocamento medido, não só por distância temporal.**

O plano parametriza a curva por `g`. Mas `g` é apenas um proxy do que interessa, que é quanto a vértebra se moveu — e é um proxy ruim, porque o paciente pode estar parado durante trinta frames e mover muito durante três. Você observou exatamente isso na sua fala de 26/08.

Há uma consequência prática importante. Com cerca de cinco anotações em posições aleatórias ao longo de duzentos frames, os pares disponíveis se concentram em valores grandes de `g`; o número de pares com `g` pequeno é reduzido. Parametrizar a curva por `g` deixa a região de intervalos curtos com poucos pares e intervalo de confiança largo — justamente a região onde a suposição rígida deveria valer.

Parametrizar por deslocamento medido resolve isso sem precisar de nada novo: **pares com `g` grande ocorridos durante um trecho em que o paciente estava parado têm deslocamento pequeno e povoam exatamente a faixa de interesse.** O procedimento é:

1. Para cada par, guardar a norma do vetor de translação em pixels e o valor absoluto do ângulo em graus, obtidos da transformação já ajustada.
2. Plotar o resíduo em função dessas quantidades, além de em função de `g`.
3. Comparar qual das duas parametrizações explica melhor o resíduo. Se o deslocamento explicar melhor que `g`, a janela operacional das integrações deve ser definida **por vídeo e por trecho**, a partir do movimento medido no cache, e não por um valor fixo de passo temporal. Isso é uma melhoria concreta sobre o plano, que hoje pretende fixar uma janela única.

**Correção 2 — transformações separadas para C2 e para C4.**

A coluna cervical é uma cadeia articulada. Sob flexão e extensão do pescoço, C2 e C4 não se deslocam como um bloco único. Ajustar uma transformação única para o conjunto das duas máscaras mistura o movimento articular entre as vértebras com o resíduo de mudança de projeção, e infla artificialmente o resíduo, o que pode levar a concluir que a suposição rígida falha quando na verdade o que falhou foi a suposição de corpo único.

Ajustar **uma transformação por vértebra**, reportar as duas curvas separadamente, e reportar também a curva do ajuste conjunto. A diferença entre as curvas separadas e a conjunta é uma medida direta do movimento articular entre C2 e C4, e é um resultado publicável por si só — é uma quantidade anatomicamente interpretável que ninguém precisou anotar para obter.

**Correção 3 — o E1 é substituído por esta saída.** As magnitudes de translação e rotação guardadas no passo 1 da Correção 1 são a variável de estratificação do X11. O E1 é descartado.

**Critério de descarte.** O plano é explícito e correto ao dizer que o resíduo soma duas coisas que não serão separadas: mudança real de projeção e diferença de traçado entre anotações. Concordo com essa escolha, e ela tem uma consequência que precisa ficar registrada: **não existe, sem anotações repetidas, uma medida isolada do ruído de anotação.** Portanto o critério não pode ser um limiar absoluto de resíduo. O critério defensável é comparativo: a suposição rígida é utilizável na faixa de deslocamento em que o resíduo após alinhamento for **menor que o erro que o próprio modelo comete naquele regime** — quantidade que o X4 mede diretamente. A convenção de trabalho já adotada no projeto (Dice residual em torno de 0,85) permanece útil como referência de ordem de grandeza, mas não como critério.

Se o resíduo for alto mesmo na faixa de deslocamento pequeno, as Integrações A, B e C saem do plano, e a curva vira o resultado negativo medido — que é reportável e é a contribuição C2 do plano do professor, independentemente do sinal do resultado.

**Custo.** Cerca de dois dias de CPU.

---

### X4 — Utilidade marginal do transporte

**Origem:** minha. Substitui o Experimento 1 do ChatGPT.
**Hipótese:** H3.

**Pergunta.** A máscara transportada de um frame vizinho acrescenta alguma informação que o modelo não tem para o frame alvo?

**Por que o X3 não responde isso.** O X3 mede se a diferença entre duas máscaras corretas é explicável por reposicionamento. É uma pergunta sobre geometria. Mesmo que a resposta seja plenamente positiva, ela não implica que transportar seja útil: o modelo pode já produzir, para o frame alvo, uma máscara melhor do que a transportada. As três integrações do Plano V2 dependem inteiramente de que a resposta a esta pergunta seja positiva, e nenhuma das fontes a formula.

**Procedimento detalhado.** Sobre os mesmos pares de frames anotados usados no X3, produzir **quatro estimativas** da máscara do frame B e medir todas contra a anotação de referência de B:

1. **Modelo sozinho.** A predição do baseline supervisionado do X1 sobre a imagem do frame B, sem nenhuma informação temporal. *Este é o comparador.*
2. **Transporte em condição-teto.** A máscara anotada de A transportada pela transformação ajustada diretamente sobre as duas máscaras anotadas — isto é, usando informação que o método real não teria. Como explicado na Parte I, essa é a melhor propagação rígida concebível: nenhuma estimativa feita a partir das imagens pode superá-la. Serve para descartar, não para prometer.
3. **Transporte realizável.** A máscara anotada de A transportada pela transformação estimada a partir das imagens, usando o melhor estimador identificado no X5. Este braço mede quanto se perde ao passar da condição ideal para a condição real.
4. **Seleção em condição-teto.** Para cada frame, escolher a melhor entre a estimativa 1 e a estimativa 2, usando a anotação de referência para decidir qual é a melhor. Nenhuma regra de fusão realizável pode superar isso, porque nenhuma regra tem acesso à resposta. **Este braço mede o teto de toda a família de métodos que combinam predição própria com informação temporal.**

Estratificar todos os resultados por magnitude de deslocamento (do X3), por `g`, por classe (C2 e C4 separadamente) e pela confiança do modelo no frame alvo. Reportar também a taxa de erro de identidade de cada estimativa, conforme a classificação do X2.

**Critérios de descarte, escritos antes de rodar.**

- **Estimativa 2 pior que estimativa 1 em todos os estratos** → o transporte rígido nunca acrescenta nada neste dataset com este modelo. A linha de propagação está encerrada. O trabalho segue como estudo de eficiência de anotação, e a curva do X3 é reportada como o resultado negativo medido.
- **Diferença entre estimativa 4 e estimativa 1 menor que a variação entre sementes do X1** → mesmo uma regra de fusão perfeita não ganharia nada. Não construir mecanismo de ponderação, de confiança ou de gating: não há o que ponderar. **Este é o critério mais duro do conjunto e o que economiza mais tempo**, porque encerra a linha antes de qualquer engenharia.
- **Estimativa 2 melhor que estimativa 1 apenas em estratos específicos** — por exemplo em deslocamento alto, ou em frames de baixa confiança do modelo, ou em frames com erro de identidade → a contribuição está localizada nesses estratos, e o método deve ser **seletivo** em vez de sempre ativo. Isso reformula as Integrações A e B: de "adicionar informação temporal a todo frame" para "usar informação temporal onde o modelo por frame falha". É uma proposta mais barata de implementar, mais fácil de defender e com mecanismo explicitável.
- **Estimativa 3 muito pior que estimativa 2** → o gargalo é a estimação da transformação, não a hipótese temporal. Essa é exatamente a sua crítica nº 2, e a decisão passa para o X5.

**Custo.** Cerca de um dia de CPU. Reaproveita o modelo do X1 e as transformações do X3. Os braços 1, 2 e 4 não dependem do X5 e podem ser executados antes dele; apenas o braço 3 precisa do estimador escolhido.

---

### X5 — Estimabilidade e confiabilidade da transformação

**Origem:** Plano V2 (E2b) + sua crítica nº 2 + Experimento 5 do ChatGPT, fundidos.
**Hipóteses:** H2 e H6.

**Pergunta em duas partes.** Primeira: a transformação pode ser estimada sem usar anotação, com erro pequeno o bastante? Segunda: é possível saber, sem anotação, quando essa estimativa é confiável?

**Parte 1 — escolha do estimador, como no plano.** Nos mesmos pares anotados, estimar a transformação de três formas: (i) usando apenas a imagem na região de interesse; (ii) usando apenas as máscaras; (iii) usando a imagem com a máscara servindo de peso espacial na medida de semelhança. Aplicar cada uma à máscara de A e medir o Dice contra a máscara de B. Vence quem reconstruir melhor.

O plano justifica a preferência pela imagem com dois argumentos que considero corretos e que vale registrar aqui, porque eles não são óbvios: primeiro, nos frames sem anotação não existe máscara, existe predição, e estimar a transformação a partir da predição para depois supervisionar a predição não traz informação nova — o sinal herda o erro do modelo; segundo, a máscara de um corpo vertebral é um blob quase convexo e liso, do qual se extraem bem o centroide e a área mas mal a rotação, enquanto a imagem tem textura interna (padrão trabecular, borda cortical, espaços discais) que fixa o ângulo.

**Parte 2 — reportar a cauda, não a mediana.** Este é o acréscimo obrigatório sobre o E2b. Um estimador com boa mediana e falha grave em uma fração dos pares é inutilizável quando aplicado a 40.000 frames sem qualquer supervisão, porque as falhas entram no treino como supervisão errada. Reportar: a distribuição completa do erro, o percentil 90, e a **taxa de falha catastrófica**, definida como a fração de pares em que o resultado do transporte é pior do que simplesmente copiar a máscara de A sem transportar. Essa definição é operacional: falha catastrófica é quando estimar a transformação piora em vez de melhorar.

**Parte 3 — estudo de preditores de confiabilidade (H6).** Para cada par, registrar quantidades calculáveis **sem nenhuma anotação**: a distância temporal `g`; a correlação entre as regiões de interesse dos dois frames; a magnitude da transformação estimada; o escore de convergência do próprio registro; a distância entre as máscaras *previstas* pelo modelo nos dois frames; a entropia média da predição em cada frame. Ajustar um modelo simples — regressão logística ou árvore rasa — para prever o erro do transporte a partir dessas quantidades.

O que reportar: a correlação de cada preditor com o erro real, e sobretudo o **ganho sobre `g` sozinho**. Este é o ponto da hipótese. Se nenhum preditor superar a distância temporal, então não vale construir um mecanismo de confiança: basta usar uma janela fixa, e a ideia de ponderação por confiabilidade não é contribuição.

**Observação conceitual sobre H6.** Conforme já registrado neste projeto, usar a semelhança entre frames vizinhos como indicador é estruturalmente análogo ao escalar de calibração que o DiffRect já computa entre suas duas versões perturbadas. A pergunta de contribuição, portanto, não é "um indicador de movimento funciona?", e sim "ele carrega informação **além** da que o método já extrai?". Se o estudo de preditores mostrar que não, essa parte da ideia deve ser descartada como contribuição, ainda que continue útil como detalhe de implementação.

**Custo.** Cerca de um dia de CPU.

---

### X6 — Auditoria de ordenação de qualidade

**Origem:** minha, formalizando a sua crítica nº 1.
**Hipótese:** H4.

**Pergunta.** A máscara transportada ocupa uma posição estável e ordenada na hierarquia de qualidade que o DiffRect usa para aprender a corrigir pseudo-rótulos?

**Por que isso importa e o que sua crítica identificou corretamente.** O DiffRect aprende um caminho de correção que vai de uma versão pior para uma versão melhor. O método pressupõe uma ordem: a predição obtida sob perturbação forte é pior que a obtida sob perturbação fraca, que por sua vez é pior que a anotação de referência. Se a máscara transportada do vizinho for, em uma fração relevante dos frames, **melhor** que o alvo de perturbação fraca, então a Integração A não estará adicionando uma perturbação: estará ensinando o modelo a caminhar na direção errada nesses frames. Sua objeção de que "duas distribuições diferentes estão sendo colocadas na mesma hierarquia" era exatamente isso; o que faltava era transformá-la em número.

**Procedimento detalhado.** Nos frames anotados, onde é possível medir qualidade, usando o modelo do X1:

1. Para cada frame, computar três objetos e a qualidade de cada um em relação à anotação de referência:
   - a predição obtida sob perturbação fraca;
   - a predição obtida sob perturbação forte;
   - a máscara do frame vizinho transportada para o frame alvo.
2. Plotar as três distribuições de qualidade sobrepostas.
3. Medir duas taxas de inversão: a fração de frames em que a máscara transportada é melhor que a versão de perturbação fraca, e a fração em que ela é pior que a versão de perturbação forte.
4. Repetir por faixa de deslocamento e por faixa de `g`, para verificar se a ordenação é estável em toda a janela operacional ou apenas em parte dela.

**Critérios de decisão.**

- Ordenação estável, com sobreposição limitada entre as distribuições → a Integração A é admissível como especificada no Plano V2, e sua objeção fica resolvida por medida.
- Taxa de inversão material → a Integração A precisa de uma de duas correções: filtrar os pares pelo preditor de confiabilidade do X5, usando apenas transportes cuja confiabilidade estimada seja alta; ou tratar o transporte como **fonte separada** — par próprio, peso próprio, possivelmente cabeça de saída própria — em vez de encaixá-lo na hierarquia existente. Note que a segunda opção é precisamente a reformulação que você propôs na sua crítica, e que ela passaria a estar apoiada em medida em vez de intuição.

**Custo.** Cerca de meio dia, depois de X1 e X5.

---

### X7 — Teto da compressão latente

**Origem:** Plano V2, E3. Inalterado.

**Procedimento.** Pegar as máscaras anotadas, passá-las pelo compressor de rótulos do DiffRect, descomprimir, e medir Dice e distância entre superfícies entre a original e a reconstruída. Não treina nada.

**Interpretação.** Esse número é o limite superior do DiffRect: nenhuma correção recupera o detalhe que a compressão descartou. A comparação correta, como o plano estabelece, **não** é com o supervisionado, e sim com o desempenho dos outros métodos semi-supervisionados. Se o teto ficar abaixo do que Mean Teacher ou FixMatch já entregam, o DiffRect não tem como vencer e as integrações devem ser aplicadas sobre o método mais simples, sem que o cronograma mude.

**Custo.** Cerca de uma hora. Por esse custo, deve ser feito imediatamente, antes de qualquer discussão de prioridade.

---

### X8 — Trajetória: registro encadeado contra registro direto

**Origem:** minha, como versão barata da pergunta que é sua (fala de 26/08) e que o ChatGPT formulou como Experimento 3.
**Hipótese:** H5.

**Pergunta.** Os frames intermediários entre dois frames anotados acrescentam informação sobre como a estrutura se moveu, além do que se obtém olhando apenas os dois extremos?

**Por que esta versão e não a versão com rede.** Testar a hipótese treinando um modelo condicionado à sequência custa semanas e confunde duas coisas: se a trajetória contém informação, e se a arquitetura escolhida consegue extraí-la. Um resultado negativo não distinguiria as duas. A versão geométrica isola a primeira pergunta e custa horas.

**Procedimento detalhado.** Para cada par anotado (A, B) separado por `g` frames:

1. **Direto.** Estimar a transformação de A para B em um único passo, transportar a máscara de A, medir contra a de B.
2. **Encadeado.** Estimar a transformação entre cada par de frames consecutivos ao longo do caminho, compor todas essas transformações, transportar a máscara de A pela composição, medir contra a de B.
3. **Encadeado esparso.** O mesmo, mas compondo saltos de tamanho intermediário — por exemplo cinco frames por passo — em vez de saltos de um frame.
4. Plotar as três curvas contra `g` e contra a magnitude de deslocamento total.

**Interpretação.** As duas alternativas falham por motivos opostos. O registro direto falha quando o deslocamento entre os dois frames excede o alcance de captura do algoritmo, isto é, quando as duas imagens estão longe demais para que a otimização encontre o alinhamento correto. O encadeado falha por acúmulo de deriva: cada estimativa tem um pequeno erro, e compor trinta estimativas acumula trinta erros. **O resultado do experimento é a existência e a posição do cruzamento entre as duas curvas.** Se o encadeado superar o direto acima de algum valor de deslocamento, os intermediários carregam informação geométrica utilizável, e isso é evidência concreta a favor da sua intuição sobre a trajetória — obtida em um dia, sem arquitetura nova. O encadeado esparso serve para localizar o compromisso entre os dois modos de falha.

**Ressalva que precisa ficar registrada.** Acúmulo de deriva em composição de registros é um fenômeno conhecido e é plausível que o encadeado perca em toda a faixa. Isso não falsifica H5 na sua versão aprendida — um modelo poderia extrair da sequência algo que a composição rígida não extrai, por exemplo aparência e oclusão em vez de apenas geometria. Mas significa que, se o encadeado perder, você não terá evidência barata a favor, e a versão cara passaria a ser uma aposta sem apoio prévio. É exatamente a situação em que a decisão de escopo da Parte VII deve ser respeitada.

**Custo.** Cerca de um dia de CPU. Reaproveita o estimador do X5.

---

### X9 — Integração B: consolidação temporal dos pseudo-rótulos

**Origem:** Plano V2. A ordem "B antes de A" do plano está correta e deve ser mantida: B é majoritariamente pré-processamento e não exige mexer na arquitetura nem na função de custo, portanto valida a hipótese temporal mais cedo e com menos risco de implementação.
**Hipótese:** H7.

**Procedimento, como no plano.** Para um frame alvo, prever a máscara de cada vizinho da janela, transportar todas para o referencial do alvo usando as transformações do cache, combiná-las por média ponderada ou voto por pixel, e usar o resultado como pseudo-rótulo do frame alvo. A janela deve ser definida pela faixa de deslocamento identificada no X3, e não por um valor fixo escolhido a priori.

**Modificação necessária: medir em dois níveis separadamente.** O plano avalia o efeito da Integração B pelo desempenho final da segmentação. Isso mistura duas perguntas que podem ter respostas diferentes:

1. **A consolidação produz pseudo-rótulos melhores?** Gerar pseudo-rótulos consolidados para os frames que possuem anotação, sem usar essa anotação no processo, e medir contra ela. Comparar com o pseudo-rótulo obtido do frame isolado. Isso responde H7 sem treinar nada, e é barato.
2. **Pseudo-rótulos melhores produzem um modelo melhor?** Treinar com os pseudo-rótulos consolidados e avaliar no teste.

Se o primeiro melhorar e o segundo não, o gargalo está em como o pseudo-rótulo é consumido pelo treino, não em como ele é gerado. Isso é um diagnóstico útil e direciona o passo seguinte, ao contrário de um resultado agregado que apenas diz "não funcionou".

**Pré-condição.** Só executar se o X4 tiver mostrado utilidade marginal em algum estrato.

---

### X10 — Integração A: o frame vizinho como perturbação real

**Origem:** Plano V2.
**Hipótese:** H7.

Implementar como no plano se o X6 mostrar ordenação estável. Se o X6 mostrar inversão material, implementar com o transporte filtrado pelo preditor de confiabilidade do X5, ou como fonte separada da hierarquia existente. Executar depois da Integração B, como o plano estabelece, porque mexe no laço de treino e é mais invasiva.

---

### X11 — Atribuição de mecanismo

**Origem:** Plano V2, E5. **É o melhor experimento do plano do professor** e deve ser mantido integralmente.
**Hipótese:** H8.

**Por que ele é bom.** Um ganho médio não distingue entre duas explicações. Se o método temporal superar a suavidade temporal — a alternativa simples, que apenas exige que as máscaras de frames vizinhos sejam parecidas — isso pode ser porque o método permite o movimento real que a suavidade pune indevidamente, ou simplesmente porque é uma restrição mais frouxa que atrapalha menos. As duas explicações produzem a mesma média e previsões diferentes quando os frames de teste são separados por quanto a vértebra realmente se moveu: na primeira, a diferença cresce com o movimento; na segunda, permanece constante.

Isso importa porque determina o que pode ser afirmado no artigo. Com a primeira explicação, é possível dizer a outro grupo quando o método vai funcionar nos dados dele. Com a segunda, só é possível dizer que funcionou neste dataset.

**Procedimento, como no plano.** Treinar três modelos no mesmo orçamento e mesma divisão: sem termo temporal; com suavidade temporal; com o mecanismo temporal proposto, usando exatamente os mesmos pares de frames nos dois últimos. Estratificar os frames de teste por movimento em três faixas. Reportar Dice e distância entre superfícies das três variantes em cada faixa. Calcular, por paciente, a diferença entre o proposto e a suavidade em cada faixa, e verificar por intervalo de confiança se essa diferença cresce da faixa baixa para a alta.

**Manter do plano:** escrever a previsão antes de rodar; estatística pareada por paciente; e a pré-condição de que existam frames de alto movimento em quantidade suficiente.

**Única correção:** estratificar pela magnitude da transformação medida (X3 e cache), não pela mudança de intensidade do E1, pelas razões da Parte V, item 5.

**Relevância adicional.** A suavidade temporal não é um baseline inventado para o experimento: é o mecanismo empregado por trabalhos publicados em cenário próximo ao seu, o que torna essa comparação obrigatória e não opcional (ver Parte VIII).

---

### X12 — Estabilidade temporal como eixo de avaliação

**Origem:** Plano V2, E7 e contribuição C4, promovidos de exploratório a eixo reportado.
**Hipótese:** H9.

**Por que promover.** A previsão mais nítida de qualquer método com informação temporal é sobre a estabilidade da saída ao longo do vídeo, não sobre o Dice de um frame isolado. Além disso, a estabilidade é mensurável nos 40.000 frames sem nenhuma anotação, o que dá ordens de magnitude mais poder estatístico do que os 1.000 frames anotados — e poder estatístico é a restrição real deste projeto, dado que a unidade de análise correta é o paciente e não o frame.

**Métricas.** Resíduo médio após alinhamento entre frames consecutivos; taxa de troca temporal de identidade (definida no X2); desvio-padrão da distância C2–C4 entre frames vizinhos.

**Ressalva metodológica obrigatória.** Estabilidade temporal é trivialmente manipulável: um modelo que prediz sempre a mesma máscara, independentemente da imagem, é perfeitamente estável e completamente inútil. Essas métricas **nunca** devem ser reportadas isoladamente, e a regra de leitura deve ser fixada antes: ganho de estabilidade só conta se o Dice e a distância entre superfícies não piorarem.

---

### X13 — Ablações

**Origem:** Plano V2, E6. Inalterado.

Uma variável por vez, mesmo orçamento, mesma divisão, um fold: qual integração produz o ganho (só A, só B, as duas); tamanho de janela, com três valores dentro e um fora da faixa indicada pelo X3, onde o esperado é piorar fora da faixa, o que liga a medição ao resultado; peso do termo temporal, com três valores, porque um método que só funciona em um valor específico é coincidência e não método.

---

## Parte VII — O que fica fora de escopo e sob que condição volta

### H10 — condicionamento temporal implícito

**Recomendação: fora de escopo nesta dissertação.** Três razões, em ordem de peso.

**Primeira.** Não é comparável às Integrações A e B por experimento controlado. As duas alternativas diferem simultaneamente em arquitetura, número de parâmetros, formato de entrada e regime de treino. Um resultado favorável ao condicionamento não poderia ser atribuído ao mecanismo temporal, e o Experimento 2 do ChatGPT, que propõe exatamente essa comparação na Fase 2, não decide nada apesar de custar semanas.

**Segunda.** O teto já é medido de graça pelo X4. Se o transporte em condição-teto — a melhor propagação rígida concebível — já não superar o modelo por frame, então não há informação temporal geométrica a extrair, e um modelo aprendido não teria o que aproveitar. Se o transporte em condição-teto superar, mas o transporte realizável ficar próximo dele, o ganho possível do condicionamento se reduz a evitar as falhas de estimação, cuja fração o X5 quantifica exatamente. Em ambos os casos, a decisão sobre o condicionamento sai de um experimento que já será feito por outro motivo.

**Terceira.** Existe evidência interna contrária que precisa ser recuperada antes de qualquer investimento. A seção 11 do Plano V2 exclui "fusão de frames vizinhos na entrada da rede — já testada neste projeto sem ganho". Esse é o teste mais próximo de H10 que já existe neste projeto, e o resultado foi negativo. Recomendo recuperar o protocolo dele: qual arquitetura, quantos frames de contexto, qual orçamento de anotação, e com que método de fusão. Se o protocolo for razoável, é evidência contra H10 e deve ser citada na dissertação como resultado negativo próprio, o que é uma contribuição legítima. Se o protocolo tiver limitações claras, o negativo é fraco e pode ser revisitado — mas com escopo definido.

**Condição de reentrada.** H10 volta ao escopo se, e somente se, o X4 mostrar folga grande entre a seleção em condição-teto e o modelo sozinho — indicando que há informação temporal a extrair — **e** o X5 mostrar taxa alta de falha catastrófica do estimador — indicando que a via geométrica não consegue extraí-la. Só nessa combinação o condicionamento tem justificativa prévia em vez de ser aposta arquitetural.

### Sobre o paper do guidewire

Verifiquei a referência que você citou: Pan et al., *Label-Efficient Data Augmentation with Video Diffusion Models for Guidewire Segmentation in Cardiac Fluoroscopy*, arXiv:2412.16050. O trabalho existe e o domínio é próximo (fluoroscopia). Mas há uma diferença que precisa ficar explícita antes de usá-lo como fundamento: <cite index="5-1">o mecanismo é a geração de vídeos de fluoroscopia rotulados para aumentar os dados de treino, modelando separadamente a distribuição de cena e a de movimento — primeiro gerando imagens com o objeto posicionado segundo uma máscara de entrada, depois gerando progressivamente os frames seguintes com uma estratégia de consistência entre frames</cite>. Não é retificação de pseudo-rótulo.

Isso cria uma tensão com o próprio Plano V2, que exclui explicitamente síntese generativa de frames na seção 11. Transferir a ideia implicaria uma de duas coisas: adotar síntese generativa, reabrindo um escopo já cortado por razões de prazo; ou reinterpretar a estratégia de consistência entre frames como mecanismo de condicionamento, o que é uma operação diferente e precisaria de justificativa própria. Recomendo usar o paper como trabalho relacionado, não como sustentação da hipótese.

### Outros itens fora de escopo

- **E1 como está** — instrumento inadequado; a variável de movimento passa a vir do X3.
- **Integração C** — o próprio plano já a marca como primeira a cair; proponho tratá-la como fora do escopo planejado desde já.
- **Qualquer discussão sobre qual integração é melhor antes do X4** — a pergunta "A ou B?" só é respondível depois de saber se existe utilidade marginal a explorar, e em quais estratos.

---

## Parte VIII — Trabalho relacionado que condiciona a alegação de novidade

Duas alegações do plano estão expostas e dependem de uma verificação que ainda não foi feita: a contribuição C2 (medir o domínio de validade da aproximação rígida em vídeo médico) e a alegação implícita de que quase ninguém usa temporalidade especificamente para retificar pseudo-rótulos neste regime.

**Verificados por mim e reais:**

- Zheng et al., *Reducing Annotation Burden: Exploiting Image Knowledge for Few-Shot Medical Video Object Segmentation via Spatiotemporal Consistency Relearning* (MICCAI 2024; arXiv:2503.14958). O regime é muito próximo do seu: <cite index="7-1">anotações de apenas alguns frames de vídeo, com um enquadramento em duas fases — primeiro um modelo de segmentação few-shot aprendido a partir de imagens rotuladas, depois um estágio de *relearning* com consistência espaço-temporal que impõe consistência entre frames consecutivos</cite>. **Isto é o baseline de suavidade temporal do X11, publicado, no domínio médico.** Precisa ser citado e, idealmente, comparado. Diferença relevante: o método depende de imagens rotuladas de outra fonte para a primeira fase, o que você não tem.
- Pan et al., arXiv:2412.16050 (guidewire, discutido acima).
- Xi, Ma & Zhuang, *Few-Shot Video Object Segmentation in X-Ray Angiography Using Local Matching and Spatio-Temporal Consistency Loss*, Neural Networks, 2026. Fluoroscopia, poucos frames anotados, perda de consistência espaço-temporal.

**Citados no documento `Explorar-métodos-alternativos.md` e ainda não verificados por mim.** Os links vieram de saída de modelo de linguagem e podem estar incorretos ou desatualizados; não usar sem checar: Nilsson e Sminchisescu (*Gated Recurrent Flow Propagation*); o repositório de segmentação de ecocardiograma temporalmente consistente; os surveys de pseudo-rótulos e de segmentação de objetos em vídeo.

**Escopo recomendado da busca.** Prazo fechado de três dias, em paralelo com os experimentos de CPU, com duas perguntas apenas: alguém já mediu resíduo rígido em função da distância ou do deslocamento em vídeo médico? Alguém já usou propagação temporal especificamente para **retificar** pseudo-rótulos, em vez de para regularizar consistência? Se a primeira já existe, C2 deixa de ser contribuição e vira metodologia citada. Se a segunda já existe, a novidade precisa ser reformulada. O objetivo é decidir sobre duas alegações, não mapear o campo.

---

## Parte IX — Consequências de trabalhar apenas com a anotação existente

Todos os experimentos deste documento usam exclusivamente os cerca de 1.000 frames já anotados. Isso tem duas consequências que precisam ficar registradas, porque afetam critérios de decisão.

**Primeira: não existe medida isolada do ruído de anotação.** O resíduo medido no X3 soma mudança real de projeção e diferença de traçado entre anotações, e separá-las exigiria anotações repetidas. O Plano V2 já toma essa decisão explicitamente e eu concordo com ela: o que interessa operacionalmente é a soma, porque é ela que determina até onde a suposição rígida erra menos do que o modelo já erra. A consequência é que **nenhum critério de decisão neste documento usa limiar absoluto de resíduo**. As réguas disponíveis são três, todas mensuráveis sem anotação nova: a variação entre sementes de treino, a variação entre folds, e o erro do próprio modelo no mesmo regime.

**Segunda: a região de intervalos curtos entre pares anotados é pouco povoada.** Como as anotações estão em posições aleatórias, há poucos pares com `g` pequeno. A solução adotada não requer anotação nova: parametrizar a curva do X3 por **deslocamento medido** além de por distância temporal, o que faz com que pares distantes ocorridos durante trechos parados povoem a faixa de pequeno deslocamento. Isso também produz um resultado melhor do que o originalmente planejado, porque permite definir a janela operacional por trecho de vídeo, adaptativamente, em vez de fixar um passo temporal único para todo o dataset.

---

## Parte X — Cronograma de decisão

| Semana | O que roda | Gate |
|---|---|---|
| 1 | X7 (uma hora); preparação dos conjuntos de pares para X3 e X5 | X7 pode trocar o veículo de DiffRect para Mean Teacher ou FixMatch |
| 2 | **GPU:** X1 · **CPU:** X3, X5, X8 · busca bibliográfica com prazo fechado | X1: existe margem? X3: a rigidez vale? |
| 3 | X2, X4, X6 — todos reaproveitam o modelo de X1 e as transformações de X3 e X5 | **X4 é o gate principal: o transporte acrescenta algo?** |
| 3–4 | Se X4 passar: cache de transformações para todos os vídeos | Caminho crítico, como no plano original |
| 5–7 | X9 — Integração B, medida nos dois níveis | B rendeu acima do baseline? |
| 7–9 | X10 — Integração A, condicionada ao X6; X11 — atribuição de mecanismo | O ganho cresce com o movimento? |
| 10–11 | X13 — ablações; consolidação de X12 sobre todos os modelos treinados | — |
| 12–13 | Tabelas, figuras, escrita | — |

**O que muda em relação ao Plano V2.** As três primeiras semanas passam a conter seis pontos de decisão em vez de três, todos antes de qualquer treino pesado ou construção de cache. Três deles são novos e custam, somados, cerca de dois dias e meio de CPU. Em troca, ao fim da semana 3 estarão respondidas seis perguntas: se existe margem para semi-supervisão; qual erro precisa ser corrigido; se a aproximação rígida vale e em que faixa de deslocamento; se o transporte acrescenta algo ao que o modelo já faz; se a transformação é estimável e quando confiar nela; e se o transporte cabe na hierarquia de qualidade do DiffRect. Nenhuma dessas respostas depende de escolher entre Integração A, Integração B ou condicionamento — e todas as seis restringem essa escolha.

---

## Parte XI — Separação entre evidência e inferência

**É evidência, verificada por mim nesta sessão:** a existência e o conteúdo dos três papers listados na Parte VIII; especificamente, que o trabalho do guidewire é sobre síntese generativa para aumento de dados e não sobre retificação de pseudo-rótulo, e que o trabalho de Zheng et al. usa consistência entre frames consecutivos em regime de poucos frames anotados.

**É conteúdo dos documentos do projeto, não verificado independentemente por mim:** as descrições do funcionamento interno do DiffRect (compressão latente de rótulos, hierarquia de perturbação fraca e forte, retificação operando apenas sobre o rótulo, escalar de calibração); o resultado negativo prévio sobre fusão de frames vizinhos na entrada da rede; o número aproximado de vídeos, frames e anotações.

**É inferência minha, e é o que os experimentos testam:** que a tarefa provavelmente satura em Dice; que o erro residual é provavelmente dominado por identidade e contorno; que a distância temporal é um proxy ruim de movimento; que a região de interesse dilatada do E1 é dominada pelo bolo e pela mandíbula; que C2 e C4 exigem transformações separadas por serem parte de uma cadeia articulada; que a comparação entre transporte explícito e condicionamento aprendido não é controlável dentro do prazo.

**São propostas minhas sem precedente nos documentos:** os experimentos X2, X4, X6 e X8; a promoção do X12 a eixo de avaliação; a remoção do E1; a reparametrização do X3 por deslocamento medido; e a separação de transformações por vértebra.
