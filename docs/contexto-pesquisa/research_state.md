# research_state.md

## Problema atual

Segmentação semissupervisionada de vértebras cervicais em vídeos VFSS do INCA com poucos frames anotados por vídeo, muitos frames nunca anotados e dependência temporal entre frames.

O problema atual não é apenas maximizar Dice. É entender quais fontes de informação explicam os ganhos sob orçamento limitado de rótulos e se a estrutura temporal pode ser convertida em supervisão útil além de SSL frame-based.

## Objetivo

Construir uma evidência experimental sólida sobre eficiência de rótulos no VFSS e, a partir dela, determinar se existe margem e justificativa para uma contribuição baseada em temporalidade explícita, propagação, consistência, representação ou decomposição da tarefa.

## O que já foi feito

- Benchmark supervisionado de arquiteturas e exploração de MedSegDiff, SDSeg e SDSeg+TCM.
- Port dos wrappers de SSL validado no ACDC.
- X1 por fração de pacientes com supervised, Mean Teacher, FixMatch e DiffRect.
- X1B com políticas `global`, `same_video`, W=5 e W=10 para supervised, FixMatch e Mean Teacher.
- X1 por 1, 2, 3 e 5 frames anotados por vídeo, mantendo 69 pacientes e 165 vídeos.
- Variante `multiclass_all` com C2 a C8.
- Variante `binary_all` com todas as vértebras fundidas em uma classe.
- Auditoria técnica do repositório, configs, logs, checkpoints, CSVs e dataset.

## Principais evidências

1. Dataset atual verificado: 961 frames anotados, 232 vídeos e 114 pacientes.
2. Split dos experimentos X1: treino 690 frames / 69 pacientes / 165 vídeos; validação 99 / 25 / 25; teste 172 / 20 / 42.
3. X1 por paciente mostra ganhos SSL em várias células, mas altera simultaneamente rótulos, pacientes, vídeos e diversidade.
4. X1 por frames por vídeo mostra ganho do baseline supervisionado de Dice macro C2/C4 0,7366 com 1 frame para 0,8074 com 5 frames.
5. SSL frame-based não apresenta vantagem consistente em todos os regimes de frames por vídeo.
6. X1B não permite conclusão confiável sobre política temporal: o baseline supervisionado varia fortemente entre políticas e não há múltiplas seeds.
7. `multiclass_all` degrada fortemente em classes inferiores; C8 não existe em validação ou teste.
8. `binary_all` é muito mais fácil e quase satura com 1 frame por vídeo: supervised 0,8816 em 1 fpv e 0,8867 em 5 fpv.
9. Todos os experimentos VFSS têm uma única seed, 42.
10. Existem reexecuções com mesma seed e `hparams.yaml` que produziram resultados diferentes.

## Interpretação atual

X1 por paciente e X1 por frames por vídeo medem fenômenos diferentes. O primeiro mistura escassez de rótulos com diversidade interpaciente; o segundo mede melhor a utilidade marginal de anotações adicionais dentro dos mesmos vídeos.

Frames adicionais do mesmo vídeo ainda ajudam na tarefa C2/C4, mas os métodos SSL frame-based atuais não convertem de forma estável o pool não rotulado em ganho adicional.

A variante binária muda a interpretação do problema multiclasse: delinear a região vertebral é relativamente fácil, enquanto distinguir identidades e lidar com classes raras parece responder por parte importante da dificuldade de C2 a C8. Isso não prova que um pipeline em duas etapas seja superior.

A hipótese de temporalidade continua aberta porque nenhum experimento atual compara, de forma controlada e reprodutível, um mecanismo temporal explícito com os baselines frame-based.

## Hipóteses

- H1: diversidade de pacientes pode valer mais do que aumentar a densidade de frames anotados dos mesmos vídeos.
- H2: SSL frame-based convencional pode subutilizar informação temporal.
- H3: consistência, correspondência, memória ou propagação temporal podem fornecer sinal adicional útil.
- H4: frames próximos são parcialmente redundantes, com redundância particularmente alta na tarefa binária.
- H5: transporte rígido pode ser útil localmente, mas a questão relevante é utilidade para o treino, não apenas estimabilidade do movimento.
- H6: identidade, frequência e visibilidade explicam parte relevante da degradação do multiclasse amplo.
- H7: pré-treinamento ou representação autossupervisionada pode reduzir overfitting, ainda sem evidência experimental própria.

## Questões abertas

- Quanto dos ganhos do X1 vem de diversidade de pacientes e quanto vem de SSL?
- Qual é o valor marginal de um frame adicional em função da distância temporal?
- Um mecanismo temporal explícito supera SSL frame-based sob o mesmo orçamento?
- Qual é o horizonte útil de propagação ou consistência temporal?
- A queda de C5 a C7 decorre principalmente de frequência, visibilidade, identidade ou qualidade de anotação?
- Uma etapa binária seguida de identificação supera o multiclasse direto?
- Qual papel final, se algum, diffusion models devem ter?
- Qual é a relação experimental final com o trabalho anterior do colega e com a literatura temporal?

## Branches ativas

- SSL frame-based como baseline.
- SSL temporal explícito.
- Consistência temporal.
- Propagação de máscaras e correspondência.
- Transformação rígida como mecanismo específico, não como hipótese geral.
- Binário seguido de identificação.
- Pré-treinamento / self-supervised / foundation models como branch complementar.
- Diffusion models como branch secundária.

## Decisões pendentes

- hipótese central do artigo/dissertação;
- contribuição metodológica final;
- foco definitivo em C2/C4 ou conjunto maior de vértebras;
- pipeline multiclasse direto ou binário + identidade;
- método temporal explícito a ser testado;
- papel final de pré-treinamento;
- protocolo estatístico final e número de seeds;
- posição final da medida clínica C2-C4 no trabalho.

## Experimentos pendentes

- repetir âncoras principais com múltiplas seeds;
- resolver a discrepância de reprodutibilidade entre runs nominalmente idênticos;
- comparar de forma controlada `multiclass_all` e `binary_all`;
- implementar e avaliar a etapa de identidade, caso a branch em duas etapas prossiga;
- medir utilidade em função de distância temporal;
- comparar mecanismo temporal explícito com baseline frame-based no mesmo orçamento;
- reprocessar X1B sob a convenção atual de ASSD/HD95, caso ele permaneça relevante;
- consolidar análise por classe, paciente e vídeo;
- continuar a revisão dirigida de SSL temporal para vídeo esparsamente rotulado.

## Próximo passo

O próximo passo operacional já sustentado pelos documentos é estabilizar a evidência experimental antes de escolher a contribuição: repetir configurações âncora com múltiplas seeds e esclarecer a fonte das diferenças entre reexecuções nominalmente idênticas. Em paralelo, a revisão focada de SSL temporal deve continuar para selecionar um mecanismo temporal explícito que responda à hipótese atual, em vez de assumir que propagação rígida será a direção final.

## O que NÃO está decidido

- não está decidido que temporalidade explícita melhora o desempenho;
- não está decidido que propagação rígida funciona;
- não está decidido que o pipeline binário + identidade é superior;
- não está decidido que diffusion models terão papel metodológico central;
- não está decidido que pré-treinamento será incorporado;
- não está decidido que C2/C4 serão as únicas classes finais;
- não existe contribuição científica final fechada.
