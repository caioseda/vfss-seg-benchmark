# dissertation.md

## 1. Escopo e problema de pesquisa

Esta pesquisa de mestrado investiga segmentação de vértebras cervicais em vídeos de videofluoroscopia da deglutição, VFSS, obtidos em colaboração com o INCA, em um cenário de anotação esparsa. O foco científico atual é a segmentação semissupervisionada quando apenas poucos frames de cada vídeo possuem máscaras manuais e existe um conjunto muito maior de frames não rotulados provenientes dos mesmos vídeos.

A motivação clínica original está ligada à automatização de medidas anatômicas usadas na análise de VFSS, em especial medidas envolvendo C2 e C4. Ao longo da pesquisa, entretanto, o problema científico foi recentrado. O objetivo atual não é apenas obter a maior métrica de segmentação possível, mas entender quais fontes de informação são úteis sob orçamento limitado de rótulos: diversidade de pacientes, número de vídeos, densidade de frames anotados por vídeo, frames nunca anotados, correlação temporal e representação das vértebras.

A hipótese ampla que motiva a etapa atual é que vídeos contêm estrutura temporal que métodos de SSL aplicados frame a frame podem não explorar adequadamente. Essa hipótese ainda não foi demonstrada. Os experimentos disponíveis caracterizam o problema e restringem interpretações, mas ainda não estabelecem uma contribuição metodológica final.

## 2. Dataset e protocolo experimental atual

A auditoria técnica de 24/09/2026 é a fonte de maior confiabilidade para as contagens e configurações efetivamente usadas nos experimentos recentes.

O dataset VFSS em disco contém:

- 961 frames anotados, todos com identificadores únicos;
- 232 vídeos representados na tabela de frames anotados;
- 114 pacientes;
- 46.137 PNGs no diretório de frames, distribuídos por 234 diretórios de vídeo;
- cinco identificadores de rotulador presentes na base: AM, BB, CS, VC e VR;
- variantes de máscara `raw`, `binary_c2_c4`, `binary_all`, `multiclass_c2_c4` e `multiclass_all`.

Na variante `multiclass_all`, os índices de classe são documentados nos arquivos de configuração como fundo, C2, C3, C4, C5, C6, C7 e C8. A presença das classes é fortemente desbalanceada. Nos 961 frames anotados, C2, C3 e C4 aparecem em todos os frames, C5 em 894, C6 em 528, C7 em 199 e C8 em apenas 18. C8 aparece somente no treino no split usado pelos experimentos atuais e não está presente em validação ou teste.

Os experimentos X1 usam `split_mode: seed` com seed 42 e divisão por paciente. O split efetivo verificado é:

| Split | Frames anotados | Pacientes | Vídeos |
|---|---:|---:|---:|
| treino | 690 | 69 | 165 |
| validação | 99 | 25 | 25 |
| teste | 172 | 20 | 42 |

Não há paciente ou vídeo compartilhado entre os splits nesse protocolo.

Os frames nunca anotados são construídos somente a partir dos vídeos de treino. Existem 31.953 frames elegíveis fora dos frames anotados. Nos principais experimentos X1 e X1 por frames por vídeo, esse pool é subamostrado com `frame_stride=3` e limitado a 8.000 frames. No X1B, foi usado o pool denso de 31.953 frames.

O pré-processamento atual normaliza cada imagem em escala de cinza, reescala para [-1, 1], redimensiona para 256 x 256 e repete o canal para três canais nos experimentos VFSS. As máscaras são redimensionadas por nearest neighbor. O dataset VFSS não aplica augmentation diretamente; as transformações adicionais dependem do wrapper de cada método.

### 2.1 Regimes de rótulo

Dois regimes principais foram efetivamente usados.

No X1 por fração de pacientes, pacientes inteiros do treino têm seus rótulos revelados ou ocultados:

| Fração | Frames revelados | Pacientes revelados | Vídeos com rótulo |
|---|---:|---:|---:|
| 25% | 176 | 10 | 39 |
| 50% | 360 | 25 | 85 |
| 75% | 519 | 43 | 124 |
| 100% | 690 | 69 | 165 |

No X1 por frames por vídeo, todos os 165 vídeos de treino permanecem representados, mas varia o número máximo de frames revelados por vídeo:

| Frames por vídeo | Frames revelados | Vídeos com rótulo |
|---|---:|---:|
| 1 | 165 | 165 |
| 2 | 330 | 165 |
| 3 | 494 | 165 |
| 5 | 690 | 165 |

O regime de 5 frames por vídeo revela todos os 690 frames anotados de treino e, portanto, usa o mesmo conjunto rotulado que o regime X1 com 100% dos pacientes.

## 3. Trajetória científica

A pesquisa começou com uma formulação mais centrada na aplicação clínica e na escolha de uma arquitetura capaz de segmentar as vértebras. Foram exploradas arquiteturas discriminativas e modelos de difusão para segmentação médica, incluindo MedSegDiff e SDSeg. MedSegDiff apresentou dificuldades práticas de reprodução ou resultados insuficientes para sustentar a linha. SDSeg foi treinado com sucesso e se tornou a principal base de difusão da fase inicial.

Também foi incorporado contexto temporal ao SDSeg por meio de um TCM e janelas temporais. Os ganhos registrados foram pequenos e inconsistentes. Em uma comparação anteriormente registrada, o TCM com janela 3 produziu alteração aproximada de Dice de 0,866 para 0,872 e mIoU de 0,770 para 0,777. Essa magnitude não foi considerada evidência suficiente, especialmente porque não havia múltiplas seeds.

A linha de difusão foi então reposicionada. Ela permanece relevante como parte da trajetória e como possível ferramenta secundária, mas deixou de ser o centro da tese. O foco migrou para eficiência de rótulos e SSL porque o dataset contém poucos frames anotados por vídeo e muitos frames nunca anotados.

O primeiro desenho de SSL, denominado X1, variou a fração de pacientes rotulados. A análise posterior identificou que esse desenho alterava simultaneamente o número de pacientes, vídeos, frames anotados e a diversidade anatômica do conjunto rotulado. Isso levou ao X1 por frames por vídeo, que mantém pacientes e vídeos do treino constantes e varia apenas a densidade de anotações por vídeo de forma mais controlada.

Experimentos posteriores ampliaram a análise para todas as vértebras em formulação multiclasse e, depois, para uma formulação binária em que todas as vértebras são fundidas em uma única classe de primeiro plano.

## 4. Fundamentação e trabalhos relacionados conhecidos

A fundamentação atualmente relevante está organizada em quatro eixos.

### 4.1 Segmentação semissupervisionada

Os baselines recentes incluem treinamento supervisionado, Mean Teacher, FixMatch e DiffRect. Mean Teacher representa o paradigma teacher-student com professor atualizado por média móvel exponencial. FixMatch usa pseudo-rótulos produzidos em uma visão fracamente aumentada e supervisiona uma visão fortemente aumentada quando a confiança ultrapassa um limiar. DiffRect adiciona um mecanismo de retificação baseado em difusão latente e foi adotado como um baseline SSL mais específico para segmentação médica.

Esses métodos são usados principalmente como referências frame-based. No protocolo atual, eles não modelam explicitamente correspondência temporal, movimento ou memória entre frames sucessivos.

### 4.2 SSL temporal e correspondência

A pesquisa passou a considerar a literatura de SSL para vídeo esparsamente rotulado, consistência temporal, propagação de rótulos, fluxo óptico, correspondência auto-supervisionada e memória temporal. Entre os trabalhos e famílias já levantados nas conversas estão SegProp, SMART, trabalhos de correspondência auto-supervisionada associados a Jabri, Owens e Efros, além de métodos de propagação e rastreamento. Esses trabalhos ainda não foram organizados em uma revisão canônica completa, portanto seu papel exato na fundamentação final permanece em consolidação.

### 4.3 Movimento e rigidez

Foi levantada a hipótese de que vértebras, por serem estruturas rígidas, possam admitir aproximações úteis por transformação rígida entre frames próximos. Também foram identificados trabalhos de rastreamento de vértebras em videofluoroscopia associados a Cerciello, Plocharski e Bifulco. O contexto disponível sugere que a literatura já contém evidência de rastreamento rígido neste domínio, o que reduz a novidade de apenas demonstrar que movimento rígido pode ser estimado. O ponto ainda aberto seria se esse mecanismo fornece supervisão adicional útil para o problema atual.

### 4.4 Difusão

MedSegDiff, SDSeg e DiffRect constituem os principais pontos de contato com diffusion models. A experiência até aqui não estabeleceu vantagem clara de usar difusão como eixo central da pesquisa. Possibilidades como incerteza, múltiplas máscaras, refinamento e pseudo-rótulos foram discutidas, mas não constituem hoje uma contribuição validada.

## 5. Metodologia atual

### 5.1 Backbone e treino

Todos os runs VFSS das famílias X1 auditadas usam uma U-Net do próprio repositório como backbone, com aproximadamente 17,3 milhões de parâmetros segundo os logs do Lightning. O treino usa AdamW, learning rate 1e-3, weight decay 1e-4, scheduler `CosineAnnealingLR`, 8.000 passos e seleção de checkpoint por Dice de validação.

O batch dos experimentos SSL contém 8 amostras, com 4 rotuladas e 4 não rotuladas. O `TwoStreamBatchSampler` suporta políticas `global`, `same_video` e `temporal_window`.

O baseline supervisionado usa perda `0.5 * (CE + Dice)` apenas nas linhas rotuladas. Mean Teacher acrescenta consistência MSE entre aluno e professor. FixMatch usa pseudo-rótulo com limiar 0,95 e augmentations fraca e forte. DiffRect usa o modo `reference` em todos os runs encontrados e combina perdas de pseudo-supervisão, refinamento e retificação.

Existe uma assimetria de augmentation entre os métodos: no VFSS, supervised e Mean Teacher supervisionam imagens rotuladas sem augmentation, enquanto DiffRect aplica a visão fracamente aumentada também na perda supervisionada. Isso deve ser tratado como parte do protocolo experimental, não como se todos os métodos recebessem transformações idênticas.

### 5.2 Métricas

As métricas atuais por frame e classe são Dice, IoU, ASSD e HD95. ASSD e HD95 estão em pixels sobre imagens 256 x 256.

Após uma correção introduzida durante a sequência experimental, casos em que a estrutura aparece apenas na predição ou apenas no ground truth recebem penalidade igual à diagonal da imagem, aproximadamente 362,0387 pixels. O X1 foi reprocessado segundo essa convenção. O X1B, porém, permanece com a convenção antiga, na qual esses casos eram NaN e removidos por `nanmean`. Portanto, métricas de distância do X1B não são diretamente comparáveis às demais famílias sem reprocessamento.

## 6. Experimentos e resultados

Os valores completos pertencem ao `experimental_log.md`. Nesta narrativa são mantidos apenas os resultados necessários para sustentar a interpretação científica.

### 6.1 X1 por fração de pacientes

O X1 compara supervisionado, Mean Teacher, FixMatch e DiffRect em 25%, 50%, 75% e 100% dos pacientes de treino rotulados, sempre com seed 42.

Em Dice médio de C2 e C4, os valores observados foram aproximadamente:

| Método | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| supervisionado | 0,622 | 0,762 | 0,734 | 0,815 |
| Mean Teacher | 0,661 | 0,824 | 0,777 | 0,850 |
| FixMatch | 0,656 | 0,740 | 0,755 | 0,828 |
| DiffRect | 0,782 | 0,834 | 0,812 | 0,831 |

Há ganhos SSL em várias células, mas o comportamento não é monotônico com o orçamento de rótulos. Mais importante, o experimento confunde quantidade de rótulos com diversidade de pacientes e vídeos. Ele não permite atribuir os ganhos ao uso dos frames nunca anotados, à estrutura temporal ou à eficiência do método sob densidade esparsa dentro de um vídeo.

Também foram encontradas reexecuções nominalmente idênticas, com a mesma seed e `hparams.yaml`, que produziram resultados diferentes. Isso reduz a força inferencial de diferenças observadas em uma única execução.

### 6.2 X1 por frames rotulados por vídeo, C2/C4

Esse experimento mantém os mesmos 69 pacientes e 165 vídeos de treino e varia 1, 2, 3 ou 5 frames anotados por vídeo.

O baseline supervisionado passa de Dice médio C2/C4 de aproximadamente 0,7366 com 1 frame por vídeo para 0,8074 com 5 frames por vídeo. Logo, frames anotados adicionais dos mesmos vídeos ainda contêm informação útil.

Entretanto, os métodos SSL não exibem uma vantagem consistente em todos os regimes. Mean Teacher atinge aproximadamente 0,8354 no regime de 5 frames, enquanto supervisionado atinge 0,8074, mas em outros regimes a ordem entre métodos muda. DiffRect, por exemplo, é forte em 2 frames por vídeo e cai no regime de 5 frames.

Esse experimento não testa temporalidade explícita. Ele testa métodos SSL frame-based sob diferentes densidades de anotação.

### 6.3 X1B: política de amostragem temporal

O X1B compara pool global, mesmo vídeo, janela temporal de 5 frames e janela temporal de 10 frames, com todos os 690 frames anotados do treino revelados e pool denso de 31.953 frames. Foram executados supervised, FixMatch e Mean Teacher, todos com seed 42. DiffRect não foi executado nessa grade.

Os resultados não sustentam uma conclusão limpa sobre política temporal. O próprio baseline supervisionado varia fortemente entre políticas, embora a política do fluxo não rotulado não devesse representar uma diferença científica do objetivo supervisionado. Isso, somado à ausência de múltiplas seeds e às divergências de reprodutibilidade observadas em outras reexecuções, impede atribuir as diferenças entre `global`, `same_video` e janelas temporais à política de amostragem.

Além disso, ASSD e HD95 do X1B usam a convenção antiga de tratamento de casos degenerados.

### 6.4 Multiclasse com C2 a C8

O X1 por frames por vídeo foi repetido com `multiclass_all`, oito saídas incluindo fundo e C2 a C8.

O desempenho cai de forma marcante nas vértebras inferiores. C2, C3 e C4 mantêm Dice moderado ou alto em diversas células, enquanto C5 e C6 são mais difíceis e C7 é muito fraca. C8 não pode ser avaliada porque não existe no conjunto de validação ou teste do split atual.

Exemplos no baseline supervisionado:

- 1 frame por vídeo: C2 0,6843; C3 0,6503; C4 0,6308; C5 0,5388; C6 0,4045; C7 0,0000;
- 3 frames por vídeo: C2 0,8612; C3 0,8718; C4 0,7755; C5 0,6833; C6 0,5125; C7 0,1447;
- 5 frames por vídeo: C2 0,8335; C3 0,7893; C4 0,7566; C5 0,6351; C6 0,4857; C7 0,0699.

A distribuição de presença das classes oferece uma explicação alternativa forte à hipótese de "falha de representação compartilhada": C7 aparece em apenas 39 frames de teste e C8 em nenhum. Assim, desbalanceamento, visibilidade e cobertura do split devem ser considerados antes de interpretar a degradação como um problema puramente arquitetural.

### 6.5 Binário com todas as vértebras fundidas

A variante `binary_all` foi executada em 23/09/2026 e possui a evidência técnica mais completa entre as famílias atuais, inclusive `run.json`, `config.yaml` e estado final para as 16 células.

A máscara `binary_all` foi verificada pixel a pixel como equivalente a `multiclass_all > 0`.

Os resultados são substancialmente mais altos e muito mais estáveis em relação ao número de frames anotados:

| Método | 1 fpv | 2 fpv | 3 fpv | 5 fpv |
|---|---:|---:|---:|---:|
| supervisionado | 0,8816 | 0,8852 | 0,8890 | 0,8867 |
| FixMatch | 0,8683 | 0,8864 | 0,8857 | 0,8839 |
| Mean Teacher | 0,8714 | 0,8956 | 0,8845 | 0,8887 |
| DiffRect | 0,8308 | 0,8282 | 0,8478 | 0,8546 |

Esse resultado estabelece que segmentar o conjunto das vértebras como uma região binária é muito mais fácil do que identificar todas as vértebras como classes distintas no mesmo protocolo. Ele também mostra uma curva quase saturada já com um frame anotado por vídeo para a tarefa binária. Isso não demonstra que um pipeline binário seguido de identificação seja superior, porque a etapa de identidade ainda não foi implementada nem avaliada. O resultado apenas torna essa decomposição uma hipótese empiricamente motivada.

### 6.6 Benchmark de arquiteturas e difusão

Há benchmarks supervisionados antigos com execução comprovada para U-Net, TransUNet e SwinUNet, além de resultados registrados para SDSeg e TCM nas sínteses de conversa. O repositório confirma, por exemplo, U-Net com Dice de aproximadamente 0,8793 em um benchmark binário e valores inferiores para as variantes SwinUNet testadas.

Entretanto, a auditoria não consegue reconstruir com segurança a configuração efetiva completa da era das arquiteturas, pois os `hparams.yaml` não registram dataset, split, variante de alvo e seed. Por isso, esses resultados devem permanecer históricos e não devem ser misturados diretamente com os experimentos X1 atuais.

### 6.7 ACDC como teste de aceitação do port SSL

O repositório também contém uma reprodução no ACDC com 16 runs, usando duas U-Nets, 5% e 10% de pacientes rotulados e os quatro wrappers. O objetivo foi validar a implementação SSL antes da grade VFSS. O notebook registrou o port como aceito segundo um critério de proximidade do DiffRect à referência publicada. Esse experimento valida parcialmente a implementação, mas não é evidência sobre a hipótese clínica ou temporal do VFSS.

## 7. Discussão

### 7.1 X1 por paciente e X1 por frames por vídeo medem fenômenos diferentes

O X1 por paciente mostra que SSL pode superar o baseline supervisionado em várias configurações. Contudo, ao reduzir pacientes rotulados, também se reduz diversidade anatômica, número de vídeos e número de frames rotulados. O experimento mede um cenário de escassez combinada.

O X1 por frames por vídeo controla melhor essa fonte de confusão. Todos os vídeos permanecem representados e apenas a densidade de anotações muda. Nesse regime, aumentar o número de frames rotulados melhora o supervisionado de 1 para 5 frames por vídeo, mas o ganho dos métodos SSL sobre o baseline não é sistemático.

A leitura atualmente sustentada é que diversidade interpaciente e densidade temporal de anotação são fontes de informação distintas. Os resultados sugerem que adicionar frames do mesmo vídeo traz ganho, mas não demonstram que SSL frame-based consiga converter de forma estável o grande pool não rotulado em desempenho adicional.

### 7.2 Ausência de ganho frame-based não invalida temporalidade

Os experimentos X1 e X1 por frames por vídeo não contêm um mecanismo temporal explícito. Portanto, ausência de ganho consistente de Mean Teacher, FixMatch ou DiffRect não demonstra que a temporalidade seja inútil. Ela demonstra apenas que esses métodos, nessa configuração, não apresentaram aproveitamento estável suficiente para estabelecer essa conclusão.

O TCM anterior também não forneceu evidência forte a favor da temporalidade, mas representa uma abordagem diferente e com protocolo experimental menos controlado.

### 7.3 A formulação multiclasse ampla mistura identidade, frequência e visibilidade

A forte queda em C5, C6 e C7 no `multiclass_all` não deve ser atribuída diretamente a uma única causa. As classes inferiores são menos frequentes e mais difíceis de visualizar; C8 sequer aparece no teste. A formulação binária, por outro lado, alcança Dice próximo de 0,88 com apenas um frame rotulado por vídeo.

Isso indica que grande parte da dificuldade do `multiclass_all` está associada à necessidade de distinguir identidades e lidar com classes raras, não apenas à capacidade de delinear tecido vertebral. A hipótese de segmentação binária seguida de identificação torna-se relevante, mas continua não testada como pipeline completo.

### 7.4 Reprodutibilidade limita a força das conclusões atuais

Nenhum experimento VFSS possui múltiplas seeds. Além disso, execuções nominalmente idênticas, com mesma seed e `hparams.yaml`, geraram resultados diferentes em datas distintas. O relatório técnico não consegue separar três explicações: mudanças de código não registradas, não determinismo de GPU e mudança no comportamento do gerador do DataLoader.

Consequentemente, diferenças de poucos pontos de Dice não podem ser tratadas como evidência estável enquanto esse problema não for controlado. Isso é particularmente importante para o X1B e para qualquer comparação entre métodos ou políticas temporais.

## 8. Hipóteses em aberto

### H1. Diversidade de pacientes pode ser mais informativa do que densidade de frames anotados dentro do mesmo vídeo

O X1 e o X1 por frames por vídeo são compatíveis com essa hipótese, mas não fornecem uma comparação causal direta. A hipótese permanece parcialmente testada.

### H2. SSL frame-based convencional pode subutilizar a estrutura temporal

Os métodos atuais não modelam correspondência temporal explicitamente e não apresentam ganho estável no eixo frames por vídeo. Isso mantém a hipótese plausível, mas não a demonstra.

### H3. Um mecanismo temporal explícito pode extrair informação adicional dos frames não rotulados

Consistência temporal, propagação, correspondência, memória e transporte de máscara continuam como branches. Ainda não existe comparação controlada que demonstre ganho sobre baselines frame-based sob o mesmo orçamento.

### H4. Frames próximos são parcialmente redundantes

A curva binária quase plana entre 1 e 5 frames por vídeo fornece evidência forte de baixa utilidade marginal adicional para a tarefa binária. Para C2/C4 multiclasse, existe ganho supervisionado entre 1 e 5 frames, portanto a redundância não é total. O efeito de distância temporal ainda não foi medido diretamente.

### H5. Transformações rígidas podem ser úteis como mecanismo local de transporte

A literatura conhecida sugere que rastreamento rígido de vértebras em videofluoroscopia não é uma ideia nova. A questão relevante passa a ser se esse transporte produz supervisão adicional útil no regime atual. Isso ainda não foi testado de forma conclusiva.

### H6. A dificuldade do multiclasse amplo está relacionada à identidade e ao desbalanceamento

O contraste entre `multiclass_all` e `binary_all` sustenta essa hipótese de forma mais forte do que antes. Ainda é necessário separar identidade, frequência, visibilidade e qualidade de anotação.

### H7. Pré-treinamento ou representação autossupervisionada pode melhorar generalização

Essa branch foi discutida em função do grande número de frames não rotulados e de sinais de overfitting. Não existe experimento concluído que a valide.

## 9. Limitações atuais

As principais limitações são:

- apenas uma seed por célula experimental;
- versão exata do código desconhecida para a maioria dos runs;
- árvore de trabalho não commitada durante parte importante da execução;
- reexecuções nominalmente idênticas com resultados diferentes;
- convenções diferentes de ASSD/HD95 entre X1B e as demais famílias;
- assimetria de augmentation entre métodos;
- classe C8 ausente em validação e teste;
- forte desbalanceamento de C5, C6 e C7;
- protocolo de anotação e critério de `selected_labeler` ainda não documentados de forma canônica;
- resultados antigos de arquitetura e difusão sem configuração integralmente recuperável;
- ausência, até o momento, de um experimento controlado que compare SSL frame-based e um mecanismo temporal explícito.

## 10. Espaço atual de contribuição

A contribuição final ainda não está definida.

O que já está sustentado é uma caracterização empírica do problema de anotação esparsa em VFSS:

1. reduzir pacientes rotulados e reduzir frames rotulados por vídeo são regimes diferentes e não devem ser tratados como equivalentes;
2. frames adicionais do mesmo vídeo trazem informação para a tarefa C2/C4, mas os métodos SSL frame-based testados não exploram essa informação de forma consistentemente superior;
3. a tarefa binária de delinear vértebras satura muito cedo e é substancialmente mais fácil do que a identificação multiclasse ampla;
4. a formulação C2 a C8 sofre com forte desbalanceamento e ausência de C8 no teste;
5. os resultados atuais ainda são limitados por reprodutibilidade e falta de múltiplas seeds.

O espaço de contribuição continua aberto entre: SSL temporal explícito, propagação ou correspondência temporal, decomposição binária seguida de identidade, análise de eficiência de rótulos e, de forma mais secundária, representação pré-treinada ou mecanismos derivados de difusão.

Nenhuma dessas branches deve ser apresentada, neste estágio, como contribuição final estabelecida.
