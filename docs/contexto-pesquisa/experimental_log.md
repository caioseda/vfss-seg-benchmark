# experimental_log.md

## Convenções

Este log usa como fonte principal a auditoria técnica do repositório `vfss-seg-benchmark` em 24/09/2026.

Estados usados:

- **execução comprovada**: há CSV, `metrics.csv`, checkpoint, console de treino ou outro artefato equivalente;
- **configurado, execução não comprovada**: existe código/config, mas não há artefato de execução;
- **planejado**: aparece em documentação ou conversa, sem código/config e sem artefato.

Para os experimentos VFSS atuais:

- seed global: 42;
- split seed: 42;
- label seed: 42;
- pool seed: 42;
- não existem múltiplas seeds por célula;
- imagens: 256 x 256;
- métricas: Dice, IoU, ASSD e HD95;
- ASSD/HD95 são reportados em pixels;
- convenção atual para estrutura presente em apenas um dos lados: penalidade 362,0387 px;
- teste atual: 172 frames de 20 pacientes e 42 vídeos.

Resultados factuais e interpretação são separados em cada seção.

---

## E0. Benchmark de arquiteturas supervisionadas

**Status:** execução comprovada.

### Pergunta

Comparar arquiteturas supervisionadas para segmentação VFSS antes da fase de SSL.

### Protocolo

Benchmarks históricos em INCA e UNM. A configuração integral da era das arquiteturas não pode ser reconstruída com segurança porque os `hparams.yaml` não registram dataset, split, `target_variant` e seed.

### Configuração

- treinamento supervisionado;
- perda CE na implementação histórica;
- modelos identificados: U-Net, UNet++, TransUNet e SwinUNet;
- `n_classes=2` nos `hparams.yaml` examinados;
- seed não registrada nos artefatos; 42 é apenas inferível pelo código padrão.

### Classes

Binário, mas a variante efetiva de alvo de cada run histórico não está registrada de forma suficiente.

### Regime de labels

100% supervisionado.

### Métodos

U-Net, UNet++, TransUNet B16 / R50-B16, SwinUNet tiny / small / base.

### Seeds

Não registrada. Provável 42 por padrão de código, mas isso é inferência.

### Métricas

`test/dice_score`, `test/mean_iou`, `test/loss`.

### Resultados factuais

Em benchmark sem `fast_dev_run` de 09/05/2026:

| Modelo | Dice | mIoU |
|---|---:|---:|
| U-Net INCA | 0,8793 | 0,7468 |
| TransUNet R50-B16 INCA | 0,8645 | 0,7216 |
| SwinUNet tiny INCA | 0,7799 | 0,6003 |
| SwinUNet small INCA | 0,7302 | 0,5451 |

Há outro benchmark sem smoke test em 13/08/2026 com U-Net 0,8772 e TransUNet R50-B16 0,8676.

### Observações

Vários diretórios históricos são smoke tests e não devem ser usados como resultados científicos.

### Limitações

A configuração efetiva completa não é recuperável.

### Interpretação

Serve como contexto histórico. Não deve ser comparado diretamente aos experimentos X1 atuais.

---

## E1. ACDC, teste de aceitação do port SSL

**Status:** execução comprovada.

### Pergunta

Verificar se os wrappers de SSL foram portados de forma plausível antes de executar a grade principal no VFSS.

### Protocolo

ACDC com 140 volumes de treino, 20 de validação e 40 de teste. Orçamentos de 7 pacientes rotulados, 5%, e 14, 10%.

### Configuração

- 4 classes: fundo, RV, Myo, LV;
- backbones `repo` e `ssl4mis`;
- supervised, FixMatch, Mean Teacher e DiffRect;
- SGD lr 0,01, momentum 0,9, wd 1e-4;
- `PolynomialLR`, 30.000 passos;
- batch 6, 3 rotulados;
- política `global`.

### Classes

RV, Myo e LV reportadas.

### Regime de labels

5% e 10% de pacientes rotulados.

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

1337, uma seed por célula.

### Métricas

Dice, Jaccard, HD95 e ASD volumétricos.

### Resultados factuais

O notebook registrou, para DiffRect com backbone `ssl4mis`, diferença de -0,64 ponto de Dice em 10% e -0,23 em 5% em relação aos números de referência digitados no notebook. O critério interno era |delta Dice| <= 2 pontos e o port foi marcado como aceito.

### Observações

HD95 e ASD do código estão em voxels, enquanto a referência publicada anotada no repositório usa mm.

### Limitações

A comparação não valida equivalência completa com o artigo de origem. A presença exata da augmentation do dataset na data dos runs não pode ser confirmada.

### Interpretação

Validação de engenharia do port, não resultado científico do VFSS.

---

## E2. X1, orçamento por fração de pacientes, `multiclass_c2_c4`

**Status:** execução comprovada.

### Pergunta

SSL supera treinamento supervisionado quando diminui a fração de pacientes com rótulo?

### Protocolo

Split por paciente com seed 42. Treino com 690 frames anotados potenciais, 69 pacientes e 165 vídeos. Pool nunca anotado de 8.000 frames, stride 3, política `same_video`.

### Configuração

- backbone U-Net do repositório;
- 8.000 passos;
- AdamW, lr 1e-3, wd 1e-4;
- batch 8: 4 rotulados + 4 não rotulados;
- checkpoint por `val/dice_score`;
- target `multiclass_c2_c4`;
- `n_classes=4`, com índice 2 ausente nessa variante;
- C2 e C4 reportadas.

### Classes

C2 e C4.

### Regime de labels

| Fração | Frames revelados | Pacientes | Vídeos com rótulo |
|---|---:|---:|---:|
| 25% | 176 | 10 | 39 |
| 50% | 360 | 25 | 85 |
| 75% | 519 | 43 | 124 |
| 100% | 690 | 69 | 165 |

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

42. Uma seed por célula.

### Métricas

Dice, IoU, ASSD e HD95 por frame e classe.

### Resultados factuais

Dice por classe:

| Método | Regime | C2 | C4 | Macro |
|---|---:|---:|---:|---:|
| Supervised | 25% | 0,7452 | 0,4987 | 0,6220 |
| FixMatch | 25% | 0,7502 | 0,5616 | 0,6559 |
| Mean Teacher | 25% | 0,7551 | 0,5674 | 0,6613 |
| DiffRect | 25% | 0,8257 | 0,7390 | 0,7824 |
| Supervised | 50% | 0,8183 | 0,7059 | 0,7621 |
| FixMatch | 50% | 0,7996 | 0,6795 | 0,7396 |
| Mean Teacher | 50% | 0,8556 | 0,7930 | 0,8243 |
| DiffRect | 50% | 0,8694 | 0,7986 | 0,8340 |
| Supervised | 75% | 0,7777 | 0,6895 | 0,7336 |
| FixMatch | 75% | 0,7629 | 0,7474 | 0,7552 |
| Mean Teacher | 75% | 0,7872 | 0,7676 | 0,7774 |
| DiffRect | 75% | 0,8456 | 0,7779 | 0,8118 |
| Supervised | 100% | 0,8476 | 0,7827 | 0,8152 |
| FixMatch | 100% | 0,8731 | 0,7825 | 0,8278 |
| Mean Teacher | 100% | 0,8542 | 0,8466 | 0,8504 |
| DiffRect | 100% | 0,8847 | 0,7775 | 0,8311 |

Resultados completos de ASSD/HD95 estão na CSV pós-correção. Há casos penalizados por 362,0387 px em vários runs.

### Observações

Nos 12 runs SSL do X1 não aparecem métricas `train/pseudo_*` ou `train/teacher_*` relativas a rótulos ocultos. Pelo desenho do regime por paciente e sampling `same_video`, o fluxo não rotulado usado na prática é o pool de frames nunca anotados.

### Limitações

- confunde quantidade de rótulos, pacientes, vídeos e diversidade;
- uma seed;
- reexecuções nominalmente idênticas produziram resultados diferentes;
- versão exata do código dos runs não está registrada.

### Interpretação

SSL supera supervised em várias células, mas o experimento não isola a causa. Não demonstra benefício temporal.

---

## E3. X1 binário C2+C4 por fração de pacientes

**Status:** configurado, execução não comprovada.

### Pergunta

Repetir X1 fundindo C2 e C4 em uma única classe de primeiro plano.

### Protocolo e configuração

Mesmos regimes e métodos do X1, com `target_variant=binary_c2_c4` e `n_classes=2`.

### Classes

Fundo e FG.

### Regime de labels

25%, 50%, 75%, 100% por paciente.

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

Config atual: 42.

### Métricas

Dice, IoU, ASSD, HD95.

### Resultados

Nenhum artefato de execução foi encontrado. O notebook correspondente não possui saída.

### Limitações

Não existe resultado factual.

### Interpretação

Nenhuma.

---

## E4. X1B, política do fluxo não rotulado

**Status:** execução comprovada, grade parcial em relação ao config atual.

### Pergunta

A origem temporal dos frames não rotulados altera o desempenho do SSL?

### Protocolo

Mesmo split do X1. Todos os 690 frames anotados de treino são revelados. Pool denso de 31.953 frames nunca anotados. Políticas executadas: `global`, `same_video`, W=5, W=10.

### Configuração

- target `multiclass_c2_c4`;
- U-Net;
- 8.000 passos;
- batch 4+4;
- 100% dos labels revelados.

### Classes

C2 e C4.

### Regime de labels

100%.

### Métodos

Supervised, FixMatch e Mean Teacher. DiffRect não foi executado.

### Seeds

42. Uma seed por célula.

### Métricas

Dice, IoU, ASSD, HD95, além de métricas de atendimento da política.

### Resultados factuais

Dice macro C2/C4 aproximado:

| Política | Supervised | FixMatch | Mean Teacher |
|---|---:|---:|---:|
| global | 0,7851 | 0,7712 | 0,7320 |
| same_video | 0,8577 | 0,8308 | 0,8596 |
| W=5 | 0,8507 | 0,8169 | 0,8237 |
| W=10 | 0,7959 | 0,8218 | 0,8222 |

`policy_served=1.0` e `policy_fallback_global=0.0` em todos os 12 runs.

### Observações

O config atual declara também W=15 e DiffRect, mas esses não foram executados. Existe apenas uma tentativa W=15 supervisionada interrompida.

### Limitações

- baseline supervisionado varia entre políticas;
- uma seed;
- ASSD/HD95 usam convenção antiga com NaNs descartados;
- a grade executada não é igual à grade atual dos configs.

### Interpretação

Não há base suficiente para atribuir diferenças de desempenho à proximidade temporal. O experimento está contaminado por variabilidade entre runs e por convenção de métrica diferente.

---

## E5. X1 por frames por vídeo, `multiclass_c2_c4`

**Status:** execução comprovada.

### Pergunta

Qual é o efeito de aumentar a densidade de frames rotulados dentro dos mesmos vídeos, e SSL aproveita o pool não rotulado melhor que o baseline?

### Protocolo

69 pacientes e 165 vídeos fixos no treino. Pool de 8.000 frames nunca anotados, stride 3, `same_video`.

### Configuração

Mesma U-Net, perdas e otimização do X1.

### Classes

C2 e C4.

### Regime de labels

1, 2, 3 e 5 frames revelados por vídeo, correspondendo a 165, 330, 494 e 690 frames.

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

42.

### Métricas

Dice, IoU, ASSD, HD95.

### Resultados factuais

Dice por classe e macro:

| Método | fpv | C2 | C4 | Macro |
|---|---:|---:|---:|---:|
| Supervised | 1 | 0,7616 | 0,7116 | 0,7366 |
| FixMatch | 1 | 0,7630 | 0,7368 | 0,7499 |
| Mean Teacher | 1 | 0,7214 | 0,7203 | 0,7209 |
| DiffRect | 1 | 0,8333 | 0,6732 | 0,7533 |
| Supervised | 2 | 0,8307 | 0,7713 | 0,8010 |
| FixMatch | 2 | 0,8452 | 0,8001 | 0,8227 |
| Mean Teacher | 2 | 0,8279 | 0,7601 | 0,7940 |
| DiffRect | 2 | 0,8773 | 0,8065 | 0,8419 |
| Supervised | 3 | 0,8106 | 0,7645 | 0,7876 |
| FixMatch | 3 | 0,8505 | 0,7743 | 0,8124 |
| Mean Teacher | 3 | 0,8569 | 0,7796 | 0,8183 |
| DiffRect | 3 | 0,8651 | 0,7573 | 0,8112 |
| Supervised | 5 | 0,8433 | 0,7715 | 0,8074 |
| FixMatch | 5 | 0,8303 | 0,7644 | 0,7974 |
| Mean Teacher | 5 | 0,8181 | 0,8527 | 0,8354 |
| DiffRect | 5 | 0,8322 | 0,6739 | 0,7531 |

### Observações

Com 5 fpv, o conjunto rotulado é exatamente o mesmo do X1 100%. Mesmo assim, os resultados diferem entre as duas famílias para todas as abordagens.

### Limitações

- uma seed;
- resultados não monotônicos;
- X1 100% e fpv5 são nominalmente equivalentes quanto ao conjunto revelado, mas diferem;
- versão exata do código não registrada.

### Interpretação

Frames adicionais do mesmo vídeo trazem informação para o supervisionado, mas SSL frame-based não apresenta vantagem estável.

---

## E6. X1 por frames por vídeo, `multiclass_all`

**Status:** execução comprovada.

### Pergunta

Como o problema se comporta quando C2 a C8 são identificadas como classes distintas?

### Protocolo

Mesmo split e regimes de 1, 2, 3 e 5 fpv.

### Configuração

- `target_variant=multiclass_all`;
- `n_classes=8`;
- U-Net;
- demais parâmetros como E5.

### Classes

C2, C3, C4, C5, C6, C7 e C8.

### Regime de labels

1, 2, 3, 5 fpv.

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

42.

### Métricas

Dice, IoU, ASSD, HD95 por classe.

### Resultados factuais

Dice:

| Método | fpv | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Supervised | 1 | 0,6843 | 0,6503 | 0,6308 | 0,5388 | 0,4045 | 0,0000 | NA |
| FixMatch | 1 | 0,7673 | 0,7142 | 0,6517 | 0,5310 | 0,3024 | 0,0496 | NA |
| Mean Teacher | 1 | 0,7904 | 0,8002 | 0,7331 | 0,6366 | 0,4555 | 0,0454 | NA |
| DiffRect | 1 | 0,7566 | 0,7431 | 0,7241 | 0,6864 | 0,4272 | 0,1796 | NA |
| Supervised | 2 | 0,8235 | 0,7591 | 0,7218 | 0,7044 | 0,5414 | 0,0597 | NA |
| FixMatch | 2 | 0,7626 | 0,7442 | 0,7082 | 0,6066 | 0,4671 | 0,0000 | NA |
| Mean Teacher | 2 | 0,7595 | 0,7965 | 0,7742 | 0,7078 | 0,5709 | 0,0603 | NA |
| DiffRect | 2 | 0,8019 | 0,7572 | 0,6990 | 0,6357 | 0,4522 | 0,2055 | NA |
| Supervised | 3 | 0,8612 | 0,8718 | 0,7755 | 0,6833 | 0,5125 | 0,1447 | NA |
| FixMatch | 3 | 0,8074 | 0,7731 | 0,7492 | 0,6836 | 0,5406 | 0,0120 | NA |
| Mean Teacher | 3 | 0,7638 | 0,8035 | 0,7742 | 0,6999 | 0,5875 | 0,3312 | NA |
| DiffRect | 3 | 0,8112 | 0,7236 | 0,6562 | 0,5601 | 0,4320 | 0,2546 | NA |
| Supervised | 5 | 0,8335 | 0,7893 | 0,7566 | 0,6351 | 0,4857 | 0,0699 | NA |
| FixMatch | 5 | 0,7691 | 0,7581 | 0,7438 | 0,6774 | 0,5067 | 0,1520 | NA |
| Mean Teacher | 5 | 0,7739 | 0,7727 | 0,7411 | 0,6435 | 0,5251 | 0,1183 | NA |
| DiffRect | 5 | 0,7852 | 0,6957 | 0,6480 | 0,5832 | 0,4117 | 0,2219 | NA |

C8 é NA porque não aparece em nenhum frame de validação ou teste.

### Observações

Frequência por teste: C2 172, C3 172, C4 172, C5 148, C6 97, C7 39, C8 0 frames com GT presente.

### Limitações

Desbalanceamento severo e ausência total de C8 no teste impedem tratar todas as classes como igualmente avaliadas.

### Interpretação

A piora das vértebras inferiores é factual. A causa não está identificada. Identidade, frequência, visibilidade e anotação são hipóteses concorrentes.

---

## E7. X1 por frames por vídeo, `binary_all`

**Status:** execução comprovada.

### Pergunta

A dificuldade principal diminui quando todas as vértebras são tratadas como uma única classe de primeiro plano?

### Protocolo

Mesmo split e regimes de 1, 2, 3 e 5 fpv.

### Configuração

- `target_variant=binary_all`;
- `n_classes=2`;
- U-Net;
- 16 runs;
- todos com `run.json`, `config.yaml`, checkpoint e status final.

### Classes

Fundo e FG.

### Regime de labels

1, 2, 3 e 5 fpv.

### Métodos

Supervised, FixMatch, Mean Teacher, DiffRect.

### Seeds

42.

### Métricas

Dice, IoU, ASSD, HD95 para FG.

### Resultados factuais

| Método | fpv | Dice | IoU | ASSD | HD95 |
|---|---:|---:|---:|---:|---:|
| Supervised | 1 | 0,8816 | 0,7957 | 1,3147 | 5,4378 |
| FixMatch | 1 | 0,8683 | 0,7751 | 1,7936 | 10,7325 |
| Mean Teacher | 1 | 0,8714 | 0,7797 | 1,2887 | 6,0656 |
| DiffRect | 1 | 0,8308 | 0,7203 | 1,5988 | 6,1687 |
| Supervised | 2 | 0,8852 | 0,8033 | 1,1994 | 5,2471 |
| FixMatch | 2 | 0,8864 | 0,8039 | 1,2644 | 5,7804 |
| Mean Teacher | 2 | 0,8956 | 0,8172 | 1,0515 | 4,7009 |
| DiffRect | 2 | 0,8282 | 0,7137 | 1,6501 | 6,3923 |
| Supervised | 3 | 0,8890 | 0,8076 | 1,2565 | 6,2852 |
| FixMatch | 3 | 0,8857 | 0,8021 | 1,2088 | 5,5919 |
| Mean Teacher | 3 | 0,8845 | 0,8007 | 1,3347 | 6,3456 |
| DiffRect | 3 | 0,8478 | 0,7431 | 1,4378 | 5,5723 |
| Supervised | 5 | 0,8867 | 0,8075 | 1,2146 | 5,3684 |
| FixMatch | 5 | 0,8839 | 0,7985 | 1,1915 | 5,3884 |
| Mean Teacher | 5 | 0,8887 | 0,8091 | 1,2038 | 5,2421 |
| DiffRect | 5 | 0,8546 | 0,7516 | 1,4229 | 5,4556 |

A máscara `binary_all` foi verificada como exatamente equivalente a `(multiclass_all > 0)` em todos os arquivos auditados.

### Observações

O run DiffRect fpv2 foi retomado de checkpoint no passo 2000 após uma tentativa interrompida. O `run.json` final está como `done`, mas preserva o campo de erro da primeira tentativa.

### Limitações

Uma seed. A tarefa não preserva identidade de vértebra.

### Interpretação

A segmentação binária de tecido vertebral é muito mais fácil e quase satura com 1 fpv. Isso sugere que a identificação multiclasse e o desbalanceamento explicam parte importante da dificuldade, mas não prova superioridade de um pipeline em duas etapas.

---

## E8. Variantes `binario` e `binario-c2-c6` por frames por vídeo

**Status:** execução não comprovada.

### Pergunta

Explorar outras variantes binárias intermediárias.

### Protocolo

Dry-runs listam 16 células por família.

### Configuração

Os configs citados nos logs não existem no estado atual do repositório. A variante de máscara C2-C6 documentada por script também não existe em disco.

### Classes, regime, métodos, seeds e métricas

Não confirmáveis além dos nomes e da grade de dry-run.

### Resultados

Nenhum.

### Interpretação

Nenhuma.

---

## E9. SDSeg + TCM, fase anterior

**Status:** execução relatada em contexto; artefatos completos não são reconstruíveis pela auditoria atual.

### Pergunta

Adicionar contexto temporal ao SDSeg melhora segmentação VFSS?

### Protocolo

SDSeg com TCM e janelas temporais, incluindo 3, 5 e 7 em registros anteriores.

### Configuração

A síntese de conversa registra que a janela é codificada pelo SDSeg e os latentes são concatenados antes do TCM. A versão exata do código e dos dados usados precisa de verificação manual.

### Classes

CONFLITO A VERIFICAR: os registros antigos não permitem fixar com segurança se cada resultado corresponde à formulação binária original ou ao recorte C2/C4.

### Regime de labels

Supervisionado.

### Métodos

SDSeg, SDSeg+TCM.

### Seeds

Não há múltiplas seeds registradas.

### Métricas e resultados

Um resultado recorrente nas sínteses é Dice aproximadamente 0,866 para SDSeg e 0,872 para TCM-3, com mIoU 0,770 para 0,777. Outro registro posterior para um recorte C2/C4 cita SDSeg 0,889 e TCM-3 0,885.

### Observações

Esses números pertencem a protocolos históricos diferentes e não devem ser fundidos.

### Limitações

Configuração incompleta, protocolo histórico e ausência de seeds repetidas.

### Interpretação

Não há evidência robusta de ganho temporal nessa linha.

---

## E10. X2 a X13 e demais mecanismos temporais propostos

**Status:** planejados, sem execução comprovada no repositório auditado.

### Pergunta

As propostas cobrem, em diferentes versões da documentação: propagação de máscara, rigidez local, estimabilidade do movimento, utilidade marginal do transporte, consistência temporal, proveniência de frames não rotulados e estabilidade temporal.

### Protocolo

Não existe um único protocolo executado. As numerações X2 a X13 aparecem em documentação de hipóteses e mudaram ao longo do projeto.

### Configuração, classes, regime de labels, métodos, seeds e métricas

Ainda não canônicos.

### Resultados

Nenhum artefato de execução com esses identificadores foi encontrado.

### Observações

Também foram discutidos copy de máscara, Procrustes, fluxo óptico/RAFT, correspondência auto-supervisionada, SAM2, memória temporal, PCA de forma e mecanismos derivados de difusão.

### Limitações

São ideias ou planos, não resultados.

### Interpretação

Não devem ser citados como evidência experimental.

---

## Inconsistências técnicas transversais

1. Nenhuma família VFSS possui múltiplas seeds.
2. X1 teve reexecuções com `hparams.yaml` idêntico e resultados diferentes.
3. X1 100% e X1-fpv 5 revelam exatamente os mesmos 690 frames, mas produzem resultados diferentes.
4. A versão exata do código não foi registrada para 76 dos 92 runs referenciados pelas CSVs auditadas.
5. X1B usa convenção antiga de ASSD/HD95.
6. A árvore de trabalho auditada estava suja, com arquivos relevantes modificados e não rastreados.
7. C8 aparece em 18 frames, todos no treino, e não pode ser avaliada no split atual.
8. A augmentation supervisionada não é idêntica entre todos os métodos.
9. O nome de alguns runs não corresponde ao prefixo atual dos configs.
10. O protocolo de anotação e o critério de `selected_labeler` não podem ser determinados pelo código.
