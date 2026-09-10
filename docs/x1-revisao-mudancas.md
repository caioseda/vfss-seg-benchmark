# Revisão das mudanças — commits `7fe17b0`, `64deff0` e árvore de trabalho atual

> Companheiro de leitura para validar o experimento X1
> (`notebooks/hyphotesis_testing/experiment_1_does_ssl_helps.ipynb`).
> Escrito em 2026-09-09 contra `main @ 64deff0` + working tree suja.

> ### ⚠️ Estado em 2026-09-09 (posterior a este documento)
>
> O desenho do X1 mudou depois desta revisão, e várias seções abaixo descrevem o estado anterior.
> O que já foi endereçado:
>
> | Item | Estado |
> |---|---|
> | **A1** (rampa de consistência presa em `//150`) | **corrigido** — o cronograma virou fração do orçamento (`consistency_rampup_fraction` / `consistency_warmup_fraction`), resolvido contra `trainer.max_steps`. Ver `src/models/ssl/base.py` e o teste `test_warmup_and_rampup_scale_with_the_budget`. |
> | Batch cheio de rotulados em 100% (fallback do sampler) | **corrigido** — com o pool não rotulado ativo o `TwoStreamBatchSampler` roda em todos os orçamentos; o fallback agora emite `RuntimeWarning`. |
> | Células degeneradas em 100% | **não existem mais** — o fluxo não rotulado independe de `label_fraction`. |
> | Orçamentos 10/20/50/100 e a ressalva dos 3 pacientes | **substituídos** por 25/50/75/100 (10/25/43/69 pacientes revelados). |
> | Falta de diagnóstico do pseudo-rótulo | **corrigido** — `expose_hidden_targets` + `train/pseudo_*` / `train/teacher_*`. |
>
> **Mudança de desenho, não só de bug:** o fluxo não supervisionado passou a vir de um pool de
> frames **nunca anotados** (~32k elegíveis no split de treino, 8000 usados), e não mais apenas dos
> frames anotados com rótulo escondido. As tabelas de composição de treino das seções 3 e 8 estão,
> portanto, defasadas.

---

## 0. Como usar este documento

Ele está organizado em três camadas:

| Seção | Para quê |
|---|---|
| **1–2** | Mapa: o que mudou, onde, em qual commit. Leia antes de abrir qualquer arquivo. |
| **3–7** | Explicação por subsistema (dados, métricas, SSL, treino, avaliação), com ponteiros `arquivo:linha`. |
| **8** | **Achados de validação** — o que eu verificaria antes de confiar nos números. Comece por aqui se tiver pressa. |
| **9** | Checklist executável (comandos) + estado atual dos runs. |

---

## 1. Escopo: os três blocos de mudança

```
7fe17b0  Refactor VFSSWindowFrameDataset          ← reescrita da camada de dados
64deff0  Métricas de distância simétrica/Hausdorff ← ASSD/HD95 + máscara multiclasse
(atual)  não commitado                             ← todo o X1: SSL, avaliação, testes, notebook
```

### 1.1 `7fe17b0` — refatoração da camada de dados

| Arquivo | O que aconteceu |
|---|---|
| `src/data/vfss_v2.py` | **removido** (401 linhas) |
| [src/data/vfss_frame_dataset.py](src/data/vfss_frame_dataset.py) | **novo** (692 linhas) — substitui o anterior |
| [src/data/samplers.py](src/data/samplers.py) | **novo** — `TwoStreamBatchSampler` |
| [src/data/unm.py](src/data/unm.py) | 185 linhas → 43: virou só configuração do dataset base |
| [src/data/base.py](src/data/base.py) | `DataModuleFromConfig` aceita `train_batch_sampler` |
| 14 YAMLs | `src.data.vfss_v2.*` → `src.data.vfss_frame_dataset.*`; UNM ganhou `dataset_path` (era `data_root`) e `window_size`/`stride`/`repeat_channels` |
| `nohup.out` | 928 linhas de log de execução foram commitadas por acidente — vale um `git rm --cached` |

⚠️ `src/data/vfss.py` (o `VFSSIncaTrain/Val/Test` **antigo**, sobre `VFSSImageDataset`) continua no
repo e expõe classes com **os mesmos nomes** das novas. Nenhum config aponta mais para lá, mas o
risco de importar o módulo errado existe. Candidato a remoção.

### 1.2 `64deff0` — ASSD/HD95 e alvo multiclasse

| Arquivo | O que aconteceu |
|---|---|
| [src/metrics.py](src/metrics.py) | **novo** (+109) — `average_symmetric_surface_distance`, `hausdorff_distance_95` |
| [src/models/base.py](src/models/base.py) | as duas métricas entram em `compute_metrics` |
| [src/data/vfss_frame_dataset.py:435](src/data/vfss_frame_dataset.py#L435) | `_preprocess_mask` passa a preservar índices de classe quando o alvo é multiclasse |
| `configs/experiment/unet/vfss-inca-unet-multiclass-c2-c4.yaml` | primeiro config multiclasse |

### 1.3 Árvore atual (não commitada) — o experimento X1

```
src/models/ssl/          base.py, supervised.py, mean_teacher.py, fixmatch.py, diffrect.py
src/third_party/ssl4mis/ losses.py (DiceLoss, MIT, com cabeçalho de proveniência)
src/data/augmentations.py weak/strong augment + ruído do Mean Teacher
src/evaluation.py        avaliação por classe/por amostra do test set
tests/                   27 testes (helpers + 3 suítes)
configs/experiment/x1/   4 YAMLs (supervised, fixmatch, meanteacher, diffrect)
M src/experiment.py      overrides, log_dir, lit_module:, ModelCheckpoint
M src/models/base.py     métricas por classe, report_class_ids, surface_metric_stages
M src/metrics.py         absent_class_to_nan
M src/data/base.py       fallback quando não há stream não-rotulado
M README.md              +61 linhas documentando tudo acima
```

---

## 2. Diagrama do fluxo X1

```
                       inca-video-frame-dataset.csv
                                  │
             split_by_group(paciente_id, seed=42, 70/10/20)      ← fold único, fixo
                                  │
             apply_label_regime(paciente_id, label_fraction)     ← só marca is_labeled
                                  │                                 NÃO descarta linhas
                    ┌─────────────┴──────────────┐
              is_labeled=True              is_labeled=False
              máscara real                 máscara = zeros (placeholder)
                    └─────────────┬──────────────┘
                                  │
                   TwoStreamBatchSampler(bs=8, labeled_bs=4)
                                  │
                    batch = [4 rotulados | 4 não rotulados]
                                  │
              ┌───────────────────┴────────────────────┐
     sup_loss = 0.5*(CE+Dice)              unsup_loss (por método)
     só nas linhas is_labeled              só nas linhas ~is_labeled
              └───────────────────┬────────────────────┘
                loss = sup_loss + w(step) * unsup_loss
```

**A invariante central:** rótulos escondidos viram máscara de zeros, e a seleção supervisionada é
feita por `batch['metadata']['is_labeled']` — **nunca** pelo conteúdo da máscara
([src/models/ssl/base.py:93](src/models/ssl/base.py#L93)). Se alguém supervisionar na máscara
placeholder, o modelo aprende "tudo é fundo" e o X1 inteiro perde o sentido. Isso está coberto por
teste (`TestSupervisionIsolation`).

---

## 3. Camada de dados — o que mudou e o que checar

### 3.1 A hierarquia nova

```
VFSSFrameDatasetBase  (vfss_frame_dataset.py:181)
  └─ VFSSWindowImageDataset  (:505)   ← janela de frames + máscara do frame central
       ├─ VFSSIncaTrain/Val/Test (:668+)      split fixado no __init__
       └─ VFSSUnmWindowImageDataset (unm.py)  outro CSV, outras variantes de alvo
```

A base concentra: carga do CSV, seleção de `target_type`/`target_variant`, split, regime de rótulo,
resolução de caminhos e pré-processamento. A subclasse só implementa `__getitem__`.

### 3.2 `split_by_group` — [vfss_frame_dataset.py:80](src/data/vfss_frame_dataset.py#L80)

Embaralha os grupos com `seed` (só o seed — **nunca** os `ratios`) e caminha na ordem fixa
acumulando contagem de **frames** até bater cada limiar. Duas consequências que valem entender:

1. **Balanceia frames, não grupos.** `split_ratios=[0.7,0.1,0.2]` mira 70% dos *frames* no treino.
2. **É o que garante o aninhamento dos regimes de rótulo** (ver 3.3).

Fold real medido agora:

| split | frames | pacientes | vídeos |
|---|---|---|---|
| train | 690 | 69 | 165 |
| val | 99 | 25 | 25 |
| test | 172 | 20 | 42 |

### 3.3 `apply_label_regime` — [vfss_frame_dataset.py:135](src/data/vfss_frame_dataset.py#L135)

Reusa `split_by_group` com `ratios=(f, 1-f)` e nomes `("labeled","unlabeled")`. Como a ordem dos
grupos depende só do seed, o prefixo revelado em `f1` é prefixo do revelado em `f2 > f1` — daí o
**aninhamento**. É por isso que `LABEL_SEED` tem que ficar fixo: mudá-lo entre orçamentos quebraria
a comparabilidade.

Regimes reais medidos agora (train = 690 frames sempre):

| orçamento | rotulados | não rotulados | pacientes revelados | vídeos revelados |
|---|---|---|---|---|
| 10% | 69 | 621 | **3** | 14 |
| 20% | 163 | 527 | 7 | 36 |
| 50% | 360 | 330 | 25 | 85 |
| 100% | 690 | 0 | 69 | 165 |

> **Isso é a ressalva mais importante do desenho.** O ponto de 10% carrega a variância de *quais* 3
> pacientes caíram, não só de *quantos* rótulos existem. O próprio notebook (célula 12) já alerta
> disso e sugere `LABEL_GROUP_COLUMN="video_id"` como checagem de robustez — 14 vídeos em vez de
> 3 pacientes, com contagem parecida de frames. **Vale rodar**, porque a conclusão do gate depende
> desse ponto da curva.

### 3.4 `TwoStreamBatchSampler` — [src/data/samplers.py:16](src/data/samplers.py#L16)

- Uma "época" = `len(rotulados) // labeled_batch_size`. **Encolhe com o orçamento.**
- O pool não rotulado é um stream infinito auto-reembaralhado.
- Entra no `DataLoader` como `batch_sampler=` (não `sampler=`/`batch_size=`).
- Layout do batch: **rotulados primeiro**, depois não rotulados. Os testes assumem esse layout.

Note o efeito no relógio de treino:

| orçamento | passos/época | épocas até 8000 passos | valida a cada N épocas |
|---|---|---|---|
| 10% | 17 | 470 | 29 |
| 20% | 40 | 200 | 12 |
| 50% | 90 | 88 | 6 |
| 100% | 86 (DataLoader normal) | 93 | 6 |

Em 10%, 8000 passos × 4 rotulados = 32.000 amostras rotuladas vistas sobre **69 frames distintos**
= ~464 passadas. O `ModelCheckpoint` em `val/dice_score` é o que segura o overfitting aqui — sem
ele os números de teste viriam da última época. Confirmado no log: o melhor checkpoint do run de
10% foi no `step=2464`, não no 8000.

### 3.5 `label_fraction=1.0` — [src/data/base.py:110](src/data/base.py#L110)

`TwoStreamBatchSampler` rejeita pool não rotulado vazio (por design). O `DataModuleFromConfig`
detecta isso e cai para o `DataLoader` normal (`shuffle=True`, `batch_size=8`). Consequência: em
100% os métodos SSL degeneram no supervisionado **por construção**, e a "época" volta a ser o
dataset inteiro (86 batches, não 172).

### 3.6 Pré-processamento

- **Imagem** ([:413](src/data/vfss_frame_dataset.py#L413)): min-max **por frame e por canal** → `[0,1]` → reescala para **`[-1,1]`**. Toda `src/data/augmentations.py` assume esse range.
- **Máscara** ([:435](src/data/vfss_frame_dataset.py#L435)): se `target_variant` começa com `multiclass`, resize preservando índices e retorna `long`; senão, binariza `>0`. `mask_interpolation != nearest` com alvo multiclasse agora **levanta erro** — antes misturava índices de classe silenciosamente.
- `repeat_channels=True` + `window_size=1` → saída `[3,H,W]`, casando com `n_channels: 3` do UNet.

---

## 4. Métricas

### 4.1 As quatro métricas e suas convenções de caso degenerado

| Métrica | Classe ausente em pred **e** target | Presente em só um dos dois |
|---|---|---|
| `dice_score(average="none")` | NaN | valor normal |
| `mean_iou(per_class=True)` | **-1.0** ← sentinela | valor normal |
| `average_symmetric_surface_distance` | 0.0 | **NaN** |
| `hausdorff_distance_95` | 0.0 | **NaN** |

`absent_class_to_nan` ([src/metrics.py:10](src/metrics.py#L10)) normaliza o `-1.0` do IoU para NaN,
para que **tudo** possa ser agregado com `.nanmean()`. Sem isso, com `multiclass_c2_c4` a classe 2
(C3) — que nunca aparece — injetaria `-1` em **toda** amostra.

### 4.2 `report_class_ids` — por que não é só cosmético

O modelo tem `n_classes=4` (`0=fundo, 1=C2, 2=C3, 3=C4`), mas **C3 nunca aparece** na variante
`multiclass_c2_c4`. Média sobre todas as classes de primeiro plano dividiria por 3 em vez de 2 e
puxaria todo agregado para baixo. `report_class_ids: {1: C2, 3: C4}`
([src/models/base.py:60](src/models/base.py#L60)) restringe tanto o reporte por classe quanto o
agregado. Indexação: tensores vêm `[B,C]` com `include_background=False`, então classe `c` está na
coluna `c-1`.

### 4.3 `surface_metric_stages`

ASSD/HD95 são laços Python duplos sobre `(batch, classe)` em torno de `edge_surface_distance`:
~170 ms/batch 8×256×256 em GPU contra ~18 s em CPU — ~1000× o custo de IoU/Dice. Por isso o
default é `("test",)`: computar por passo de treino dominaria o wall clock.

---

## 5. Os métodos SSL

### 5.1 O esqueleto — [src/models/ssl/base.py](src/models/ssl/base.py)

```python
supervised_loss = 0.5 * (cross_entropy + dice)      # composição do SSL4MIS 2D
loss            = supervised_loss + w(step) * unsupervised_loss
```

Todo método difere **apenas** no `unsupervised_loss`. Isso é o que faz a comparação X1 ser uma
leitura limpa do efeito da supervisão. `training_step` está em
[base.py:129](src/models/ssl/base.py#L129).

`DiceLoss` vem de [src/third_party/ssl4mis/losses.py](src/third_party/ssl4mis/losses.py) — porte do
SSL4MIS com desvios apenas mecânicos (aceita target em formato índice, one-hot vetorizado, sem
`.cuda()` hard-coded). A aritmética (denominadores quadráticos, `smooth=1e-5`, média sobre todas as
classes **incluindo fundo**) é idêntica, e há teste de equivalência numérica.

`sigmoid_rampup` foi **reimplementado do paper** (arXiv:1610.02242) em vez de copiado do
`code/utils/ramps.py` do SSL4MIS: aquele arquivo carrega cabeçalho CC BY-NC 4.0 (Curious AI) mesmo o
repo sendo MIT, e copiá-lo puxaria cláusula NonCommercial para cá. Boa decisão — está documentada
em [base.py:31](src/models/ssl/base.py#L31).

### 5.2 `SupervisedLitWrapper`

Baseline do orçamento. **Vê os mesmos batches** que os métodos SSL (as linhas não rotuladas passam
pelo forward, só não carregam loss). Manter a composição do batch idêntica é o que isola o efeito
da supervisão.

### 5.3 `MeanTeacherLitWrapper` — [src/models/ssl/mean_teacher.py](src/models/ssl/mean_teacher.py)

Porte fiel de `train_mean_teacher_2D.py`:

| Elemento | Implementação | Onde |
|---|---|---|
| Teacher | `deepcopy` do student, params destacados, `requires_grad_(False)` | `:55` |
| EMA | `alpha = min(1 - 1/(step+1), 0.99)` — "true average until the exponential average is more correct" | `:63` |
| Buffers (BN) | **copiados**, não promediados — como na referência | `:67` |
| Entrada do teacher | **só as linhas não rotuladas**, + `clamp(randn*0.1, ±0.2)` | `:71` |
| Consistência | MSE entre softmax do student e do teacher, nas mesmas linhas | `:78` |
| Avaliação | pelo **teacher** (`evaluate_with_teacher=True`) | `:87` |

O único reescrito é o `add_(scalar, tensor)`, overload removido no torch 2.x.
`configure_optimizers` usa `self.model.parameters()`, então o teacher nunca entra no otimizador —
há teste explícito para isso.

### 5.4 `FixMatchLitWrapper` — [src/models/ssl/fixmatch.py](src/models/ssl/fixmatch.py)

**Divergência deliberada do SSL4MIS.** O `train_fixmatch_standard_augs.py` de lá não é o FixMatch
canônico: adiciona uma *complementary loss* ponderada por entropia. O X1 quer o piso mais simples
possível, então aqui é o FixMatch do paper:

```
vista fraca   → pseudo-rótulo = argmax, mantido onde max softmax ≥ 0.95
vista forte   → CE contra esse pseudo-rótulo, só nos pixels mantidos
```

A vista forte é construída **em cima da vista fraca** e é **fotométrica apenas** — assim o
pseudo-rótulo continua alinhado pixel a pixel e não é preciso warp inverso. Buracos de cutout são
excluídos da loss (`valid`), porque não carregam evidência.

### 5.5 `DiffRectLitWrapper`

Slot de fase 2. `__init__` levanta `NotImplementedError` com instruções. O config, o runner e a
grade do notebook já carregam a célula; `METHODS` no notebook simplesmente não o inclui.

---

## 6. CTAugment — a pergunta direta

### O que "deveria" ser

O FixMatch original define duas famílias de augmentation forte:

- **RandAugment** — sorteia *N* operações de um pool fixo com magnitude aleatória.
- **CTAugment** (*Control Theory Augment*, vindo do ReMixMatch) — mantém, para cada transformação,
  uma distribuição sobre *bins* de magnitude. A cada passo, aplica uma transformação a um exemplo
  **rotulado**, mede o quanto a predição do modelo continua batendo com o rótulo, e **atualiza os
  pesos dos bins online**: magnitudes que preservam a predição ganham peso, as que destroem perdem.
  É um controlador em malha fechada — nada de hiperparâmetro de magnitude a ajustar.

No SSL4MIS existe um `train_fixmatch_cta.py` que segue esse caminho, e é o que um porte "fiel"
usaria.

### O que fizemos no lugar

Uma **política fotométrica fixa**, em [src/data/augmentations.py:112](src/data/augmentations.py#L112):

| Operação | Faixa |
|---|---|
| brightness | ×[0.6, 1.4] |
| contrast | ×[0.6, 1.4] |
| gamma | [0.7, 1.4] |
| gaussian blur (p=0.5) | σ ∈ [0.1, 1.5], kernel 5×5 |
| ruído gaussiano | σ ∈ [0, 0.05] |
| cutout (p=`cutout_prob`=0.5) | quadrado de 10–30% da dimensão, preenchido com cinza médio, **excluído da loss** |

E uma **política fraca geométrica** ([:55](src/data/augmentations.py#L55)): flip horizontal p=0.5,
rotação ±10°, translação ±5%, escala [0.9, 1.1] — parâmetros sorteados uma vez por elemento do batch
e aplicados igualmente a imagem (bilinear) e máscara (nearest).

### Por que a troca é defensável (e onde ela custa)

**A favor:**

1. **Metade do pool do CTAugment não faz sentido aqui.** Color jitter, solarize, posterize,
   equalize, sharpness em cor — as imagens são fluoroscopia em tom de cinza, normalizadas por frame
   para `[-1,1]`. As operações cromáticas ou são no-ops ou introduzem artefato sem contrapartida.
2. **Predição densa quebra a suposição do CTAugment.** O CTAugment mede a degradação num rótulo
   *escalar* de classificação. Em segmentação o rótulo é espacial; qualquer operação geométrica na
   vista forte exigiria warp inverso do pseudo-rótulo. Manter a vista forte fotométrica é
   exatamente o que evita esse acoplamento — e é a solução que a literatura de FixMatch para
   segmentação (UniMatch e derivados) também adota.
3. **O CTAugment é estado que aprende durante o treino, alimentado por exemplos rotulados.** Com
   orçamentos de 3 a 69 pacientes, o controlador teria qualidade de estimativa radicalmente
   diferente entre células da grade — introduziria uma variável confundida exatamente no eixo que
   o X1 quer medir.

**Contra (o custo, para registrar na dissertação):**

- A força da augmentation deixa de ser adaptativa e vira hiperparâmetro fixo, **não ajustado** neste
  dataset. As faixas acima são plausíveis, não calibradas.
- Se o resultado for "FixMatch não ajuda", uma revisão pode legitimamente responder *"vocês não
  usaram a augmentation do paper"*. A defesa é a que está acima — vale deixá-la escrita no texto,
  não só na docstring do módulo.

**Sugestão de ablação barata**, se quiser blindar o argumento: rodar FixMatch em 20% com as faixas
fotométricas ×0.5 e ×1.5. Se a curva mal se mexer, a escolha de política não é o que decide o gate.

### Ponto de atenção na implementação

`weak_augment` e `strong_augment` são **laços Python sobre o batch** com chamadas `torchvision.v2`
por elemento. Custa mais do que uma versão batched, e cada `_rand` faz um sync GPU→CPU. Em batch 8
é aceitável; se subir batch ou resolução, vira gargalo. Não é bug — é dívida a conhecer.

---

## 7. Treino, avaliação e hiperparâmetros

### 7.1 Mudanças em `src/experiment.py`

| Adição | Onde | Para quê |
|---|---|---|
| `overrides=` | `:37` | notebook varre um knob sem escrever YAML por célula; `OmegaConf.merge` (aninhado, não substitui) |
| `log_dir=` | `:41` | notebook roda do próprio diretório; sem isso, `logs/` iria parar dentro de `notebooks/` |
| bloco `lit_module:` | `:94` | seleciona a subclasse do LightningModule. **Sem o bloco, o comportamento é o `LitWrapper` histórico** — configs antigos não mudam |
| `trainer.checkpoint` | `:115` | `ModelCheckpoint` monitorando `val/dice_score`. Sem ele os números de teste vêm da **última** época — injusto justo nos regimes que mais overfittam |
| `self.lit_module` / `self.data_module` | `:59` | expostos para o notebook avaliar o modelo treinado |
| `trainer.test(ckpt_path="best")` | `:177` | testa o melhor checkpoint, não o último |

### 7.2 Hiperparâmetros dos configs X1

Idênticos nos quatro YAMLs, exceto o bloco `lit_module`:

| Parâmetro | Valor | Comentário |
|---|---|---|
| modelo | UNet, `n_channels=3`, `n_classes=4`, `bilinear=True` | C3 existe como saída mas nunca no alvo |
| otimizador | AdamW, `lr=1e-3`, `wd=1e-4` | |
| scheduler | `CosineAnnealingLR`, `T_max=8000`, `eta_min=1e-6`, `interval: step` | `T_max` sobrescrito para `MAX_STEPS` pelo notebook |
| `max_steps` | 8000, `max_epochs=-1` | **compute fixo em passos, não épocas** — premissa do experimento |
| batch | 8, sendo 4 rotulados | |
| `consistency_weight` | 0.1 | SSL4MIS `--consistency` |
| `consistency_rampup_steps` | 200.0 | SSL4MIS `--consistency_rampup` |
| `consistency_rampup_divisor` | **150 (default, não escrito no YAML)** | ver achado **A1** abaixo |
| `consistency_warmup_steps` | 1000 | SSL4MIS: `if iter_num < 1000: consistency_loss = 0` |
| FixMatch `confidence_threshold` | 0.95 | τ do paper |
| Mean Teacher `ema_decay` | 0.99 | SSL4MIS `--ema_decay` |
| dados | `multiclass_c2_c4`, 256², `window_size=1`, `split_seed=42`, `label_seed=42` | janela de 1 frame: **sem componente temporal ainda** |

### 7.3 `src/evaluation.py` — por que existe

O Lightning agrega métrica logada como média **sobre batches**. ASSD/HD95 são NaN quando a classe
aparece em só um dos lados — plausível no regime de 10% — e um batch inteiro NaN envenena a média
da época. `evaluate_per_class` acumula por amostra e reduz com `nanmean` **uma vez só**, e devolve a
tabela por amostra (permite agrupar por paciente/vídeo depois).

Isso é verificável: para o run `meanteacher-lf010`, Dice/C2 bate nas duas rotas (0.5137 no
`console.log` e no CSV), mas ASSD/C2 dá **13.482** pelo Lightning e **13.400** pelo
`evaluate_per_class` — exatamente a diferença que motivou o módulo.

---

## 8. Achados de validação

Ordenados por impacto na leitura dos resultados.

---

### 🔴 A1 — A rampa de consistência nunca sai do chão: o termo SSL é ~2,6% do nominal

**O quê.** `current_consistency_weight` ([base.py:109](src/models/ssl/base.py#L109)) avalia a rampa
em `global_step // consistency_rampup_divisor`, com divisor **150** e `rampup_length=200`. Isso vem
do SSL4MIS, onde `max_iterations=30000`: `30000/150 = 200` → a rampa fecha **exatamente** no fim do
treino, atingindo o peso nominal 0.1.

Aqui `max_steps=8000`. Então `8000/150 ≈ 53` de 200 — a rampa percorre só **27%** do caminho.

| passo | posição na rampa | peso efetivo |
|---|---|---|
| 1000 (fim do warm-up) | 6 | 0.00091 |
| 2000 | 13 | 0.00126 |
| 4000 | 26 | 0.00227 |
| 8000 (fim) | 53 | **0.00671** |

Peso **médio ao longo do treino: 0.0026** — 2,6% do nominal 0.1. O pico (0.0067) é 6,7% do nominal.

**Confirmado empiricamente:** `logs/x1-meanteacher-lf010-s42/.../console.log` fecha com
`train/consistency_weight: 0.006713` e `train/unsup_loss: 0.002525` contra
`train/sup_loss: 0.002316`. O termo de consistência contribuiu **1,7e-5** para uma loss de 2,3e-3 —
menos de 1%.

Pior: o melhor checkpoint do run de 10% foi no `step=2464`, onde o peso era ~0.0014. **A seleção de
modelo aconteceu essencialmente com a consistência desligada.**

**Por que importa.** Os métodos SSL, como configurados, são quase exatamente o supervisionado.
Qualquer conclusão do tipo "FixMatch/Mean Teacher não ajudam neste dataset" está medindo, na
prática, "um termo de consistência com 2,6% do peso pretendido não ajuda" — que é uma afirmação bem
mais fraca e não sustenta o gate.

**Correção.** O divisor precisa acompanhar o orçamento de passos: `divisor = max_steps / 200 = 40`.
Aí a rampa fecha em 8000 passos como fecha em 30000 no SSL4MIS, e o peso médio vai a 0.039.
Uma linha em cada YAML do X1:

```yaml
lit_module:
  params:
    consistency_rampup_divisor: 40   # max_steps/rampup_steps = 8000/200
```

O mesmo vale para o `consistency_warmup_steps: 1000`, que no SSL4MIS é 3,3% do treino e aqui é
12,5%. Escalando: ~270. Menos crítico que o divisor, mas do mesmo tipo de erro.

**Isso invalida a grade já rodada?** Para a leitura "o gate está aberto ou fechado?" (que depende só
da curva do supervisionado), não. Para "SSL recupera quanto da margem?", sim — precisa re-rodar.

---

### 🟠 A2 — As diferenças entre métodos podem não vir do termo de consistência

Se o peso da consistência é ~0.003 (A1), de onde vêm as diferenças observadas?

```
20%:  supervisionado C2 Dice 0.741  |  FixMatch 0.785  |  Mean Teacher 0.768
50%:  supervisionado C2 Dice 0.794  |  FixMatch 0.832  |  Mean Teacher 0.818
```

Dois mecanismos que **não são semi-supervisão** explicam boa parte disso:

1. **Mean Teacher: a avaliação é pelo teacher EMA.** `evaluate_with_teacher=True` faz val/test
   rodarem pelos pesos EMA. Com o termo de consistência quase nulo, o student é praticamente o
   supervisionado — e o ganho vira essencialmente **média de Polyak dos pesos**. É uma
   regularização real e legítima, mas é *weight averaging*, não uso de dados não rotulados.
2. **FixMatch: BatchNorm.** O UNet usa `BatchNorm2d`
   ([src/models/modules.py:15](src/models/modules.py#L15)). O `unsupervised_loss` do FixMatch faz
   **dois forwards extras em modo `train()`** por passo (vista fraca + vista forte), atualizando as
   **running statistics** da BN com dados não rotulados e fortemente augmentados. Isso muda as
   estatísticas usadas na avaliação — sem passar por nenhum gradiente.

**Como separar.** Duas ablações baratas, cada uma isolando um mecanismo:

- Mean Teacher com `evaluate_with_teacher: False`. Se o ganho sumir, era EMA.
- Supervisionado + um forward dummy em `torch.no_grad()` nas linhas não rotuladas em modo train
  (mesmo efeito na BN, zero gradiente). Se ele empatar com FixMatch, era BN.

Não são bugs — os dois comportamentos batem com as respectivas referências. Mas mudam a
**interpretação** do resultado, e é o tipo de coisa que uma banca pergunta.

---

### 🟠 A3 — A célula "degenerada" de 100% não é degenerada para o Mean Teacher

O runner ([célula 15](notebooks/hyphotesis_testing/experiment_1_does_ssl_helps.ipynb)) trata
`método != supervised and fraction >= 1.0` como degenerado e **reusa a linha do supervisionado**,
argumentando que sem stream não rotulado os três métodos colapsam no mesmo.

Isso é verdade para o **student**: sem linhas `~is_labeled`, o ramo de consistência nunca é
executado, nenhum RNG a mais é consumido, e os pesos treinados são idênticos.

Mas **não é verdade para o que é reportado**:

- Mean Teacher avaliaria pelo **teacher EMA**, que é um modelo diferente do student.
- O `ModelCheckpoint` monitora `val/dice_score`, que para o Mean Teacher é computado **pelo
  teacher** — logo a época selecionada como "melhor" também seria outra.

Confirmado no CSV: `meanteacher-lf100-s42` e `fixmatch-lf100-s42` têm valores **idênticos ao dígito**
aos de `supervised-lf100-s42`, porque foram copiados.

**Consequência para a curva:** o Mean Teacher tem o benefício do EMA em 10/20/50% e **não tem** em
100%. A curva dele é internamente inconsistente, e a métrica "% da margem recuperada" (célula 27)
usa o teto do supervisionado — logo compara maçãs com laranjas na ponta de cima.

**Opções.** (a) Rodar de verdade meanteacher@100% (é uma linha: `is_degenerate` retorna False para
`meanteacher`); ou (b) manter o reuso e dizer isso explicitamente no texto da dissertação. A opção
(a) custa um treino e resolve.

---

### 🟡 A4 — Nenhuma augmentation no ramo supervisionado

`weak_augment` só é chamado dentro do `unsupervised_loss` do FixMatch. As linhas **rotuladas** vão
para a `supervised_loss` sem augmentation nenhuma — não há transform aleatório no `Dataset`
(`image_transform` default é só `Resize`).

Isso é **justo** (todos os métodos sofrem igual), mas:

- deprime o desempenho absoluto de todo mundo, principalmente em 10% (69 frames, ~464 passadas);
- no FixMatch canônico a vista fraca é aplicada **também** aos rotulados;
- e infla artificialmente a margem 10%→100%, que é exatamente a quantidade que o gate mede.

Vale ao menos registrar. Se quiser fechar essa lacuna, `weak_augment` já aceita `(image, mask)` e
transforma os dois com os mesmos parâmetros — a plumbing existe.

---

### 🟡 A5 — Detalhes menores

| # | Achado | Onde |
|---|---|---|
| A5.1 | `src/data/vfss.py` ainda define `VFSSIncaTrain/Val/Test` com os mesmos nomes das classes novas. Nenhum config aponta pra lá, mas é uma armadilha de import. | [src/data/vfss.py:84](src/data/vfss.py#L84) |
| A5.2 | `nohup.out` (928 linhas de log) foi commitado em `7fe17b0`. | raiz |
| A5.3 | No reuso da célula degenerada, `output_dir` copiado aponta para o run **do supervisionado**. Confunde rastreabilidade. | notebook, célula 19 |
| A5.4 | `weak_augment`/`strong_augment` são laços Python por elemento do batch, com sync GPU→CPU por sorteio. Aceitável em bs=8; gargalo se escalar. | [src/data/augmentations.py](src/data/augmentations.py) |
| A5.5 | `DiceLoss` inclui o fundo na média (fiel à referência), então C3 — sempre vazia — entra como termo perfeito e dilui o sinal. Fiel, mas note a interação com `report_class_ids`, que **exclui** C3. Loss e métrica não olham para o mesmo conjunto de classes. | [losses.py:73](src/third_party/ssl4mis/losses.py#L73) |
| A5.6 | `window_size=1` em todos os configs X1: o eixo temporal ainda não está em jogo. Coerente com o X1 ser o *gate* para a contribuição temporal, mas vale explicitar. | `configs/experiment/x1/*.yaml` |

---

## 9. Validação prática

### 9.1 O que já está verificado

| Verificação | Resultado |
|---|---|
| Suíte de testes | ✅ **27 testes, OK, 8.7s** (`python -m unittest discover -s tests -t .`) |
| Aninhamento dos regimes (10%⊆20%⊆50%⊆100%) | ✅ reproduzido: 3 → 7 → 25 → 69 pacientes |
| Total de frames de treino constante | ✅ 690 nos quatro orçamentos |
| Test set intocado pelo regime | ✅ `label_regime_splits=("train",)` |
| `evaluate_per_class` usa o melhor checkpoint | ✅ Dice/C2 bate com o `console.log` (0.5137) |
| Grade completa rodada | ✅ 12 células (3 métodos × 4 orçamentos), seed 42, 4128 linhas no CSV |
| Mutation testing da suíte | 17/18 mutantes pegos; o sobrevivente é equivalente (`x*x == x` em one-hot) |

### 9.2 Comandos

```bash
# suíte de testes (CPU, ~9s)
~/miniconda3/envs/vfss/bin/python -m unittest discover -s tests -t . -v

# auditoria do fold, sem GPU (Parte 1 do notebook, células 1–13)

# trajetória real do peso de consistência (achado A1)
~/miniconda3/envs/vfss/bin/python - <<'PY'
import math
r = lambda c,L: math.exp(-5*(1-min(c,L)/L)**2)
for s in (1000,2000,4000,8000):
    print(s, round(0.1*r(s//150,200),5), '->  com divisor 40:', round(0.1*r(s//40,200),5))
PY

# resultados agregados
~/miniconda3/envs/vfss/bin/python -c "
import pandas as pd
d = pd.read_csv('logs/x1/results_per_sample.csv')
print(d.groupby(['run_id','class'])[['dice','iou','assd','hd95']].mean().round(4))"
```

### 9.3 Estado da grade (seed 42)

| método | 10% | 20% | 50% | 100% |
|---|---|---|---|---|
| supervised C2 | 0.5466 | 0.7410 | 0.7937 | 0.8816 |
| supervised C4 | 0.1827 | 0.4815 | 0.6954 | 0.8472 |
| fixmatch C2 | 0.4873 | 0.7850 | 0.8318 | *(reuso)* |
| fixmatch C4 | 0.1475 | 0.5937 | 0.7772 | *(reuso)* |
| meanteacher C2 | 0.5137 | 0.7678 | 0.8181 | *(reuso)* |
| meanteacher C4 | 0.2047 | 0.5654 | 0.7211 | *(reuso)* |

(Dice médio no test set.)

**Margem do supervisionado (10% → 100%): +0.335 Dice em C2 e +0.665 em C4.** O gate está
**amplamente aberto** — há muito a recuperar, e a leitura do X1 não depende dos achados A1–A3, que
afetam só a segunda pergunta ("quanto o SSL recupera dessa margem?").

### 9.4 Ordem sugerida de ação

1. Corrigir `consistency_rampup_divisor` (A1) e re-rodar a grade — é o que dá sentido à comparação SSL.
2. Rodar meanteacher@100% de verdade, ou documentar o reuso (A3).
3. Ablação `evaluate_with_teacher: False` e forward-dummy no supervisionado (A2), para atribuir o ganho.
4. `LABEL_GROUP_COLUMN="video_id"` como checagem de robustez do ponto de 10% (3 pacientes).
5. Mais seeds em `SEEDS` — hoje há uma só, e nenhuma barra de erro.
