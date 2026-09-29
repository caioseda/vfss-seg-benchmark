# experiment_code_report.md

Auditoria técnica do que o repositório `vfss-seg-benchmark` implementa, configura e comprovadamente executou.

- **Data da auditoria:** 2026-09-24.
- **Estado do repositório auditado:** `main @ 95e4467` (2026-09-22 19:45) **com árvore de trabalho suja**: 14 arquivos rastreados modificados (entre eles `src/experiment.py`, `src/callbacks.py`, `src/data/base.py`, `src/data/samplers.py`, `src/data/vfss_frame_dataset.py`) e 33 arquivos não rastreados (entre eles `src/sweep.py`, `src/sweeps/`, `src/tracking.py`, `configs/sweep/`, `configs/experiment/x1-binario-all/`, todas as CSVs em `logs/`). "O código" neste relatório é o da árvore de trabalho, salvo indicação contrária.
- **Fontes lidas:** código em `src/`, `scripts/`, `tests/`; todos os YAML em `configs/`; `logs/` inteiro (CSVs de resultado, `hparams.yaml`, `metrics.csv`, `run.json`, `state.json`, `console.log`, nomes de checkpoint, cache `predictions/*.npz`, manifesto de migração, `_archive/`); saídas salvas dos notebooks em `notebooks/hyphotesis_testing/`; `docs/*.md` e `README.md` (tratados como **alegações**, não como evidência); o dataset em disco (`/data_ssd/caioseda/data/dataset_inca_all_frames/`, `/data_ssd/caioseda/data/ACDC/`), somente leitura.

## Convenções

| Rótulo | Significado neste relatório |
|---|---|
| **IMPLEMENTADO** | Existe código que realiza a funcionalidade. |
| **CONFIGURADO** | Existe YAML (experimento ou sweep) que a aciona. |
| **EXECUTADO** | Há artefato de execução: diretório de run com `metrics.csv` que chega ao passo final e/ou checkpoint, `console.log` de treino. |
| **RESULTADO ENCONTRADO** | Há métrica de teste gravada (CSV de resultado, `metrics.csv` com linha `test/*`, ou `results.csv` de benchmark). |
| **NÃO É POSSÍVEL CONFIRMAR EXECUÇÃO** | Há código/config, mas nenhum artefato de execução foi encontrado. |

Status técnico por experimento (seção 4) usa só: *implementado*, *configurado*, *execução comprovada*, *execução não comprovada*.

**Como os números agregados deste relatório foram obtidos.** As CSVs de resultado guardam uma linha por (amostra de teste × classe). Todo agregado que aparece aqui foi calculado por mim, a partir dessas CSVs, **com a mesma regra do código** (`src/evaluation.py:117`, `aggregate_per_class`: `nanmean` sobre as amostras, por classe). Nenhum valor por amostra foi alterado. Agregados são exibidos com 4 casas decimais (arredondamento só de exibição). Para o ACDC, o agregado é a média por classe sobre os volumes e, na coluna "macro", a média das 3 classes (mesmo resultado que o notebook do ACDC imprime, conferido na seção 7.6).

**Verificação de integridade das CSVs.** Para 5 runs de 4 famílias diferentes, recalculei as métricas por amostra a partir do cache de previsões (`logs/<família>/predictions/<run>.npz` + `test_frames.npz`) com o código atual (`src/evaluation.py:_per_sample_metrics`). Diferença máxima contra a CSV: Dice/IoU 0.0; ASSD/HD95 ≤ 6e-06 (vem de a CSV do X1 guardar a penalidade como 362.0387 em vez de 362.038672…). Nenhuma divergência de NaN. Runs conferidos: `x1 supervised-lf025-s42`, `x1 diffrect-lf100-s42`, `x1-frames-per-video meanteacher-fpv03-s42`, `x1-frames-per-video-multiclass-all fixmatch-fpv02-s42`, `x1-frames-per-video-binario-all diffrect-fpv02-s42`.

---

## 1. Estrutura experimental do projeto

O código organiza experimentos em três camadas.

1. **Experimento** (`src/experiment.py`, classe `Experiment`). Carrega um YAML `type: experiment` (padrão `target`/`params` instanciado por `src/utils.py:instantiate_from_config`), aplica `overrides` via `OmegaConf.merge`, chama `pl.seed_everything(seed)` (`src/experiment.py:340`), instancia `lit_module` (se ausente, `LitWrapper`), `DataModuleFromConfig`, `pl.Trainer(**trainer.params)`, treina, e testa com `ckpt_path="best"` quando existe bloco `trainer.checkpoint`. Grava TensorBoard + CSV logger. O código atual também grava `config.yaml`, `run.json` (git sha, seed, host, status) e um `last.ckpt` de retomada — **só os 16 runs de `x1-frames-per-video-binario-all` têm `run.json`/`config.yaml`**; todos os demais runs são anteriores a essa funcionalidade.
2. **Benchmark** (`src/benchmark.py`). Lê um YAML `type: benchmark` com uma lista de experimentos, roda em sequência e escreve `results.csv`. Usado apenas na fase de comparação de arquiteturas (fev–ago/2026); os artefatos estão em `logs/_archive/benchmarks/`.
3. **Sweep** (`src/sweep.py` + `src/sweeps/{base,x1,acdc}.py` + `configs/sweep/*.yaml`, todos **não rastreados no git**). Produto cartesiano de eixos → uma célula por combinação → um `Experiment` com overrides. Após o treino, a célula é avaliada no teste por `SweepPlan.evaluate` (`evaluate_per_class` para VFSS, `evaluate_volumes` para ACDC) e as linhas por amostra são anexadas à CSV da família (`logs/<família>/results_per_sample.csv` ou `results_per_volume.csv`). Células já presentes na CSV são puladas.

Antes do runner de sweep, as grades eram executadas dentro dos notebooks de `notebooks/hyphotesis_testing/` (INFERÊNCIA A PARTIR DO CÓDIGO E DOS ARTEFATOS: as famílias `x1`, `x1b`, `acdc`, `x1-frames-per-video` e `x1-frames-per-video-multiclass-all` não têm `_sweep/state.json`; `logs/x1/_sweep/console.log` contém apenas execuções `--dry-run`; os notebooks contêm células com `Experiment(...)`/`train_and_evaluate`; `src/sweep.py` não existe no histórico do git). A única família com evidência de execução via sweep é `x1-frames-per-video-binario-all` (`_sweep/state.json`, `_sweep/console.log` com `[treina] ...`).

**Layout de `logs/`** (após a migração de 2026-09-23 14:00:20, manifesto em `logs/_migration/2026-09-23-14-00-20/manifest.json`, 120 movimentos):

```
logs/<família>/results_*.csv        registro por amostra (+ cópia .bak pré-migração)
logs/<família>/runs/<nome>/<uid>/   hparams.yaml, metrics.csv, tfevents, checkpoints/best-*.ckpt, console.log
logs/<família>/predictions/*.npz    previsões de teste em cache (não existe para x1b e acdc)
logs/_archive/unclaimed/            30 runs que nenhuma CSV referencia
logs/_archive/incomplete/           3 runs sem checkpoints/
logs/_archive/benchmarks/ e 10 dirs "era das arquiteturas"
```

Os arquivos `.bak` diferem das CSVs **apenas** na coluna `output_dir` (caminho antigo vs. novo) — verificado coluna a coluna para as 6 CSVs que têm `.bak`.

---

## 2. Dataset no código

### 2.1 VFSS (INCA) — o dataset de todos os experimentos X1

**Caminho.** `/data_ssd/caioseda/data/dataset_inca_all_frames/` (em todos os YAML das famílias X1). Tabela de frames: `inca-video-frame-dataset.csv` (`src/data/vfss_frame_dataset.py:23`).

**Contagens verificadas em disco:**

| Item | Valor |
|---|---|
| Linhas da tabela (frames anotados) | 961 (961 `video_frame` únicos) |
| Vídeos na tabela | 232 |
| Pacientes (`paciente_id`) | 114 |
| Diretórios em `frames/` | 234 |
| PNGs em `frames/` (todos os vídeos) | 46.137 |
| Arquivos por variante de máscara (`targets/mask/<variante>/`) | 971 (5 rotuladores: AM, BB, CS, VC, VR) |
| `selected_labeler` | VC 447, BB 185, CS 144, VR 121, AM 64 |
| `momento` | pos 632, pre 329 |

**Variantes de alvo** (`TARGET_VARIANTS`, `src/data/vfss_frame_dataset.py:18`): `raw`, `binary_c2_c4`, `binary_all`, `multiclass_c2_c4`, `multiclass_all`. Em disco existem também `binary_c2_c3_c4` e `multiclass_c2_c3_c4` (não aceitas pela lista do VFSS INCA). `binary_all` foi adicionada à lista na árvore suja (diff em `vfss_frame_dataset.py`); o diretório `targets/mask/binary_all/` tem data de criação 2026-09-23 15:30.

Verificação pixel a pixel (971 arquivos): `binary_all == (multiclass_all > 0)` — 0 divergências; `binary_c2_c4 == isin(multiclass_all, {1, 3})` — 0 divergências. Uma amostra de `multiclass_c2_c4` contém os valores {0, 1, 3} (índice 2 ausente).

**Pré-processamento da máscara** (`_preprocess_mask`, linha 672): variantes `multiclass_*` preservam o índice de classe (resize `nearest` obrigatório, validado na linha 533); qualquer outra variante é binarizada por `(mask > 0)`.

**Classes.** O código não nomeia classes; os nomes vêm dos comentários dos YAML (`0=fundo, 1=C2, 2=C3, 3=C4, 4=C5, 5=C6, 6=C7, 7=C8` em `multiclass_all`). O que é verificável: os índices presentes e em quantos frames aparecem.

Frames que contêm cada índice em `multiclass_all` (nomes conforme os YAML), por split `seed=42` (definido abaixo):

| | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|
| todos (961) | 961 | 961 | 961 | 894 | 528 | 199 | 18 |
| train (690) | 690 | 690 | 690 | 656 | 385 | 147 | 18 |
| val (99) | 99 | 99 | 99 | 90 | 46 | 13 | 0 |
| test (172) | 172 | 172 | 172 | 148 | 97 | 39 | 0 |

C8 não aparece em nenhum frame de validação ou teste.

**Splits.**

- `split_mode: seed` (todas as famílias X1): `split_by_group` (linha 84) embaralha os **pacientes** com `RandomState(42)` e os atribui gulosamente até atingir 70/10/20% dos *frames*. Resultado verificado: train 690 frames / 69 pacientes / 165 vídeos; val 99 / 25 / 25; test 172 / 20 / 42. Nenhum paciente nem vídeo em mais de um split.
- `split_mode: legado` (padrão do código; usado pelos configs da era das arquiteturas): usa a coluna `split_legado` do CSV: train 559, val 80, test 160, **NaN 162** — sob esse modo, 162 frames anotados não pertencem a nenhum split.

**Regimes de rótulo** (somente no split `train`; val/test sempre rotulados). Nenhum frame é descartado: frames com rótulo escondido ficam no dataset com `is_labeled=False` e máscara de zeros (linha 892).

- Por fração de pacientes — `apply_label_regime` (linha 139), `label_group_column: paciente_id`, `label_seed: 42`. Verificado:

| `label_fraction` | frames revelados (de 690) | pacientes revelados (de 69) | vídeos com rótulo (de 165) |
|---|---|---|---|
| 0.25 | 176 | 10 | 39 |
| 0.50 | 360 | 25 | 85 |
| 0.75 | 519 | 43 | 124 |
| 1.00 | 690 | 69 | 165 |

- Por contagem por vídeo — `apply_label_regime_by_count` (linha 185), `label_group_column: video_id`, `label_seed: 42`. Distribuição de frames anotados por vídeo de treino: 2 frames → 1 vídeo; 3 → 10; 4 → 112; 5 → 42 (máximo 5). Verificado:

| `label_frames_per_group` | frames revelados (de 690) | vídeos com rótulo |
|---|---|---|
| 1 | 165 | 165 |
| 2 | 330 | 165 |
| 3 | 494 | 165 |
| 5 | **690** | 165 |

Com `frames_per_group=5`, todos os 690 frames de treino são revelados — o mesmo conjunto revelado que `label_fraction=1.0`.

**Pool de frames nunca anotados** — `build_unlabeled_frame_pool` (linha 232), só de vídeos do split `train`, excluindo os frames anotados. Verificado: 31.953 frames elegíveis (165 vídeos); com `frame_stride=3`, 10.699; com teto `unlabeled_pool_size=8000` (seed 42), 8.000 (165 vídeos). X1, X1-fpv, X1-fpv-multiclass-all e X1-fpv-binario-all usam 8000/stride 3; X1B usa o pool denso (sem teto, stride 1 → 31.953, confirmado pela saída salva do notebook do X1B).

**Amostragem de batch** — `TwoStreamBatchSampler` (`src/data/samplers.py:51`): batch 8 = 4 rotulados + 4 não rotulados (em todos os configs VFSS). Uma "época" = `len(rotulados) // 4` batches. Sorteio não rotulado uniforme com reposição, RNG `RandomState(seed + época)`. Políticas: `global`, `same_video` (frame não rotulado do mesmo vídeo de um dos rotulados do batch), `temporal_window` (±W frames); fallback `temporal_window → same_video → global` com contagem em `draw_counts`.

**Pré-processamento da imagem** (`_preprocess_image`, linha 650): leitura em escala de cinza, min-max por imagem para [0, 1], reescala para [−1, 1], `Resize(256)` bilinear; máscara `Resize(256)` nearest. `repeat_channels: True` → 3 canais. `window_size: 1` (um frame por amostra) em todos os configs.

**Augmentations.** O dataset VFSS **não aplica nenhuma augmentation** (só o `Resize`; nenhum config passa `image_transform`). As augmentations existentes estão dentro dos wrappers (`src/data/augmentations.py`) e diferem por método — ver tabela em 4.

### 2.2 ACDC — teste de aceitação do port SSL

- Caminho `/data_ssd/caioseda/data/ACDC` (preparado por `scripts/prepare_acdc.py`). Listas verificadas: `train.list` 140, `val.list` 20, `test.list` 40, `train_slices.list` 1312.
- 4 classes (0 fundo, 1 RV, 2 Myo, 3 LV, conforme configs). Imagens 1 canal, 256×256.
- Rótulos por número de pacientes: `ACDC_PATIENTS_TO_SLICES` (`src/data/acdc_dataset.py:49`); os runs usam `labeled_patients` 7 (“5%”, 136 fatias) e 14 (“10%”, 256 fatias) — confirmado nos `hparams.yaml` (8 runs cada).
- Augmentation no dataset: `random_generator` (`src/data/acdc_dataset.py:107`; rot90+flip com p=1/2, senão rotação ±20° com p=1/4), ligada por padrão no split `train` (linha 172), RNG `default_rng([seed=1337, worker+1])`.
- Avaliação volumetrica (`src/evaluation_volume.py:103`): Dice/Jaccard/HD95/ASD por volume. `evaluate_volumes` **não passa `spacing`** → HD95/ASD em unidades de voxel, não em mm.

### 2.3 UNM

`src/data/unm.py`, caminho `/data_ssd/caioseda/data/dataset_unm` (existe: `frames/`, `targets/`, `unm-video-frame-dataset.csv`). Referenciado só pelos configs da era das arquiteturas. Conteúdo não auditado.

---

## 3. Métodos implementados

| Método | Tipo | Implementado | Evidência |
|---|---|---|---|
| `LitWrapper` (CE simples) | Supervisionado (base) | Sim | `src/models/base.py:17`; loss `cross_entropy` em `:125`. Usado quando o YAML não tem `lit_module` (era das arquiteturas). |
| `SupervisedLitWrapper` | Supervisionado no orçamento de rótulo | Sim | `src/models/ssl/supervised.py:11`; termo não supervisionado ≡ 0 (`:19`); loss `0.5·(CE+Dice)` herdada de `src/models/ssl/base.py:235`. Linhas não rotuladas passam pelo forward sem loss. |
| `MeanTeacherLitWrapper` | Semi-supervisionado | Sim | `src/models/ssl/mean_teacher.py:32`; EMA `alpha=min(1-1/(step+1), 0.99)` (`:62`); MSE entre softmax aluno/professor nas linhas não rotuladas, ruído `clamp(randn·0.1, ±0.2)` na entrada do professor (`:71`); val/test pelo professor (`evaluate_with_teacher: True`). |
| `FixMatchLitWrapper` (canônico) | Semi-supervisionado | Sim | `src/models/ssl/fixmatch.py:27`; pseudo-rótulo da view fraca com limiar 0.95, CE na view forte só nos pixels mantidos e fora do cutout (`:45`). |
| `DiffRectLitWrapper` | Semi-supervisionado + retificador por difusão latente | Sim | `src/models/ssl/diffrect.py:57`, `compute_training_losses` `:292`, módulos em `src/models/ssl/diffrect_modules.py` (`RectificationNet` `:299`), difusão em `src/third_party/diffrect/diffusion.py`. Uma única otimização com dois grupos de parâmetros. |
| DiffRect `semi_supervised_base: canonical` | Variante | Implementado, **nunca usado** | `src/models/ssl/diffrect.py:464`; todos os 28 runs DiffRect com `hparams.yaml` encontrados (20 nas CSVs + 8 arquivados) têm `semi_supervised_base: reference` no `hparams.yaml`. |
| Rampa de consistência por fração do orçamento | Cronograma | Sim | `src/models/ssl/base.py:60,192`; em todos os runs das CSVs: `consistency_rampup_fraction 1.0`, `consistency_warmup_fraction 0.0333`, `consistency_weight 0.1`. Runs arquivados de 2026-09-08 usam a versão anterior (`consistency_rampup_steps 200`, `consistency_rampup_divisor 150`, `consistency_warmup_steps 1000`). |
| Políticas de amostragem `global` / `same_video` / `temporal_window` | Amostragem do fluxo não rotulado | Sim | `src/data/samplers.py:25,172,206`. |
| UNet (repo) | Modelo | Sim | `src/models/unet.py` (17.3 M parâmetros segundo o sumário do Lightning no `console.log`). Usado em todos os runs VFSS X1 e no eixo `backbone: repo` do ACDC. |
| SSL4MIS UNet | Modelo | Sim | `src/third_party/ssl4mis/unet.py:146`; eixo `backbone: ssl4mis` do ACDC. |
| UNet++ | Modelo | Sim | `src/models/unet_plus_plus/adapters.py:6` |
| TransUNet (B16, R50-B16) | Modelo | Sim | `src/models/trans_unet/adapters.py:9`; pesos em `model_weights/vit/` |
| SwinUNet (tiny/small/base) | Modelo | Sim | `src/models/swin_unet/adapters.py:10`; pesos em `model_weights/swin-transformer/` |
| Métricas 2D: Dice, IoU, ASSD, HD95 | Avaliação | Sim | `src/evaluation.py:29`, `src/metrics.py:101,131` |
| Métricas volumétricas: Dice, Jaccard, HD95, ASD | Avaliação | Sim | `src/metrics.py:194,255`, `src/evaluation_volume.py:103` |

**Convenções de casos degenerados no código atual** (`src/metrics.py`): classe ausente em pred e GT → Dice NaN, IoU NaN (sentinela −1 convertida), ASSD/HD95 0.0; classe presente em só um lado → Dice 0, ASSD/HD95 = diagonal da imagem (362.0387 px em 256×256). A penalidade pela diagonal entrou no git no commit `95e4467` (2026-09-22); antes, esse caso gerava NaN (ver 8.3).

---

## 4. Experimentos identificados

### 4.1 Benchmark de arquiteturas (era pré-SSL)

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** comparar UNet, UNet++, TransUNet e SwinUNet em segmentação binária de VFSS (INCA e UNM). INFERÊNCIA A PARTIR DO CÓDIGO (README e nomes dos configs).

**Dataset:** INCA (`dataset_inca_all_frames`) e UNM (`dataset_unm`); configs mais antigos (`configs/benchmark/vfss-swinunet.yaml`) usam `src.data.vfss.VFSSIncaTrain` com `data_root: ../dados_inca/metadados` (pacote externo `vfss_data_split`).

**Classes:** `n_classes: 2` em todos os `hparams.yaml` examinados (benchmarks 014 e 016). `target_variant` não é registrado nesses `hparams.yaml`; nos configs atuais é `raw` (binarizado por `>0`) ou `binary_c2_c4` (`vfss-baseline-c2-c4-only.yaml`).

**Estratégia de labels:** 100% supervisionado; split `legado` pelo config atual (não registrado nos artefatos).

**Modelos:** UNet, UNet++, TransUNet (B16, R50-B16, "official"), SwinUNet (tiny, small, base, tiny sem pré-treino).

**Loss:** `LitWrapper.compute_loss` = CE.

**Augmentations:** nenhuma no código atual do dataset VFSS.

**Seeds:** não registradas nos artefatos. `Experiment.run(seed=42)` é o padrão desde o commit `5274a23` (fev/2026) e `Benchmark` não passa seed → 42. INFERÊNCIA A PARTIR DO CÓDIGO.

**Métricas:** `test/dice_score`, `test/mean_iou`, `test/loss` (média Lightning por batch).

**Artefatos encontrados:** `logs/_archive/benchmarks/{vfss-baseline (16 dirs), vfss-transunet (5), vfss-swinunet (10)}`; 10 dirs de runs avulsos em `logs/_archive/vfss-*`. `results.csv` em 9 diretórios de benchmark.

**Resultados encontrados:** seção 7.8. Os benchmarks `vfss-baseline/005`, `/011` e `/015` contêm a string `fast_dev_run` nos `console.log` (8, 6 e 10 ocorrências) — são smoke tests; `/015` só tem `console.log` nos subdiretórios.

**Evidência da execução:** `logs/_archive/benchmarks/vfss-baseline/014-2026-05-09-17-53-57/results.csv`, `.../016-2026-08-13-19-21-27/results.csv` (sem `fast_dev_run`, com `csv/metrics.csv` e tfevents por experimento).

### 4.2 ACDC — reprodução do port SSL

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** medir os 4 wrappers num benchmark com números publicados, como teste de aceitação do port (comentários de `configs/sweep/acdc.yaml` e `src/sweeps/acdc.py`). INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** ACDC, 140/20/40 volumes (seção 2.2).

**Classes:** 4 (fundo, RV, Myo, LV); reportadas RV, Myo, LV.

**Estratégia de labels:** `labeled_patients` 7 (5%) ou 14 (10%); fatias não rotuladas = restantes do treino (dataset totalmente anotado; rótulos escondidos).

**Modelos:** eixo `backbone`: `ssl4mis` (`src.third_party.ssl4mis.SSL4MISUNet`) e `repo` (`src.models.unet.UNet`, `n_channels: 1`). Confirmado por run no `hparams.yaml` (`model/target`).

**Loss:** SSL wrappers (seção 3). Otimizador SGD lr 0.01, momentum 0.9, wd 1e-4; `PolynomialLR(total_iters=30000, power=0.9)`; `max_steps 30000`; batch 6 (3 rotulados); política `global`.

**Augmentations:** `random_generator` no dataset de treino (ver 2.2) + augmentations internas de FixMatch/DiffRect/MT. A existência de `random_generator` no código em 2026-09-10/11 (datas dos runs) não é verificável (ver seção 9).

**Seeds:** 1337 (seed global, sampler, dataset).

**Métricas:** volumétricas por volume (val e test) na CSV; seleção de checkpoint por `val/dice_volume` (callback `VolumetricValidation`).

**Artefatos encontrados:** `logs/acdc/results_per_volume.csv` (2880 linhas = 16 runs × 60 volumes × 3 classes); `logs/acdc/runs/` (16 runs, 1 checkpoint `best-*` cada, `max_step` 30000 em todos); 5 runs ACDC em `_archive/unclaimed/` (tentativas anteriores, seção 7.9).

**Resultados encontrados:** seção 7.6. Nenhuma linha com NaN; maior HD95 = 94.918 (nenhum volume caiu no caso degenerado "classe só de um lado").

**Evidência da execução:** CSV acima; `logs/acdc/runs/*/*/csv/metrics.csv` (o máximo de `val/dice_volume` de cada run coincide com o Dice macro de validação calculado da CSV em todos os 16 runs); saída salva do notebook `experiment_acdc_ssl_reproduction.ipynb` (células 15, 16, 19).

### 4.3 X1 — orçamento por fração de pacientes, `multiclass_c2_c4`

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** comparar supervisionado × FixMatch × Mean Teacher × DiffRect em 4 orçamentos de rótulo com um fluxo não rotulado de frames nunca anotados presente em todos os orçamentos (docstrings de `build_unlabeled_frame_pool` e `configs/sweep/x1.yaml`). INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** VFSS INCA, `split_mode: seed`, `split_seed 42`, 70/10/20 por paciente; teste = 172 frames.

**Classes:** `n_classes: 4` (0 fundo, 1 C2, 2 C3, 3 C4); `target_variant: multiclass_c2_c4`; reportadas C2 e C4 (`report_class_ids {1: C2, 3: C4}`); o índice 2 nunca aparece nas máscaras desta variante.

**Estratégia de labels:** `label_fraction ∈ {0.25, 0.50, 0.75, 1.00}` por paciente (`label_group_column: paciente_id`, `label_seed 42`), pool 8000 (stride 3), política `same_video`, batch 4+4. **Observação de protocolo verificada:** nenhum dos 12 runs semi-supervisionados do X1 registrou `train/pseudo_*` ou `train/teacher_*` (métricas logadas só quando o batch contém linha não rotulada com GT escondida — `src/models/ssl/base.py:hidden_targets`). Como o regime esconde pacientes inteiros e `same_video` sorteia do vídeo do frame rotulado, os frames anotados com rótulo escondido não entram no fluxo não rotulado; esse fluxo é, na prática, o pool nunca anotado. INFERÊNCIA A PARTIR DO CÓDIGO, sustentada pela ausência dessas chaves em todos os `metrics.csv` do X1 (as mesmas chaves existem nos runs X1-fpv com fpv 1, 2 e 3).

**Modelos:** UNet do repo (3 canais de entrada).

**Loss:**
- supervisionado: `0.5·(CE + Dice)` nas linhas rotuladas;
- FixMatch: + `w(t)·CE(view forte, pseudo-rótulo)`, limiar 0.95, `cutout_prob 0.5`;
- Mean Teacher: + `w(t)·MSE(softmax)`, EMA 0.99;
- DiffRect (`reference`): + `w(t)·L_semi` (CE+Dice contra pseudo-rótulo min-max normalizado com limiar 0.8, + perda complementar) + `refine_loss` + `rect_loss` a partir de 3,33% do orçamento;
- `w(t) = 0.1·exp(−5(1−t/T)²)`, zero nos primeiros 3,33% de `T = max_steps = 8000`.

AdamW lr 1e-3, wd 1e-4; `CosineAnnealingLR(T_max=8000, eta_min=1e-6)` por passo; `max_steps 8000`; `check_val_every_n_epoch` 11/6/4/3 (lf 0.25/0.50/0.75/1.00); checkpoint por `val/dice_score` (média C2/C4, por fatia).

**Augmentations:**

| Método | Linhas rotuladas | Linhas não rotuladas |
|---|---|---|
| supervised | nenhuma | nenhuma (só forward) |
| meanteacher | nenhuma | aluno: nenhuma; professor: ruído gaussiano |
| fixmatch | nenhuma | fraca (flip p=0.5, rotação ±10°, translação ±5%, escala 0.9–1.1) + forte fotométrica (brilho/contraste/gamma, blur p=0.5, ruído, cutout p=0.5) |
| diffrect | **fraca** (a loss supervisionada usa a view fraca, `supervise_weak_view: True`) | fraca + forte fotométrica, sem cutout |

**Seeds:** 42 (global, sampler, split, label, pool). Uma seed por célula.

**Métricas:** Dice, IoU, ASSD, HD95 por frame de teste e classe (C2, C4).

**Artefatos encontrados:** `logs/x1/results_per_sample.csv` (5504 linhas = 16 × 172 × 2, com coluna `undetected`); `logs/x1/results_per_sample_pre_a2_fix.csv` (mesmas linhas com ASSD/HD95 NaN no lugar da penalidade); `logs/x1/runs/` (16 runs de 2026-09-14 16:43 a 2026-09-15 01:34, `max_step` 8000, 1 checkpoint cada); `logs/x1/predictions/` (16 + `test_frames.npz`, datados de 2026-09-17); 20 runs anteriores em `_archive/unclaimed/` (seção 7.9).

**Resultados encontrados:** seções 7.1 e 7.2.

**Evidência da execução:** CSVs e diretórios acima; notebook `experiment_1_does_ssl_helps.ipynb` com saídas (células 35 e 37) que batem com a CSV pós-correção; `tests/test_sweeps.py` (rodado nesta auditoria, OK) confere 16 células do X1 contra os `hparams.yaml`.

### 4.4 X1-binário — orçamento por fração de pacientes, `binary_c2_c4`

**Status técnico:** configurado; execução não comprovada.

**Objetivo inferível pelo código:** mesma grade do X1, treinando desde o início fundo × estrutura (C2+C4 fundidos). INFERÊNCIA A PARTIR DO CÓDIGO (comentários dos YAML).

**Dataset / Estratégia de labels / Loss / Augmentations / Seeds:** idênticos ao X1 (os configs de `configs/experiment/x1-binario/` diferem dos de `x1/` só em `name`, `n_classes`, `report_class_ids` e `target_variant`).

**Classes:** `n_classes: 2`, `target_variant: binary_c2_c4`, reportada `FG`.

**Modelos:** UNet do repo.

**Métricas:** as do X1, com uma classe.

**Artefatos encontrados:** `configs/sweep/x1-binario.yaml`, `configs/experiment/x1-binario/*.yaml`, notebook `experiment_1_does_ssl_helps_binary.ipynb` **com 0 células com saída**. Não existe `logs/x1-binario/`. Nenhum diretório de run com prefixo `x1-binario` em `logs/` ou `_archive/`.

**Resultados encontrados:** nenhum.

**Evidência da execução:** nenhuma. `tests/test_sweeps.py` reporta 0 células conferidas para `x1-binario`. NÃO É POSSÍVEL CONFIRMAR EXECUÇÃO.

### 4.5 X1B — política de amostragem do fluxo não rotulado

**Status técnico:** execução comprovada (parcial em relação ao config atual).

**Objetivo inferível pelo código:** comparar políticas `global`, `same_video` e janelas temporais com orçamento fixo em 100% (docstring de `src/data/samplers.py`, `configs/sweep/x1b.yaml`). INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** VFSS INCA, mesmo fold do X1; pool denso (31.953 frames, sem teto, stride 1, `use_unlabeled_pool: true`).

**Classes:** idênticas ao X1 (4 saídas, C2/C4 reportadas).

**Estratégia de labels:** `label_fraction 1.0` (todos os 690 frames de treino rotulados); políticas executadas: `global`, `same_video`, `temporal_window` W=5 e W=10.

**Modelos / Loss / Augmentations:** como no X1 (reusa `configs/experiment/x1/`). `check_val_every_n_epoch: 3`.

**Seeds:** 42.

**Métricas:** as do X1 + `policy_served`, `policy_fallback_global` na CSV (1.0 e 0.0 em todos os 12 runs).

**Artefatos encontrados:** `logs/x1b/results_per_sample.csv` (4128 linhas = 12 × 344); `logs/x1b/runs/` (12 runs, 2026-09-10 11:17 a 14:07); sem `predictions/`; `_archive/unclaimed/x1b-supervised-W15-s42/2026-09-10-14-21-56` interrompido no passo 4149.

**Resultados encontrados:** seção 7.5. **ASSD/HD95 desta CSV usam a convenção antiga (NaN + `nanmean`)**: 55 linhas NaN, 0 linhas com a penalidade; nunca reprocessada.

**Evidência da execução:** CSV e runs acima. Grade executada = 4 políticas × 3 métodos (sem DiffRect, sem W=15). O config atual pede 5 políticas × 4 métodos (ver 8.1).

### 4.6 X1 por frames-por-vídeo, `multiclass_c2_c4` (4 classes)

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** mesma comparação do X1 com o orçamento definido como N frames anotados por vídeo (docstring de `X1FramesSweep`). INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** como o X1.

**Classes:** como o X1 (4 saídas, C2/C4 reportadas).

**Estratégia de labels:** `label_frames_per_group ∈ {1, 2, 3, 5}`, `label_group_column: video_id` (os `hparams.yaml` também registram `label_fraction: 1.0`, ignorado pelo código quando `label_frames_per_group` está definido — `vfss_frame_dataset.py:469`). Pool 8000/stride 3, `same_video`. `check_val_every_n_epoch` 12/6/4/3.

**Modelos / Loss / Augmentations:** como o X1 (mesmos YAML de `configs/experiment/x1/`).

**Seeds:** 42.

**Métricas:** como o X1 (sem coluna `undetected`).

**Artefatos encontrados:** `logs/x1-frames-per-video/results_per_sample.csv` (5504 linhas); `runs/` (16 runs, 2026-09-17 23:10 a 2026-09-18 20:11); `predictions/` (16 + `test_frames.npz`); `_archive/incomplete/x1-frames-fixmatch-fpv01-s42/2026-09-17-23-39-04` (parou no passo 249) e `_archive/unclaimed/x1-frames-diffrect-fpv01-s42/2026-09-18-13-14-09` (passo 1699).

**Resultados encontrados:** seção 7.3. ASSD/HD95 com a convenção da penalidade (0 NaN).

**Evidência da execução:** CSV, runs, notebook `experiment_1_does_ssl_helps_frames_per_video.ipynb` (saídas das células 9–24 batem com as contagens da seção 2.1), `tests/test_sweeps.py` (16 células).

### 4.7 X1 por frames-por-vídeo, `multiclass_all` (8 classes)

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** mesma grade de 4.6 com todas as vértebras anotadas como classes distintas. INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** como o X1.

**Classes:** `n_classes: 8`; reportadas C2..C8 (`report_class_ids` 1..7). C8 não existe em val/test (seção 2.1): Dice/IoU de C8 são NaN em todas as 172 linhas de todos os 16 runs; ASSD/HD95 de C8 = 0.0 (convenção "ausente dos dois lados").

**Estratégia de labels / Modelos / Loss / Augmentations / Seeds:** como 4.6, com os YAML de `configs/experiment/x1-multiclass-all/`.

**Métricas:** Dice, IoU, ASSD, HD95 por classe (7 classes).

**Artefatos encontrados:** `logs/x1-frames-per-video-multiclass-all/results_per_sample.csv` (19264 linhas = 16 × 172 × 7); `runs/` (16 runs, 2026-09-21 16:14 a 2026-09-22 11:32); `predictions/`; `_archive/unclaimed/x1-frames-diffrect-fpv02-s42/2026-09-21-19-22-36` (passo 4799, `multiclass_all`) e `x1-frames-meanteacher-fpv05-s42/2026-09-22-02-26-05` (passo 4987).

**Resultados encontrados:** seção 7.4.

**Evidência da execução:** CSV, runs, notebook `experiment_1_does_ssl_helps_frames_per_video_multiclass_all.ipynb`, `tests/test_sweeps.py` (16 células; o teste isenta explicitamente o campo `name`, ver 8.6).

### 4.8 X1 por frames-por-vídeo, `binary_all` (2 classes)

**Status técnico:** execução comprovada.

**Objetivo inferível pelo código:** mesma grade de 4.6 com todas as vértebras fundidas em uma classe de primeiro plano. INFERÊNCIA A PARTIR DO CÓDIGO.

**Dataset:** como o X1; máscaras `targets/mask/binary_all` (idênticas a `multiclass_all > 0`, verificado).

**Classes:** `n_classes: 2`; reportada `FG`.

**Estratégia de labels / Modelos / Loss / Augmentations / Seeds:** como 4.6, com os YAML de `configs/experiment/x1-binario-all/` (não rastreados no git).

**Métricas:** Dice, IoU, ASSD, HD95 de FG. Nenhuma linha NaN, nenhuma com penalidade.

**Artefatos encontrados:** `logs/x1-frames-per-video-binario-all/results_per_sample.csv` (2752 linhas = 16 × 172); `runs/` (16 runs, 2026-09-23 15:53 a 21:57, cada um com `run.json` e `config.yaml`); `_sweep/state.json` (16 células `done`; `diffrect-fpv02-s42` com `attempts: 2`); `predictions/`.

**Resultados encontrados:** seção 7.7.

**Evidência da execução:** `run.json` de cada run: `status: done`, `final_step: 8000`, `git.sha 95e44672…`, `git.dirty: true`; `diffrect-fpv02-s42` com `resumed_from_step: 2000` (ver 8.8).

### 4.9 X1 por frames-por-vídeo, "binario" e "binario-c2-c6"

**Status técnico:** execução não comprovada (nem configurado no estado atual).

- `logs/x1-frames-per-video-binario/_sweep/console.log` e `logs/x1-frames-per-video-binario-c2-c6/_sweep/console.log`: 23 linhas cada, apenas uma listagem de 16 células `pendente` seguida de `--dry-run: nada foi treinado.`
- Os configs que esses logs citam (`configs/sweep/x1-frames-per-video-binario.yaml`, `configs/sweep/x1-frames-per-video-binario-c2-c6.yaml`) **não existem** no repositório.
- `scripts/build_binary_mask_variant.py` gera, por padrão documentado, a variante `binary_c2_c3_c4_c5_c6`; essa variante **não existe** em `targets/mask/` e não está em `TARGET_VARIANTS`.

NÃO É POSSÍVEL CONFIRMAR EXECUÇÃO.

### 4.10 X2–X13

Descritos apenas em `docs/hipoteses_experimentos.md` (Parte VI). Nenhum código, config ou artefato com esses identificadores foi encontrado. Nem implementados nem configurados.

---

## 5. Variantes X1 identificadas

Nomes exatamente como aparecem no código; a coluna "possível correspondência" é só uma leitura dos nomes.

| Família (`logs/`) | Sweep | Configs de experimento | Prefixo dos dirs de run (real) | `target_variant` | `n_classes` | Classes reportadas | Orçamento | Runs com resultado | Possível correspondência |
|---|---|---|---|---|---|---|---|---|---|
| `x1` | `configs/sweep/x1.yaml` | `configs/experiment/x1/` | `x1-` | `multiclass_c2_c4` | 4 | C2, C4 | `label_fraction` 0.25/0.50/0.75/1.00 por paciente | 16 | "X1"; também "X1 C2/C4" (é a única variante por fração com C2/C4 multiclasse) |
| `x1-binario` | `configs/sweep/x1-binario.yaml` | `configs/experiment/x1-binario/` | — | `binary_c2_c4` | 2 | FG | `label_fraction` | **0** | "X1 binário" (C2+C4) — não executado |
| `x1b` | `configs/sweep/x1b.yaml` | reusa `configs/experiment/x1/` | `x1b-` | `multiclass_c2_c4` | 4 | C2, C4 | fixo 1.0; varia política | 12 | "X1B" |
| `x1-frames-per-video` | `configs/sweep/x1-frames-per-video.yaml` | reusa `configs/experiment/x1/` | `x1-frames-` | `multiclass_c2_c4` | 4 | C2, C4 | 1/2/3/5 frames por vídeo | 16 | "X1 frame por vídeo" (C2/C4) |
| `x1-frames-per-video-multiclass-all` | `configs/sweep/x1-frames-per-video-multiclass-all.yaml` | `configs/experiment/x1-multiclass-all/` | `x1-frames-` (o config atual define `x1-frames-mc-all`) | `multiclass_all` | 8 | C2..C8 | 1/2/3/5 frames por vídeo | 16 | "X1 frame por vídeo multiclasse" |
| `x1-frames-per-video-binario-all` | `configs/sweep/x1-frames-per-video-binario-all.yaml` | `configs/experiment/x1-binario-all/` | `x1-frames-bin-all-` | `binary_all` | 2 | FG | 1/2/3/5 frames por vídeo | 16 | "X1 binário" (todas as vértebras, por frames/vídeo) |
| `x1-frames-per-video-binario` | config ausente | — | — | desconhecido | — | — | 1/2/3/5 (pelo dry-run) | 0 | só dry-run |
| `x1-frames-per-video-binario-c2-c6` | config ausente | — | — | variante inexistente em disco | — | — | 1/2/3/5 (pelo dry-run) | 0 | só dry-run |

Sobre "C2/C4" fora das famílias X1: `configs/experiment/unet/vfss-inca-unet-c2-c4-only.yaml` (`binary_c2_c4`), `vfss-inca-unet-multiclass-c2-c4.yaml` (`multiclass_c2_c4`) e `configs/benchmark/vfss-baseline-c2-c4-only.yaml` pertencem à era das arquiteturas (supervisionado puro, sem regime de rótulo). Há 3 tentativas `logs/_archive/vfss-unet-multiclass-c2-c4/` (2026-09-07) com só `console.log`.

Tentativas anteriores do X1 (arquivadas, nenhuma CSV as referencia):

| Grupo | Datas | Orçamentos | Configuração registrada | Estado |
|---|---|---|---|---|
| X1 "desenho antigo" | 2026-09-08 | lf 0.10/0.20/0.50/1.00 (pelo nome do dir) | `hparams.yaml` sem bloco de dados; rampa antiga (`consistency_rampup_divisor 150`) | 10 runs, passo 8000, com `test/*` |
| X1 "desenho atual, 1ª execução" | 2026-09-11 | lf 0.25/0.50/0.75/1.00 | `hparams.yaml` **idêntico** ao do run correspondente de 2026-09-14 (`diff` vazio, verificado para supervised-lf025 e diffrect-lf025) | 10 runs, passo 8000, com `test/*` |
| X1 fixmatch-lf050 | 2026-09-14 18:50 | 0.50 | igual ao run da CSV | interrompido no passo 5579 |

---

## 6. Seeds

**Onde são configuradas:**

| Seed | Onde | Valor nos runs encontrados |
|---|---|---|
| Global (`pl.seed_everything`) | `Experiment.run(seed=…)`, `src/experiment.py:340`; célula do sweep (`axes.seed`) | 42 (VFSS), 1337 (ACDC); registrada como `seed:` no `hparams.yaml` de todos os runs das CSVs |
| Sampler (`TwoStreamBatchSampler.seed`) | `data.params.train_batch_sampler.params.seed`, sobrescrito pela seed da célula (`src/sweeps/x1.py`, `src/sweeps/acdc.py`) | 42 / 1337 |
| Split | `split_seed` | 42 (todas as famílias X1) |
| Regime de rótulo | `label_seed` | 42 |
| Pool não rotulado | `unlabeled_pool_seed` | 42 |
| Augmentation do ACDC | `ACDCSliceDataset(seed=1337)` padrão (`src/data/acdc_dataset.py:162`) | não registrada no `hparams.yaml` (valor padrão) |
| Augmentations dos wrappers | `torch` global (sem `generator` próprio) | derivada da seed global |
| Gerador do DataLoader | `_LOADER_GENERATOR_SEED = 0` (`src/data/base.py`) | **mudança não commitada**; não é possível saber quais runs a tiveram |

**Runs por seed:**

| Experimento | Seeds | Runs com resultado por célula | Múltiplas seeds? |
|---|---|---|---|
| Benchmark de arquiteturas | não registrada (42 por padrão do código) | 1 | Não |
| ACDC | 1337 | 1 | Não |
| X1 | 42 | 1 | Não |
| X1B | 42 | 1 | Não |
| X1-fpv (4 classes) | 42 | 1 | Não |
| X1-fpv multiclass_all | 42 | 1 | Não |
| X1-fpv binary_all | 42 | 1 | Não |

**Nenhum experimento tem mais de uma seed.** Todos os `configs/sweep/*.yaml` declaram `seed: [42]` ou `seed: [1337]`. As únicas repetições da mesma configuração com a mesma seed são as re-execuções da seção 8.5 (que não são seeds diferentes).

---

## 7. Resultados encontrados

Todas as runs referenciadas por alguma CSV de resultado estão listadas. Colunas `dice/iou/assd/hd95 <classe>`: média por classe sobre os 172 frames de teste (`nanmean`, regra de `aggregate_per_class`). "linhas c/ penalidade": linhas (frame×classe) com HD95 = 362.0387 (estrutura presente só em pred ou só em GT). "test/dice_score (Lightning)": valor que o próprio Lightning gravou no `metrics.csv` do run (média por batch sobre as classes reportadas, calculada pelo código da época do run). ASSD/HD95 em pixels de uma imagem 256×256.

### 7.1 X1 (`logs/x1/results_per_sample.csv`, convenção pós-A2)

| run_id | método | orçamento | dice C2 | dice C4 | iou C2 | iou C4 | assd C2 | assd C4 | hd95 C2 | hd95 C4 | linhas c/ penalidade (hd95=diag) | linhas NaN assd/hd95 | ckpt selecionado | test/dice_score (Lightning) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised-lf025-s42 | supervised | 0.25 | 0.7452 | 0.4987 | 0.6231 | 0.3649 | 20.4800 | 5.5943 | 35.0751 | 18.1948 | 8 | 0 | best-epoch=54-step=2419 | 0.6219 |
| fixmatch-lf025-s42 | fixmatch | 0.25 | 0.7502 | 0.5616 | 0.6267 | 0.4588 | 4.9914 | 7.0765 | 11.8360 | 14.1011 | 2 | 0 | best-epoch=153-step=6775 | 0.6559 |
| meanteacher-lf025-s42 | meanteacher | 0.25 | 0.7551 | 0.5674 | 0.6431 | 0.4576 | 7.0906 | 5.1794 | 15.1460 | 14.6878 | 2 | 0 | best-epoch=98-step=4355 | 0.6612 |
| diffrect-lf025-s42 | diffrect | 0.25 | 0.8257 | 0.7390 | 0.7196 | 0.6402 | 2.6238 | 23.0405 | 8.8846 | 28.0457 | 10 | 0 | best-epoch=142-step=6291 | 0.7824 |
| supervised-lf050-s42 | supervised | 0.5 | 0.8183 | 0.7059 | 0.7099 | 0.6112 | 3.9653 | 24.2206 | 8.5567 | 31.7752 | 11 | 0 | best-epoch=53-step=4859 | 0.7621 |
| fixmatch-lf050-s42 | fixmatch | 0.5 | 0.7996 | 0.6795 | 0.6884 | 0.5721 | 2.1230 | 3.2778 | 6.5973 | 9.5969 | 0 | 0 | best-epoch=83-step=7559 | 0.7396 |
| meanteacher-lf050-s42 | meanteacher | 0.5 | 0.8556 | 0.7930 | 0.7697 | 0.7118 | 5.7901 | 6.7575 | 9.8445 | 10.7281 | 4 | 0 | best-epoch=35-step=3239 | 0.8243 |
| diffrect-lf050-s42 | diffrect | 0.5 | 0.8694 | 0.7986 | 0.7761 | 0.6999 | 1.9056 | 1.7070 | 7.8180 | 5.9932 | 0 | 0 | best-epoch=77-step=7019 | 0.8340 |
| supervised-lf075-s42 | supervised | 0.75 | 0.7777 | 0.6895 | 0.6709 | 0.5994 | 2.2997 | 43.7282 | 6.2219 | 46.2389 | 20 | 0 | best-epoch=39-step=5159 | 0.7336 |
| fixmatch-lf075-s42 | fixmatch | 0.75 | 0.7629 | 0.7474 | 0.6783 | 0.6657 | 19.2491 | 8.6712 | 22.9038 | 14.2248 | 10 | 0 | best-epoch=43-step=5675 | 0.7552 |
| meanteacher-lf075-s42 | meanteacher | 0.75 | 0.7872 | 0.7676 | 0.6899 | 0.6955 | 4.5740 | 18.4087 | 9.6205 | 21.8372 | 8 | 0 | best-epoch=27-step=3611 | 0.7774 |
| diffrect-lf075-s42 | diffrect | 0.75 | 0.8456 | 0.7779 | 0.7458 | 0.6784 | 5.5861 | 8.2337 | 7.7480 | 11.7769 | 5 | 0 | best-epoch=51-step=6707 | 0.8117 |
| supervised-lf100-s42 | supervised | 1.0 | 0.8476 | 0.7827 | 0.7615 | 0.7111 | 1.4995 | 24.8281 | 4.4209 | 27.7045 | 10 | 0 | best-epoch=17-step=3095 | 0.8152 |
| fixmatch-lf100-s42 | fixmatch | 1.0 | 0.8731 | 0.7825 | 0.7856 | 0.6994 | 1.2929 | 3.6710 | 3.7703 | 11.9104 | 0 | 0 | best-epoch=20-step=3611 | 0.8278 |
| meanteacher-lf100-s42 | meanteacher | 1.0 | 0.8542 | 0.8466 | 0.7709 | 0.7717 | 3.4803 | 3.4807 | 5.9204 | 6.2794 | 2 | 0 | best-epoch=23-step=4127 | 0.8504 |
| diffrect-lf100-s42 | diffrect | 1.0 | 0.8847 | 0.7775 | 0.7996 | 0.6962 | 1.1354 | 26.4733 | 3.0575 | 28.4206 | 12 | 0 | best-epoch=38-step=6707 | 0.8311 |

### 7.2 X1 — mesma grade, ASSD/HD95 da CSV original (`results_per_sample_pre_a2_fix.csv`)

Dice e IoU são idênticos à tabela 7.1 (verificado: 0 linhas com Dice diferente). A única diferença entre os dois arquivos são as 104 linhas NaN → 362.0387; nenhum outro valor mudou (verificado).

| run_id | assd C2 (pré-A2, nanmean) | assd C4 (pré-A2, nanmean) | hd95 C2 (pré-A2, nanmean) | hd95 C4 (pré-A2, nanmean) | linhas NaN |
|---|---|---|---|---|---|
| supervised-lf025-s42 | 3.8186 | 5.5943 | 19.1257 | 18.1948 | 8 |
| fixmatch-lf025-s42 | 2.9034 | 5.0007 | 9.7880 | 12.0664 | 2 |
| meanteacher-lf025-s42 | 2.9148 | 5.1794 | 11.0649 | 14.6878 | 2 |
| diffrect-lf025-s42 | 2.6238 | 2.1147 | 8.8846 | 7.4289 | 10 |
| supervised-lf050-s42 | 1.8713 | 3.3676 | 6.4895 | 11.3886 | 11 |
| fixmatch-lf050-s42 | 2.1230 | 3.2778 | 6.5973 | 9.5969 | 0 |
| meanteacher-lf050-s42 | 1.5989 | 2.5777 | 5.7010 | 6.5951 | 4 |
| diffrect-lf050-s42 | 1.9056 | 1.7070 | 7.8180 | 5.9932 | 0 |
| supervised-lf075-s42 | 2.2997 | 1.8452 | 6.2219 | 4.6863 | 20 |
| fixmatch-lf075-s42 | 2.5277 | 4.5140 | 6.3607 | 10.1329 | 10 |
| meanteacher-lf075-s42 | 2.4835 | 3.8305 | 7.5595 | 7.4045 | 8 |
| diffrect-lf075-s42 | 1.3925 | 1.9531 | 3.5798 | 5.5593 | 5 |
| supervised-lf100-s42 | 1.4995 | 4.0126 | 4.4209 | 7.0666 | 10 |
| fixmatch-lf100-s42 | 1.2929 | 3.6710 | 3.7703 | 11.9104 | 0 |
| meanteacher-lf100-s42 | 1.3834 | 1.3839 | 3.8378 | 4.1989 | 2 |
| diffrect-lf100-s42 | 1.1354 | 1.3059 | 3.0575 | 3.3993 | 12 |

### 7.3 X1 por frames-por-vídeo, 4 classes (`logs/x1-frames-per-video/results_per_sample.csv`)

| run_id | método | orçamento | dice C2 | dice C4 | iou C2 | iou C4 | assd C2 | assd C4 | hd95 C2 | hd95 C4 | linhas c/ penalidade (hd95=diag) | linhas NaN assd/hd95 | ckpt selecionado | test/dice_score (Lightning) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 0.7616 | 0.7116 | 0.6695 | 0.6194 | 9.2727 | 13.5163 | 15.7747 | 18.4979 | 8 | 0 | best-epoch=143-step=5903 | 0.7366 |
| fixmatch-fpv01-s42 | fixmatch | 1 | 0.7630 | 0.7368 | 0.6723 | 0.6477 | 17.2056 | 25.1919 | 21.9772 | 29.2227 | 17 | 0 | best-epoch=143-step=5903 | 0.7499 |
| meanteacher-fpv01-s42 | meanteacher | 1 | 0.7214 | 0.7203 | 0.6296 | 0.6332 | 18.4583 | 11.0100 | 22.0787 | 15.9187 | 10 | 0 | best-epoch=119-step=4919 | 0.7209 |
| diffrect-fpv01-s42 | diffrect | 1 | 0.8333 | 0.6732 | 0.7301 | 0.5609 | 5.7002 | 9.4460 | 8.0082 | 15.9170 | 5 | 0 | best-epoch=191-step=7871 | 0.7533 |
| supervised-fpv02-s42 | supervised | 2 | 0.8307 | 0.7713 | 0.7293 | 0.6907 | 7.8744 | 5.5183 | 11.6071 | 9.1912 | 3 | 0 | best-epoch=41-step=3443 | 0.8010 |
| fixmatch-fpv02-s42 | fixmatch | 2 | 0.8452 | 0.8001 | 0.7544 | 0.7281 | 5.9471 | 19.4169 | 10.2563 | 21.7727 | 10 | 0 | best-epoch=59-step=4919 | 0.8227 |
| meanteacher-fpv02-s42 | meanteacher | 2 | 0.8279 | 0.7601 | 0.7421 | 0.6930 | 5.7865 | 39.5445 | 8.3611 | 41.7430 | 20 | 0 | best-epoch=29-step=2459 | 0.7940 |
| diffrect-fpv02-s42 | diffrect | 2 | 0.8773 | 0.8065 | 0.7948 | 0.7306 | 5.3031 | 13.1112 | 7.3453 | 17.6833 | 3 | 0 | best-epoch=71-step=5903 | 0.8419 |
| supervised-fpv03-s42 | supervised | 3 | 0.8106 | 0.7645 | 0.7201 | 0.6853 | 6.0579 | 26.4268 | 8.9027 | 29.4909 | 13 | 0 | best-epoch=35-step=4427 | 0.7875 |
| fixmatch-fpv03-s42 | fixmatch | 3 | 0.8505 | 0.7743 | 0.7553 | 0.6809 | 5.5982 | 8.4076 | 8.1677 | 11.8104 | 5 | 0 | best-epoch=59-step=7379 | 0.8124 |
| meanteacher-fpv03-s42 | meanteacher | 3 | 0.8569 | 0.7796 | 0.7623 | 0.6909 | 1.4177 | 4.0676 | 4.0050 | 7.5611 | 1 | 0 | best-epoch=39-step=4919 | 0.8182 |
| diffrect-fpv03-s42 | diffrect | 3 | 0.8651 | 0.7573 | 0.7748 | 0.6746 | 5.4725 | 37.0485 | 8.0070 | 38.8056 | 19 | 0 | best-epoch=63-step=7871 | 0.8112 |
| supervised-fpv05-s42 | supervised | 5 | 0.8433 | 0.7715 | 0.7554 | 0.6983 | 3.6742 | 21.5716 | 6.5739 | 23.9794 | 10 | 0 | best-epoch=32-step=5675 | 0.8074 |
| fixmatch-fpv05-s42 | fixmatch | 5 | 0.8303 | 0.7644 | 0.7428 | 0.6880 | 7.9087 | 17.9684 | 10.6923 | 20.8658 | 10 | 0 | best-epoch=41-step=7223 | 0.7973 |
| meanteacher-fpv05-s42 | meanteacher | 5 | 0.8181 | 0.8527 | 0.7430 | 0.7795 | 8.2447 | 2.2689 | 11.1947 | 5.6049 | 3 | 0 | best-epoch=26-step=4643 | 0.8354 |
| diffrect-fpv05-s42 | diffrect | 5 | 0.8322 | 0.6739 | 0.7271 | 0.5648 | 5.9223 | 9.4402 | 10.0033 | 19.8703 | 4 | 0 | best-epoch=41-step=7223 | 0.7531 |

### 7.4 X1 por frames-por-vídeo, `multiclass_all` (`logs/x1-frames-per-video-multiclass-all/results_per_sample.csv`)

"—" = NaN em todas as 172 linhas (classe ausente de pred e GT em todo o teste).

**Dice por classe**

| run_id | método | fpv | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 0.6843 | 0.6503 | 0.6308 | 0.5388 | 0.4045 | 0.0000 | — |
| fixmatch-fpv01-s42 | fixmatch | 1 | 0.7673 | 0.7142 | 0.6517 | 0.5310 | 0.3024 | 0.0496 | — |
| meanteacher-fpv01-s42 | meanteacher | 1 | 0.7904 | 0.8002 | 0.7331 | 0.6366 | 0.4555 | 0.0454 | — |
| diffrect-fpv01-s42 | diffrect | 1 | 0.7566 | 0.7431 | 0.7241 | 0.6864 | 0.4272 | 0.1796 | — |
| supervised-fpv02-s42 | supervised | 2 | 0.8235 | 0.7591 | 0.7218 | 0.7044 | 0.5414 | 0.0597 | — |
| fixmatch-fpv02-s42 | fixmatch | 2 | 0.7626 | 0.7442 | 0.7082 | 0.6066 | 0.4671 | 0.0000 | — |
| meanteacher-fpv02-s42 | meanteacher | 2 | 0.7595 | 0.7965 | 0.7742 | 0.7078 | 0.5709 | 0.0603 | — |
| diffrect-fpv02-s42 | diffrect | 2 | 0.8019 | 0.7572 | 0.6990 | 0.6357 | 0.4522 | 0.2055 | — |
| supervised-fpv03-s42 | supervised | 3 | 0.8612 | 0.8718 | 0.7755 | 0.6833 | 0.5125 | 0.1447 | — |
| fixmatch-fpv03-s42 | fixmatch | 3 | 0.8074 | 0.7731 | 0.7492 | 0.6836 | 0.5406 | 0.0120 | — |
| meanteacher-fpv03-s42 | meanteacher | 3 | 0.7638 | 0.8035 | 0.7742 | 0.6999 | 0.5875 | 0.3312 | — |
| diffrect-fpv03-s42 | diffrect | 3 | 0.8112 | 0.7236 | 0.6562 | 0.5601 | 0.4320 | 0.2546 | — |
| supervised-fpv05-s42 | supervised | 5 | 0.8335 | 0.7893 | 0.7566 | 0.6351 | 0.4857 | 0.0699 | — |
| fixmatch-fpv05-s42 | fixmatch | 5 | 0.7691 | 0.7581 | 0.7438 | 0.6774 | 0.5067 | 0.1520 | — |
| meanteacher-fpv05-s42 | meanteacher | 5 | 0.7739 | 0.7727 | 0.7411 | 0.6435 | 0.5251 | 0.1183 | — |
| diffrect-fpv05-s42 | diffrect | 5 | 0.7852 | 0.6957 | 0.6480 | 0.5832 | 0.4117 | 0.2219 | — |

**IoU por classe**

| run_id | método | fpv | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 0.5721 | 0.5525 | 0.5282 | 0.4463 | 0.3258 | 0.0000 | — |
| fixmatch-fpv01-s42 | fixmatch | 1 | 0.6527 | 0.6074 | 0.5339 | 0.4138 | 0.2387 | 0.0342 | — |
| meanteacher-fpv01-s42 | meanteacher | 1 | 0.7081 | 0.7314 | 0.6616 | 0.5588 | 0.3924 | 0.0316 | — |
| diffrect-fpv01-s42 | diffrect | 1 | 0.6501 | 0.6530 | 0.6239 | 0.5913 | 0.3470 | 0.1365 | — |
| supervised-fpv02-s42 | supervised | 2 | 0.7257 | 0.6894 | 0.6555 | 0.6388 | 0.4840 | 0.0439 | — |
| fixmatch-fpv02-s42 | fixmatch | 2 | 0.6677 | 0.6569 | 0.6143 | 0.5014 | 0.3857 | 0.0000 | — |
| meanteacher-fpv02-s42 | meanteacher | 2 | 0.6734 | 0.7301 | 0.7077 | 0.6342 | 0.5044 | 0.0388 | — |
| diffrect-fpv02-s42 | diffrect | 2 | 0.6877 | 0.6499 | 0.5850 | 0.5261 | 0.3591 | 0.1460 | — |
| supervised-fpv03-s42 | supervised | 3 | 0.7695 | 0.7971 | 0.7010 | 0.6046 | 0.4526 | 0.1042 | — |
| fixmatch-fpv03-s42 | fixmatch | 3 | 0.7071 | 0.7031 | 0.6749 | 0.6107 | 0.4778 | 0.0078 | — |
| meanteacher-fpv03-s42 | meanteacher | 3 | 0.6809 | 0.7406 | 0.7117 | 0.6350 | 0.5287 | 0.2787 | — |
| diffrect-fpv03-s42 | diffrect | 3 | 0.6954 | 0.5951 | 0.5305 | 0.4411 | 0.3307 | 0.1854 | — |
| supervised-fpv05-s42 | supervised | 5 | 0.7414 | 0.7157 | 0.6752 | 0.5515 | 0.4297 | 0.0550 | — |
| fixmatch-fpv05-s42 | fixmatch | 5 | 0.6819 | 0.6766 | 0.6622 | 0.5913 | 0.4383 | 0.1182 | — |
| meanteacher-fpv05-s42 | meanteacher | 5 | 0.6925 | 0.7045 | 0.6631 | 0.5649 | 0.4542 | 0.0875 | — |
| diffrect-fpv05-s42 | diffrect | 5 | 0.6644 | 0.5702 | 0.5209 | 0.4469 | 0.3123 | 0.1604 | — |

**ASSD por classe** (C8 = 0.0 pela convenção "ausente dos dois lados")

| run_id | método | fpv | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 25.6653 | 8.4461 | 8.7415 | 57.6378 | 102.5153 | 82.0902 | 0.0000 |
| fixmatch-fpv01-s42 | fixmatch | 1 | 6.6017 | 2.6552 | 3.5457 | 34.7459 | 107.3312 | 82.2046 | 0.0000 |
| meanteacher-fpv01-s42 | meanteacher | 1 | 7.2734 | 2.6712 | 3.3333 | 58.6585 | 91.4561 | 79.0384 | 0.0000 |
| diffrect-fpv01-s42 | diffrect | 1 | 6.4868 | 6.8174 | 11.1764 | 35.4598 | 106.7780 | 114.5515 | 0.0000 |
| supervised-fpv02-s42 | supervised | 2 | 1.8300 | 2.5235 | 7.3436 | 43.6978 | 80.7822 | 73.8250 | 0.0000 |
| fixmatch-fpv02-s42 | fixmatch | 2 | 20.8787 | 3.4059 | 4.0768 | 53.3177 | 91.7993 | 82.0902 | 0.0000 |
| meanteacher-fpv02-s42 | meanteacher | 2 | 10.5045 | 2.9789 | 5.2050 | 52.0629 | 70.7862 | 84.5748 | 0.0000 |
| diffrect-fpv02-s42 | diffrect | 2 | 2.0281 | 6.3174 | 10.9773 | 40.0215 | 98.4228 | 85.0373 | 0.0000 |
| supervised-fpv03-s42 | supervised | 3 | 1.4171 | 1.4174 | 12.6461 | 52.2318 | 93.6747 | 66.1529 | 0.0000 |
| fixmatch-fpv03-s42 | fixmatch | 3 | 3.9937 | 6.4786 | 2.8681 | 52.2520 | 91.4524 | 80.1843 | 0.0000 |
| meanteacher-fpv03-s42 | meanteacher | 3 | 7.3907 | 7.0112 | 7.1466 | 57.9981 | 78.6527 | 76.3814 | 0.0000 |
| diffrect-fpv03-s42 | diffrect | 3 | 4.1077 | 2.2813 | 3.1648 | 55.1155 | 88.0251 | 95.5725 | 0.0000 |
| supervised-fpv05-s42 | supervised | 5 | 5.8410 | 3.6247 | 3.2336 | 50.4913 | 100.1762 | 103.2785 | 0.0000 |
| fixmatch-fpv05-s42 | fixmatch | 5 | 14.6037 | 4.0973 | 4.0926 | 52.4057 | 108.3950 | 70.0763 | 0.0000 |
| meanteacher-fpv05-s42 | meanteacher | 5 | 25.6439 | 11.5587 | 11.7943 | 60.6361 | 85.2186 | 99.3693 | 0.0000 |
| diffrect-fpv05-s42 | diffrect | 5 | 6.2428 | 12.9216 | 14.6147 | 19.6272 | 90.2057 | 106.1280 | 0.0000 |

**HD95 por classe**

| run_id | método | fpv | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 29.8155 | 13.2817 | 14.6776 | 61.8314 | 104.5651 | 82.0902 | 0.0000 |
| fixmatch-fpv01-s42 | fixmatch | 1 | 10.8196 | 6.6030 | 11.5604 | 41.7844 | 110.6359 | 82.4369 | 0.0000 |
| meanteacher-fpv01-s42 | meanteacher | 1 | 10.5946 | 5.8521 | 7.5107 | 61.8435 | 93.1243 | 79.3648 | 0.0000 |
| diffrect-fpv01-s42 | diffrect | 1 | 10.0902 | 10.2194 | 15.6300 | 39.8289 | 109.5721 | 115.6205 | 0.0000 |
| supervised-fpv02-s42 | supervised | 2 | 4.9184 | 5.4517 | 10.5034 | 46.4514 | 82.0548 | 74.1157 | 0.0000 |
| fixmatch-fpv02-s42 | fixmatch | 2 | 23.7172 | 7.0131 | 9.5695 | 59.5914 | 94.0177 | 82.0902 | 0.0000 |
| meanteacher-fpv02-s42 | meanteacher | 2 | 14.8638 | 5.9688 | 9.2571 | 55.2395 | 75.5642 | 85.1179 | 0.0000 |
| diffrect-fpv02-s42 | diffrect | 2 | 5.2431 | 9.6562 | 16.0045 | 44.7514 | 102.0886 | 86.2081 | 0.0000 |
| supervised-fpv03-s42 | supervised | 3 | 4.6435 | 4.5199 | 15.4983 | 55.6685 | 95.7263 | 67.0641 | 0.0000 |
| fixmatch-fpv03-s42 | fixmatch | 3 | 7.3938 | 9.3183 | 6.6826 | 55.6742 | 93.4076 | 80.3234 | 0.0000 |
| meanteacher-fpv03-s42 | meanteacher | 3 | 10.8535 | 10.3828 | 10.5433 | 60.1288 | 79.8131 | 77.2061 | 0.0000 |
| diffrect-fpv03-s42 | diffrect | 3 | 7.8596 | 5.6011 | 7.9138 | 58.8380 | 90.4361 | 96.6797 | 0.0000 |
| supervised-fpv05-s42 | supervised | 5 | 9.2103 | 11.8760 | 10.4756 | 54.6244 | 102.5222 | 103.5053 | 0.0000 |
| fixmatch-fpv05-s42 | fixmatch | 5 | 19.0821 | 9.0581 | 8.6332 | 56.9041 | 110.7514 | 70.9303 | 0.0000 |
| meanteacher-fpv05-s42 | meanteacher | 5 | 28.3501 | 14.0786 | 15.4647 | 63.9401 | 87.0959 | 100.0181 | 0.0000 |
| diffrect-fpv05-s42 | diffrect | 5 | 9.7121 | 16.4335 | 21.2267 | 23.5714 | 92.6410 | 107.1563 | 0.0000 |

**Contagens e checkpoint por run** ("n dice≠NaN": frames em que a classe está em pred ou GT)

| run_id | n dice≠NaN C2 | n dice≠NaN C3 | n dice≠NaN C4 | n dice≠NaN C5 | n dice≠NaN C6 | n dice≠NaN C7 | n dice≠NaN C8 | linhas hd95=diag | ckpt | test/dice_score (Lightning) |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | 172 | 172 | 172 | 172 | 135 | 39 | 0 | 128 | best-epoch=143-step=5903 | 0.5223 |
| fixmatch-fpv01-s42 | 172 | 172 | 172 | 163 | 130 | 43 | 0 | 106 | best-epoch=131-step=5411 | 0.5353 |
| meanteacher-fpv01-s42 | 172 | 172 | 172 | 171 | 122 | 44 | 0 | 109 | best-epoch=83-step=3443 | 0.6041 |
| diffrect-fpv01-s42 | 172 | 172 | 172 | 160 | 140 | 86 | 0 | 128 | best-epoch=191-step=7871 | 0.5810 |
| supervised-fpv02-s42 | 172 | 172 | 172 | 165 | 123 | 40 | 0 | 95 | best-epoch=83-step=6887 | 0.6390 |
| fixmatch-fpv02-s42 | 172 | 172 | 172 | 172 | 131 | 39 | 0 | 115 | best-epoch=65-step=5411 | 0.5889 |
| meanteacher-fpv02-s42 | 172 | 172 | 172 | 172 | 125 | 51 | 0 | 101 | best-epoch=59-step=4919 | 0.6369 |
| diffrect-fpv02-s42 | 172 | 172 | 172 | 162 | 140 | 72 | 0 | 110 | best-epoch=71-step=5903 | 0.5911 |
| supervised-fpv03-s42 | 172 | 172 | 172 | 172 | 138 | 56 | 0 | 104 | best-epoch=35-step=4427 | 0.6666 |
| fixmatch-fpv03-s42 | 172 | 172 | 172 | 172 | 133 | 41 | 0 | 108 | best-epoch=59-step=7379 | 0.6360 |
| meanteacher-fpv03-s42 | 172 | 172 | 172 | 172 | 131 | 72 | 0 | 106 | best-epoch=31-step=3935 | 0.6648 |
| diffrect-fpv03-s42 | 172 | 172 | 172 | 172 | 137 | 84 | 0 | 112 | best-epoch=51-step=6395 | 0.5751 |
| supervised-fpv05-s42 | 172 | 172 | 172 | 171 | 143 | 56 | 0 | 121 | best-epoch=26-step=4643 | 0.6173 |
| fixmatch-fpv05-s42 | 172 | 172 | 172 | 172 | 146 | 54 | 0 | 113 | best-epoch=26-step=4643 | 0.6242 |
| meanteacher-fpv05-s42 | 172 | 172 | 172 | 172 | 132 | 64 | 0 | 134 | best-epoch=26-step=4643 | 0.6208 |
| diffrect-fpv05-s42 | 172 | 172 | 172 | 154 | 137 | 88 | 0 | 112 | best-epoch=38-step=6707 | 0.5588 |

### 7.5 X1B (`logs/x1b/results_per_sample.csv`, convenção pré-A2: ASSD/HD95 NaN descartados por `nanmean`)

| run_id | método | orçamento | dice C2 | dice C4 | iou C2 | iou C4 | assd C2 | assd C4 | hd95 C2 | hd95 C4 | linhas c/ penalidade (hd95=diag) | linhas NaN assd/hd95 | ckpt selecionado | test/dice_score (Lightning) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised-global-s42 | supervised | global | 0.8031 | 0.7671 | 0.7182 | 0.6942 | 2.0645 | 3.0254 | 4.9727 | 6.0820 | 0 | 7 | best-epoch=35-step=6191 | 0.7851 |
| fixmatch-global-s42 | fixmatch | global | 0.8338 | 0.7086 | 0.7299 | 0.6081 | 1.6238 | 4.2548 | 4.4797 | 9.8353 | 0 | 3 | best-epoch=38-step=6707 | 0.7712 |
| meanteacher-global-s42 | meanteacher | global | 0.8007 | 0.6633 | 0.6945 | 0.5638 | 2.2240 | 3.3376 | 8.7829 | 7.5491 | 0 | 9 | best-epoch=26-step=4643 | 0.7320 |
| supervised-same_video-s42 | supervised | same_video | 0.8680 | 0.8474 | 0.7820 | 0.7675 | 1.2864 | 1.3897 | 3.8825 | 3.7687 | 0 | 3 | best-epoch=32-step=5675 | 0.8577 |
| fixmatch-same_video-s42 | fixmatch | same_video | 0.8678 | 0.7937 | 0.7769 | 0.7050 | 1.3382 | 2.1644 | 3.7233 | 5.9622 | 0 | 0 | best-epoch=38-step=6707 | 0.8307 |
| meanteacher-same_video-s42 | meanteacher | same_video | 0.8785 | 0.8406 | 0.7923 | 0.7674 | 1.5691 | 1.3432 | 6.9227 | 3.4266 | 0 | 2 | best-epoch=17-step=3095 | 0.8596 |
| supervised-W5-s42 | supervised | W=5 | 0.8829 | 0.8185 | 0.7989 | 0.7455 | 1.3181 | 1.2291 | 4.3409 | 3.8338 | 0 | 8 | best-epoch=20-step=3611 | 0.8507 |
| fixmatch-W5-s42 | fixmatch | W=5 | 0.8612 | 0.7726 | 0.7724 | 0.6968 | 1.3609 | 2.4702 | 3.7112 | 5.7054 | 0 | 7 | best-epoch=38-step=6707 | 0.8169 |
| meanteacher-W5-s42 | meanteacher | W=5 | 0.8372 | 0.8101 | 0.7457 | 0.7451 | 1.7852 | 2.0744 | 6.6790 | 4.6942 | 0 | 3 | best-epoch=29-step=5159 | 0.8237 |
| supervised-W10-s42 | supervised | W=10 | 0.8312 | 0.7605 | 0.7380 | 0.6856 | 2.1473 | 4.2700 | 7.2305 | 9.7146 | 0 | 8 | best-epoch=14-step=2579 | 0.7958 |
| fixmatch-W10-s42 | fixmatch | W=10 | 0.8468 | 0.7968 | 0.7625 | 0.7353 | 1.7381 | 2.8149 | 5.0030 | 5.7224 | 0 | 3 | best-epoch=23-step=4127 | 0.8218 |
| meanteacher-W10-s42 | meanteacher | W=10 | 0.8468 | 0.7975 | 0.7637 | 0.7333 | 1.5365 | 4.0530 | 4.3684 | 6.5846 | 0 | 2 | best-epoch=23-step=4127 | 0.8222 |

### 7.6 ACDC (`logs/acdc/results_per_volume.csv`)

Macro = média das médias por classe (RV, Myo, LV) sobre 20 volumes de validação ou 40 de teste. Dice/Jaccard em [0, 1]; HD95/ASD em voxels. "dir" é o nome real do diretório do run.

| run_id | método | backbone | ratio | val dice | val jaccard | val hd95 | val asd | test dice | test jaccard | test hd95 | test asd | max val/dice_volume (metrics.csv) | ckpt | dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised-repo-10pct-s1337 | supervised | repo | 10% | 0.8678 | 0.7802 | 3.7949 | 0.7607 | 0.8712 | 0.7802 | 4.4625 | 0.8145 | 0.8678 | best-epoch=71-step=6119 | acdc-supervised-10pct-s1337 |
| fixmatch-repo-10pct-s1337 | fixmatch | repo | 10% | 0.8848 | 0.8007 | 2.6331 | 0.6271 | 0.8801 | 0.7945 | 2.1397 | 0.5283 | 0.8848 | best-epoch=305-step=26009 | acdc-fixmatch-repo-10pct-s1337 |
| meanteacher-repo-10pct-s1337 | meanteacher | repo | 10% | 0.8595 | 0.7717 | 3.7972 | 0.7887 | 0.8667 | 0.7770 | 4.5352 | 0.9051 | 0.8595 | best-epoch=107-step=9179 | acdc-meanteacher-repo-10pct-s1337 |
| diffrect-repo-10pct-s1337 | diffrect | repo | 10% | 0.8857 | 0.8054 | 2.3305 | 0.5077 | 0.8831 | 0.8000 | 3.4054 | 0.6529 | 0.8857 | best-epoch=173-step=14789 | acdc-diffrect-repo-10pct-s1337 |
| supervised-ssl4mis-10pct-s1337 | supervised | ssl4mis | 10% | 0.8521 | 0.7642 | 10.9781 | 2.3533 | 0.8702 | 0.7800 | 4.7890 | 0.9970 | 0.8521 | best-epoch=209-step=17849 | acdc-supervised-ssl4mis-10pct-s1337 |
| fixmatch-ssl4mis-10pct-s1337 | fixmatch | ssl4mis | 10% | 0.8654 | 0.7750 | 9.8849 | 1.8886 | 0.8708 | 0.7830 | 3.8716 | 0.8952 | 0.8654 | best-epoch=209-step=17849 | acdc-fixmatch-ssl4mis-10pct-s1337 |
| meanteacher-ssl4mis-10pct-s1337 | meanteacher | ssl4mis | 10% | 0.8443 | 0.7557 | 10.1568 | 2.1952 | 0.8733 | 0.7851 | 5.0423 | 1.0691 | 0.8443 | best-epoch=299-step=25499 | acdc-meanteacher-ssl4mis-10pct-s1337 |
| diffrect-ssl4mis-10pct-s1337 | diffrect | ssl4mis | 10% | 0.8954 | 0.8169 | 4.7002 | 1.0282 | 0.8831 | 0.7990 | 4.2384 | 0.9113 | 0.8954 | best-epoch=347-step=29579 | acdc-diffrect-ssl4mis-10pct-s1337 |
| supervised-repo-5pct-s1337 | supervised | repo | 5% | 0.8224 | 0.7237 | 11.7147 | 2.6845 | 0.8383 | 0.7372 | 11.2075 | 2.1152 | 0.8224 | best-epoch=120-step=5444 | acdc-supervised-repo-5pct-s1337 |
| fixmatch-repo-5pct-s1337 | fixmatch | repo | 5% | 0.8596 | 0.7659 | 4.3144 | 0.8109 | 0.8611 | 0.7691 | 3.4467 | 0.7788 | 0.8596 | best-epoch=604-step=27224 | acdc-fixmatch-repo-5pct-s1337 |
| meanteacher-repo-5pct-s1337 | meanteacher | repo | 5% | 0.8266 | 0.7287 | 9.4382 | 2.1208 | 0.8417 | 0.7419 | 12.2121 | 2.3283 | 0.8266 | best-epoch=186-step=8414 | acdc-meanteacher-repo-5pct-s1337 |
| diffrect-repo-5pct-s1337 | diffrect | repo | 5% | 0.8503 | 0.7613 | 2.1165 | 0.5556 | 0.8603 | 0.7656 | 5.0414 | 0.9453 | 0.8503 | best-epoch=351-step=15839 | acdc-diffrect-repo-5pct-s1337 |
| supervised-ssl4mis-5pct-s1337 | supervised | ssl4mis | 5% | 0.8433 | 0.7458 | 10.8619 | 1.7837 | 0.8616 | 0.7667 | 5.6219 | 1.3008 | 0.8433 | best-epoch=362-step=16334 | acdc-supervised-ssl4mis-5pct-s1337 |
| fixmatch-ssl4mis-5pct-s1337 | fixmatch | ssl4mis | 5% | 0.8311 | 0.7294 | 9.5331 | 2.0260 | 0.8469 | 0.7469 | 8.1662 | 1.5154 | 0.8311 | best-epoch=571-step=25739 | acdc-fixmatch-ssl4mis-5pct-s1337 |
| meanteacher-ssl4mis-5pct-s1337 | meanteacher | ssl4mis | 5% | 0.8363 | 0.7367 | 9.0821 | 1.7197 | 0.8494 | 0.7502 | 8.9044 | 1.7845 | 0.8363 | best-epoch=428-step=19304 | acdc-meanteacher-ssl4mis-5pct-s1337 |
| diffrect-ssl4mis-5pct-s1337 | diffrect | ssl4mis | 5% | 0.8672 | 0.7801 | 4.9782 | 0.9316 | 0.8626 | 0.7689 | 8.3886 | 1.6262 | 0.8672 | best-epoch=450-step=20294 | acdc-diffrect-ssl4mis-5pct-s1337 |

**Dice por classe**

| run_id | val dice RV | val dice Myo | val dice LV | test dice RV | test dice Myo | test dice LV |
|---|---|---|---|---|---|---|
| supervised-repo-10pct-s1337 | 0.8205 | 0.8663 | 0.9165 | 0.8770 | 0.8380 | 0.8986 |
| supervised-ssl4mis-10pct-s1337 | 0.7926 | 0.8700 | 0.8938 | 0.8590 | 0.8544 | 0.8972 |
| fixmatch-ssl4mis-10pct-s1337 | 0.8302 | 0.8736 | 0.8922 | 0.8553 | 0.8606 | 0.8965 |
| meanteacher-ssl4mis-10pct-s1337 | 0.7785 | 0.8678 | 0.8864 | 0.8642 | 0.8579 | 0.8977 |
| diffrect-ssl4mis-10pct-s1337 | 0.8854 | 0.8810 | 0.9196 | 0.8917 | 0.8576 | 0.8999 |
| supervised-ssl4mis-5pct-s1337 | 0.7943 | 0.8447 | 0.8909 | 0.8534 | 0.8381 | 0.8931 |
| fixmatch-ssl4mis-5pct-s1337 | 0.7850 | 0.8420 | 0.8663 | 0.8453 | 0.8152 | 0.8803 |
| meanteacher-ssl4mis-5pct-s1337 | 0.7936 | 0.8439 | 0.8716 | 0.8446 | 0.8279 | 0.8758 |
| diffrect-ssl4mis-5pct-s1337 | 0.8505 | 0.8572 | 0.8939 | 0.8689 | 0.8318 | 0.8872 |
| fixmatch-repo-10pct-s1337 | 0.8553 | 0.8778 | 0.9213 | 0.8658 | 0.8576 | 0.9169 |
| meanteacher-repo-10pct-s1337 | 0.7886 | 0.8761 | 0.9139 | 0.8469 | 0.8511 | 0.9021 |
| diffrect-repo-10pct-s1337 | 0.8621 | 0.8767 | 0.9183 | 0.8650 | 0.8655 | 0.9187 |
| supervised-repo-5pct-s1337 | 0.7498 | 0.8292 | 0.8882 | 0.8072 | 0.8150 | 0.8928 |
| fixmatch-repo-5pct-s1337 | 0.8126 | 0.8591 | 0.9070 | 0.8338 | 0.8397 | 0.9098 |
| meanteacher-repo-5pct-s1337 | 0.7403 | 0.8482 | 0.8914 | 0.8025 | 0.8276 | 0.8951 |
| diffrect-repo-5pct-s1337 | 0.8094 | 0.8306 | 0.9107 | 0.8298 | 0.8413 | 0.9097 |

A saída salva do notebook `experiment_acdc_ssl_reproduction.ipynb` (célula 15) reproduz os mesmos valores de validação (×100) e compara com uma tabela publicada digitada no próprio notebook (DiffRect 10% 90.18, 5% 86.95). A célula 16 imprime: `10% [ssl4mis] nosso 89.54 publicado 90.18 delta -0.64 PASSA`, `5% [ssl4mis] nosso 86.72 publicado 86.95 delta -0.23 PASSA`, `PORT ACEITO -- a grade do X1 pode rodar.` (critério declarado na célula: |Δ Dice| ≤ 2.0 pontos, só DiffRect, só validação).

### 7.7 X1 por frames-por-vídeo, `binary_all` (`logs/x1-frames-per-video-binario-all/results_per_sample.csv`)

| run_id | método | orçamento | dice FG | iou FG | assd FG | hd95 FG | linhas c/ penalidade (hd95=diag) | linhas NaN assd/hd95 | ckpt selecionado | test/dice_score (Lightning) |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised-fpv01-s42 | supervised | 1 | 0.8816 | 0.7957 | 1.3147 | 5.4378 | 0 | 0 | best-epoch=83-step=3443 | 0.8816 |
| fixmatch-fpv01-s42 | fixmatch | 1 | 0.8683 | 0.7751 | 1.7936 | 10.7325 | 0 | 0 | best-epoch=191-step=7871 | 0.8683 |
| meanteacher-fpv01-s42 | meanteacher | 1 | 0.8714 | 0.7797 | 1.2887 | 6.0656 | 0 | 0 | best-epoch=71-step=2951 | 0.8714 |
| diffrect-fpv01-s42 | diffrect | 1 | 0.8308 | 0.7203 | 1.5988 | 6.1687 | 0 | 0 | best-epoch=119-step=4919 | 0.8308 |
| supervised-fpv02-s42 | supervised | 2 | 0.8852 | 0.8033 | 1.1994 | 5.2471 | 0 | 0 | best-epoch=41-step=3443 | 0.8852 |
| fixmatch-fpv02-s42 | fixmatch | 2 | 0.8864 | 0.8039 | 1.2644 | 5.7804 | 0 | 0 | best-epoch=53-step=4427 | 0.8864 |
| meanteacher-fpv02-s42 | meanteacher | 2 | 0.8956 | 0.8172 | 1.0515 | 4.7009 | 0 | 0 | best-epoch=53-step=4427 | 0.8956 |
| diffrect-fpv02-s42 | diffrect | 2 | 0.8282 | 0.7137 | 1.6501 | 6.3923 | 0 | 0 | best-epoch=65-step=5395 | 0.8282 |
| supervised-fpv03-s42 | supervised | 3 | 0.8890 | 0.8076 | 1.2565 | 6.2852 | 0 | 0 | best-epoch=47-step=5903 | 0.8890 |
| fixmatch-fpv03-s42 | fixmatch | 3 | 0.8857 | 0.8021 | 1.2088 | 5.5919 | 0 | 0 | best-epoch=55-step=6887 | 0.8857 |
| meanteacher-fpv03-s42 | meanteacher | 3 | 0.8845 | 0.8007 | 1.3347 | 6.3456 | 0 | 0 | best-epoch=27-step=3443 | 0.8845 |
| diffrect-fpv03-s42 | diffrect | 3 | 0.8478 | 0.7431 | 1.4378 | 5.5723 | 0 | 0 | best-epoch=43-step=5411 | 0.8478 |
| supervised-fpv05-s42 | supervised | 5 | 0.8867 | 0.8075 | 1.2146 | 5.3684 | 0 | 0 | best-epoch=29-step=5159 | 0.8867 |
| fixmatch-fpv05-s42 | fixmatch | 5 | 0.8839 | 0.7985 | 1.1915 | 5.3884 | 0 | 0 | best-epoch=23-step=4127 | 0.8839 |
| meanteacher-fpv05-s42 | meanteacher | 5 | 0.8887 | 0.8091 | 1.2038 | 5.2421 | 0 | 0 | best-epoch=26-step=4643 | 0.8887 |
| diffrect-fpv05-s42 | diffrect | 5 | 0.8546 | 0.7516 | 1.4229 | 5.4556 | 0 | 0 | best-epoch=26-step=4643 | 0.8546 |

### 7.8 Benchmark de arquiteturas (`logs/_archive/benchmarks/*/*/results.csv`)

Valores exatamente como gravados (`test::test/*`, média Lightning por batch). Linhas de benchmarks marcados como smoke test (`fast_dev_run` no `console.log`) estão indicadas.

| benchmark | experiment_name | test/dice_score | test/mean_iou | test/loss | fast_dev_run no log |
|---|---|---|---|---|---|
| vfss-baseline/005-2026-02-27-18-02-28 | vfss-unet | 0.105236 | 0.074042 | 0.708524 | sim |
| vfss-baseline/005-2026-02-27-18-02-28 | vfss-unm-unet | 0.062566 | 0.018616 | 0.691582 | sim |
| vfss-baseline/005-2026-02-27-18-02-28 | vfss-inca-transunet-r50-b16 | 0.000027 | 0.000013 | 0.585606 | sim |
| vfss-baseline/005-2026-02-27-18-02-28 | vfss-unm-transunet-r50-b16 | 0.000000 | 0.000000 | 0.662356 | sim |
| vfss-baseline/011-2026-05-08-20-26-26 | vfss-unm-unet | 0.069653 | 0.016534 | 0.688178 | sim |
| vfss-baseline/011-2026-05-08-20-26-26 | vfss-inca-swinunet-tiny | 0.000000 | 0.000000 | 0.174419 | sim |
| vfss-baseline/011-2026-05-08-20-26-26 | vfss-inca-transunet-r50-b16 | 0.046008 | 0.023462 | 0.698730 | sim |
| vfss-baseline/014-2026-05-09-17-53-57 | vfss-unet | 0.879288 | 0.746844 | 0.026596 | não |
| vfss-baseline/014-2026-05-09-17-53-57 | vfss-unm-unet | 0.794081 | 0.664525 | 0.034893 | não |
| vfss-baseline/014-2026-05-09-17-53-57 | vfss-inca-swinunet-tiny | 0.779937 | 0.600275 | 0.035320 | não |
| vfss-baseline/014-2026-05-09-17-53-57 | vfss-inca-swinunet-small | 0.730246 | 0.545085 | 0.044021 | não |
| vfss-baseline/014-2026-05-09-17-53-57 | vfss-inca-transunet-r50-b16 | 0.864535 | 0.721558 | 0.044233 | não |
| vfss-baseline/015-2026-08-13-19-20-11 | vfss-unet | 0.067266 | 0.025506 | 0.688424 | sim |
| vfss-baseline/015-2026-08-13-19-20-11 | vfss-unm-unet | 0.069653 | 0.016534 | 0.688178 | sim |
| vfss-baseline/015-2026-08-13-19-20-11 | vfss-inca-swinunet-tiny | 0.000000 | 0.000000 | 0.174419 | sim |
| vfss-baseline/015-2026-08-13-19-20-11 | vfss-inca-swinunet-small | 0.000000 | 0.000000 | 0.173082 | sim |
| vfss-baseline/015-2026-08-13-19-20-11 | vfss-inca-transunet-r50-b16 | 0.046007 | 0.023461 | 0.698730 | sim |
| vfss-baseline/016-2026-08-13-19-21-27 | vfss-unet | 0.877174 | 0.743918 | 0.026741 | não |
| vfss-baseline/016-2026-08-13-19-21-27 | vfss-unm-unet | 0.785010 | 0.648096 | 0.034843 | não |
| vfss-baseline/016-2026-08-13-19-21-27 | vfss-inca-swinunet-tiny | 0.779937 | 0.600275 | 0.035320 | não |
| vfss-baseline/016-2026-08-13-19-21-27 | vfss-inca-swinunet-small | 0.730246 | 0.545085 | 0.044021 | não |
| vfss-baseline/016-2026-08-13-19-21-27 | vfss-inca-transunet-r50-b16 | 0.867626 | 0.727719 | 0.039909 | não |
| vfss-swinunet/001-2026-02-26-19-50-19 | vfss-inca-swinunet-tiny | 0.000000 | 0.000000 | 0.255743 | sem console.log |
| vfss-swinunet/001-2026-02-26-19-50-19 | vfss-inca-swinunet-small | 0.000000 | 0.000000 | 0.251899 | sem console.log |
| vfss-swinunet/001-2026-02-26-19-50-19 | vfss-inca-swinunet-base | 0.000000 | 0.000000 | 0.315944 | sem console.log |
| vfss-swinunet/005-2026-02-26-20-20-21 | vfss-inca-swinunet-tiny | 0.000000 | 0.000000 | 0.281195 | sem console.log |
| vfss-swinunet/005-2026-02-26-20-20-21 | vfss-inca-swinunet-tiny | 0.000000 | 0.000000 | 0.279800 | sem console.log |
| vfss-swinunet/005-2026-02-26-20-20-21 | vfss-inca-swinunet-small | 0.000000 | 0.000000 | 0.337900 | sem console.log |
| vfss-swinunet/005-2026-02-26-20-20-21 | vfss-inca-swinunet-base | 0.001206 | 0.000609 | 0.405471 | sem console.log |
| vfss-swinunet/007-2026-02-27-22-39-05 | vfss-inca-swinunet-tiny | 0.641657 | 0.421568 | 0.171790 | não |
| vfss-swinunet/007-2026-02-27-22-39-05 | vfss-inca-swinunet-base | 0.423292 | 0.253201 | 0.178100 | não |
| vfss-swinunet/010-2026-03-02-14-31-33 | vfss-inca-swinunet-small | 0.605992 | 0.395648 | 0.172134 | não |
| vfss-swinunet/010-2026-03-02-14-31-33 | vfss-inca-swinunet-tiny | 0.641657 | 0.421568 | 0.171790 | não |

### 7.9 Runs arquivados com métrica de teste mas fora das CSVs (`logs/_archive/unclaimed/`, `incomplete/`)

"max_step" = último passo no `metrics.csv`; `test/dice_score` do `metrics.csv` quando existe. "rampa": forma do cronograma de consistência registrada no `hparams.yaml` ("—" = não registrada, caso dos wrappers supervisionados atuais, em que a rampa é inerte).

| diretório | target_variant | rampa | max_step | test/dice_score | checkpoint |
|---|---|---|---|---|---|
| incomplete/x1-frames-fixmatch-fpv01-s42/2026-09-17-23-39-04 | multiclass_c2_c4 | fração (atual) | 249 | — | — |
| incomplete/x1-unet-meanteacher/2026-09-08-14-59-36 | não registrado | — | — (sem metrics.csv) | — | — |
| incomplete/x1-unet-supervised/2026-09-07-18-58-42 | não registrado | — | — (sem metrics.csv) | — | — |
| unclaimed/acdc-diffrect-ssl4mis-10pct-s1337/2026-09-11-00-51-45 | não registrado | fração (atual) | 6799 | — | best-epoch=65-step=5609 |
| unclaimed/acdc-diffrect-ssl4mis-5pct-s1337/2026-09-10-22-54-03 | não registrado | fração (atual) | 30000 | 0.7931 | best-epoch=417-step=18809 |
| unclaimed/acdc-fixmatch-ssl4mis-5pct-s1337/2026-09-10-22-08-21 | não registrado | fração (atual) | 30000 | 0.7522 | best-epoch=571-step=25739 |
| unclaimed/acdc-meanteacher-ssl4mis-5pct-s1337/2026-09-10-21-30-32 | não registrado | fração (atual) | 30000 | 0.7738 | best-epoch=329-step=14849 |
| unclaimed/acdc-supervised-ssl4mis-5pct-s1337/2026-09-10-20-59-05 | não registrado | — | 30000 | 0.7680 | best-epoch=274-step=12374 |
| unclaimed/x1-diffrect-lf025-s42/2026-09-11-17-55-13 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7310 | best-epoch=153-step=6775 |
| unclaimed/x1-diffrect-lf050-s42/2026-09-11-18-45-28 | multiclass_c2_c4 | fração (atual) | 8000 | 0.8243 | best-epoch=77-step=7019 |
| unclaimed/x1-diffrect-lf075-s42/2026-09-11-20-20-55 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7707 | best-epoch=47-step=6191 |
| unclaimed/x1-diffrect-lf100-s42/2026-09-11-21-06-27 | multiclass_c2_c4 | fração (atual) | 8000 | 0.8286 | best-epoch=41-step=7223 |
| unclaimed/x1-fixmatch-lf010-s42/2026-09-08-00-49-14 | não registrado | divisor 150 (antiga) | 8000 | 0.3174 | best-epoch=144-step=2464 |
| unclaimed/x1-fixmatch-lf020-s42/2026-09-08-02-14-32 | não registrado | divisor 150 (antiga) | 8000 | 0.6894 | best-epoch=119-step=4799 |
| unclaimed/x1-fixmatch-lf025-s42/2026-09-11-17-12-19 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7042 | best-epoch=164-step=7259 |
| unclaimed/x1-fixmatch-lf050-s42/2026-09-08-03-12-24 | não registrado | divisor 150 (antiga) | 8000 | 0.8045 | best-epoch=41-step=3779 |
| unclaimed/x1-fixmatch-lf050-s42/2026-09-14-18-50-21 | multiclass_c2_c4 | fração (atual) | 5579 | — | best-epoch=35-step=3239 |
| unclaimed/x1-fixmatch-lf075-s42/2026-09-11-19-46-17 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7850 | best-epoch=43-step=5675 |
| unclaimed/x1-frames-diffrect-fpv01-s42/2026-09-18-13-14-09 | multiclass_c2_c4 | fração (atual) | 1699 | — | best-epoch=35-step=1475 |
| unclaimed/x1-frames-diffrect-fpv02-s42/2026-09-21-19-22-36 | multiclass_all | fração (atual) | 4799 | — | best-epoch=53-step=4427 |
| unclaimed/x1-frames-meanteacher-fpv05-s42/2026-09-22-02-26-05 | multiclass_all | fração (atual) | 4987 | — | best-epoch=20-step=3611 |
| unclaimed/x1-meanteacher-lf010-s42/2026-09-08-01-24-22 | não registrado | divisor 150 (antiga) | 8000 | 0.3592 | best-epoch=144-step=2464 |
| unclaimed/x1-meanteacher-lf020-s42/2026-09-08-02-38-10 | não registrado | divisor 150 (antiga) | 8000 | 0.6666 | best-epoch=95-step=3839 |
| unclaimed/x1-meanteacher-lf025-s42/2026-09-11-17-35-28 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7625 | best-epoch=65-step=2903 |
| unclaimed/x1-meanteacher-lf050-s42/2026-09-08-03-31-34 | não registrado | divisor 150 (antiga) | 8000 | 0.7696 | best-epoch=47-step=4319 |
| unclaimed/x1-meanteacher-lf075-s42/2026-09-11-20-05-38 | multiclass_c2_c4 | fração (atual) | 8000 | 0.7715 | best-epoch=31-step=4127 |
| unclaimed/x1-supervised-lf010-s42/2026-09-08-00-18-36 | não registrado | divisor 150 (antiga) | 8000 | 0.3646 | best-epoch=231-step=3943 |
| unclaimed/x1-supervised-lf020-s42/2026-09-08-01-55-20 | não registrado | divisor 150 (antiga) | 8000 | 0.6112 | best-epoch=59-step=2399 |
| unclaimed/x1-supervised-lf025-s42/2026-09-11-16-51-26 | multiclass_c2_c4 | — | 8000 | 0.6637 | best-epoch=109-step=4839 |
| unclaimed/x1-supervised-lf050-s42/2026-09-08-02-58-12 | não registrado | divisor 150 (antiga) | 8000 | 0.7445 | best-epoch=35-step=3239 |
| unclaimed/x1-supervised-lf075-s42/2026-09-11-19-32-17 | multiclass_c2_c4 | — | 8000 | 0.7575 | best-epoch=23-step=3095 |
| unclaimed/x1-supervised-lf100-s42/2026-09-08-03-47-39 | não registrado | divisor 150 (antiga) | 8000 | 0.8644 | best-epoch=41-step=3653 |
| unclaimed/x1b-supervised-W15-s42/2026-09-10-14-21-56 | multiclass_c2_c4 | — | 4149 | — | best-epoch=17-step=3095 |

---

## 8. Divergências ou inconsistências técnicas

**8.1 X1B: grade declarada em três lugares, três grades diferentes, e uma quarta executada.**
- `README.md`: "X1B compares `global`, `same_video`, and temporal windows of W = 3, 5 and 10 frames".
- `configs/sweep/x1b.yaml`: `policy: [global, same_video, "W=5", "W=10", "W=15"]`, `method: [supervised, fixmatch, meanteacher, diffrect]` → 20 células.
- Saída salva do notebook `experiment_1b_unlabeled_sampling_policy.ipynb` (célula 5): `Grade: 5 politicas x 3 metodos x 1 semente(s) = 15 runs`; célula 15 lista global, same_video, W=5, W=10, W=15.
- Executado (`logs/x1b/results_per_sample.csv`): 12 células — global, same_video, W=5, W=10 × supervised, fixmatch, meanteacher. Nenhum run DiffRect X1B e nenhum W=3 em lugar nenhum de `logs/`; o único W=15 é `_archive/unclaimed/x1b-supervised-W15-s42/2026-09-10-14-21-56`, parado no passo 4149.

**8.2 Convenções de ASSD/HD95 diferentes entre CSVs.** `x1b/results_per_sample.csv` (runs de 2026-09-10) ainda usa NaN para "estrutura presente de um lado só" (55 linhas NaN) e não tem coluna `undetected`; `x1/results_per_sample.csv` foi reprocessada para a penalidade de 362.0387 e tem `undetected`; as CSVs `x1-frames-per-video*` já nasceram com a penalidade (0 NaN), sem coluna `undetected`. ASSD/HD95 do X1B não estão na mesma convenção que os das outras famílias.

**8.3 Os `metrics.csv` do X1 carregam ASSD/HD95 da convenção antiga; os da X1-fpv, da nova.** Exemplo `x1 supervised-lf025-s42`: `test/assd` no `metrics.csv` = 4.7017 contra 13.0372 (média C2/C4 da CSV pós-A2) e 4.70645 (média C2/C4 da CSV pré-A2). Para `x1-frames-per-video supervised-fpv01-s42`: `test/assd` = 11.3945 e média C2/C4 da CSV = (9.2727 + 13.5163)/2 = 11.3945. Consequência factual: a mudança NaN → penalidade estava ativa no código entre o último run do X1 (2026-09-15 01:34) e o primeiro run da X1-fpv (2026-09-17 23:10), embora só tenha entrado no git em 2026-09-22 (`95e4467`). INFERÊNCIA A PARTIR DOS ARTEFATOS.

**8.4 A documentação diz que o notebook do X1 não foi re-executado após a correção A2; a saída salva do notebook diz o contrário.** `docs/x1-revisao-mudancas.md` ("As tabelas e gráficos do notebook (… células 25+) ainda não foram re-executados sobre o CSV corrigido"). A célula 35 de `experiment_1_does_ssl_helps.ipynb` mostra `assd @25%` C2 = 20.4800 e `hd95 @25%` C2 = 35.0751 — valores da CSV pós-A2 (na pré-A2 seriam 3.8186 e 19.1257).

**8.5 Configurações nominalmente idênticas, resultados diferentes.**

(a) X1 executado duas vezes com `hparams.yaml` idêntico (1ª vez em 2026-09-11, arquivada; 2ª em 2026-09-14, na CSV):

| run_id | hparams.yaml idêntico | 2026-09-11 test/dice_score | 2026-09-11 ckpt | 2026-09-14/15 test/dice_score | 2026-09-14/15 ckpt |
|---|---|---|---|---|---|
| supervised-lf025-s42 | sim | 0.6637 | best-epoch=109-step=4839 | 0.6219 | best-epoch=54-step=2419 |
| fixmatch-lf025-s42 | sim | 0.7042 | best-epoch=164-step=7259 | 0.6559 | best-epoch=153-step=6775 |
| meanteacher-lf025-s42 | sim | 0.7625 | best-epoch=65-step=2903 | 0.6612 | best-epoch=98-step=4355 |
| diffrect-lf025-s42 | sim | 0.7310 | best-epoch=153-step=6775 | 0.7824 | best-epoch=142-step=6291 |
| diffrect-lf050-s42 | sim | 0.8243 | best-epoch=77-step=7019 | 0.8340 | best-epoch=77-step=7019 |
| supervised-lf075-s42 | sim | 0.7575 | best-epoch=23-step=3095 | 0.7336 | best-epoch=39-step=5159 |
| fixmatch-lf075-s42 | sim | 0.7850 | best-epoch=43-step=5675 | 0.7552 | best-epoch=43-step=5675 |
| meanteacher-lf075-s42 | sim | 0.7715 | best-epoch=31-step=4127 | 0.7774 | best-epoch=27-step=3611 |
| diffrect-lf075-s42 | sim | 0.7707 | best-epoch=47-step=6191 | 0.8117 | best-epoch=51-step=6707 |
| diffrect-lf100-s42 | sim | 0.8286 | best-epoch=41-step=7223 | 0.8311 | best-epoch=38-step=6707 |

Em `diffrect-lf050` e `fixmatch-lf075` o checkpoint selecionado tem o mesmo epoch/passo nas duas execuções, e ainda assim o Dice de teste difere.

(b) X1 `lf100` × X1-fpv `fpv05`: ambos revelam os mesmos 690 frames de treino (seção 2.1); o `diff` dos `hparams.yaml` mostra só `name`, `label_group_column` (paciente_id × video_id) e a presença de `label_frames_per_group: 5` — ambos sem efeito sobre o conjunto revelado, que já é o conjunto inteiro.

| método | X1 lf100 dice C2 | X1 lf100 dice C4 | X1 lf100 ckpt | X1-fpv fpv05 dice C2 | X1-fpv fpv05 dice C4 | X1-fpv fpv05 ckpt | datas |
|---|---|---|---|---|---|---|---|
| supervised | 0.8476 | 0.7827 | best-epoch=17-step=3095 | 0.8433 | 0.7715 | best-epoch=32-step=5675 | 2026-09-15 × 2026-09-18 |
| fixmatch | 0.8731 | 0.7825 | best-epoch=20-step=3611 | 0.8303 | 0.7644 | best-epoch=41-step=7223 | 2026-09-15 × 2026-09-18 |
| meanteacher | 0.8542 | 0.8466 | best-epoch=23-step=4127 | 0.8181 | 0.8527 | best-epoch=26-step=4643 | 2026-09-15 × 2026-09-18 |
| diffrect | 0.8847 | 0.7775 | best-epoch=38-step=6707 | 0.8322 | 0.6739 | best-epoch=41-step=7223 | 2026-09-15 × 2026-09-18 |

Nenhum desses runs grava git sha; não é possível saber se a diferença vem de mudança de código ou de não determinismo (seção 9).

**8.6 Nome de run contradiz o config atual (multiclass_all).** `configs/sweep/x1-frames-per-video-multiclass-all.yaml` define `run_name_prefix: x1-frames-mc-all`, mas os 16 runs da CSV estão em `runs/x1-frames-<método>-fpvNN-s42/` e têm `name: x1-frames-…` no `hparams.yaml` — o mesmo nome dos runs de 4 classes em `logs/x1-frames-per-video/runs/`. Os dois conjuntos de diretórios só se distinguem pela família/timestamp. `tests/test_sweeps.py` documenta essa isenção (só o `name` pode divergir).

**8.7 Nome de run sem o eixo `backbone` (ACDC).** A linha `supervised-repo-10pct-s1337` da CSV aponta para `logs/acdc/runs/acdc-supervised-10pct-s1337/2026-09-10-17-25-08`. O `hparams.yaml` desse run registra `model/target: src.models.unet.UNet` — coerente com `backbone: repo` da CSV; só o nome do diretório não carrega o backbone.

**8.8 Run retomado com histórico de validação truncado no CSV logger.** `x1-frames-per-video-binario-all/runs/x1-frames-bin-all-diffrect-fpv02-s42/2026-09-23-18-43-42`: tentativa 1 terminou em `SystemExit: 1`; tentativa 2 retomou de `last.ckpt` no passo 2000 (`run.json: resumed_from_step 2000`; `_sweep/console.log`: "Retomando … (passo 2000)"). O `metrics.csv` só tem validação a partir do passo 2443 (12 pontos, contra 16 nos runs irmãos); o arquivo tfevents da 1ª tentativa (`events.out.tfevents.1790199822…`) continua no diretório. O `run.json` final tem `status: done` e **também** `error: "SystemExit: 1"` (herdado da tentativa anterior).

**8.9 "Unclaimed" descrito como "treinou inteiro", mas 6 desses runs são parciais.** `docs/mlops-execucao-e-tracking.md`: "`_archive/unclaimed/` (treinou inteiro, nenhuma CSV referencia)". Parciais pelo `metrics.csv`: `acdc-diffrect-ssl4mis-10pct` (6799/30000), `x1-fixmatch-lf050` 2026-09-14 (5579/8000), `x1-frames-diffrect-fpv01` (1699), `x1-frames-diffrect-fpv02` (4799), `x1-frames-meanteacher-fpv05` (4987), `x1b-supervised-W15` (4149). O critério de `incomplete/` era "sem `checkpoints/`"; esses têm um `best-*.ckpt`.

**8.10 Configs referenciados que não existem.** `configs/sweep/x1-frames-per-video-binario.yaml` e `configs/sweep/x1-frames-per-video-binario-c2-c6.yaml` (citados nos `console.log` de dry-run); variante `binary_c2_c3_c4_c5_c6` (padrão documentado em `scripts/build_binary_mask_variant.py`) ausente do disco e de `TARGET_VARIANTS`. Os configs da era das arquiteturas declaram `trainer.target: src.trainers.base.TrainerFromConfig`, módulo que não existe em `src/`; `Experiment` ignora esse `target` e usa `pl.Trainer(**trainer.params)`.

**8.11 Docstring e README descrevem um fluxo não rotulado que o X1 não produz.** `src/models/ssl/base.py` (cabeçalho) diz que chegam ao treino como não rotulados (1) frames anotados com rótulo escondido e (2) frames do pool; o README diz que o treino loga `train/pseudo_accuracy`, `train/pseudo_dice`… "against the ground truth the label regime hides". Nos 12 runs SSL do X1 essas chaves não existem no `metrics.csv` (ver 4.3). Elas existem nos runs X1-fpv com fpv 1, 2 e 3 e no ACDC.

**8.12 ACDC: unidades e métodos na comparação com o publicado.** A tabela publicada no comentário de `configs/experiment/acdc/acdc-unet-supervised.yaml` e no notebook tem HD95/ASD em mm; `evaluate_volumes` calcula sem `spacing` (voxels). No notebook, os runs `meanteacher` são pareados com a linha publicada "UAMT" (célula 15: coluna `linha publicada`).

**8.13 Contagem de células conferidas.** `docs/mlops-execucao-e-tracking.md` diz "83 células conferidas"; rodando `tests/test_sweeps.py` nesta auditoria: `{'acdc': 16, 'x1': 16, 'x1-binario': 0, 'x1-frames-per-video': 16, 'x1-frames-per-video-binario-all': 16, 'x1-frames-per-video-multiclass-all': 16, 'x1b': 12}` = 92 (4 testes, OK). Nem 92 nem 76 (92 menos os 16 runs `binary_all`, posteriores ao documento) coincidem com 83; o documento não detalha a contagem.

**8.14 Split legado incompleto.** A coluna `split_legado`, usada pelo modo padrão `split_mode: legado` (configs da era das arquiteturas), é NaN em 162 dos 961 frames; esses frames não entram em nenhum split nesse modo.

**8.15 Assimetria de augmentation entre métodos (protocolo, não erro de config).** No VFSS, supervised e Mean Teacher treinam as linhas rotuladas sem nenhuma augmentation; DiffRect calcula a loss supervisionada sobre a view fracamente aumentada (`supervise_weak_view: True`). O próprio YAML do DiffRect registra isso ("os outros tres metodos do X1 supervisionam a imagem crua"). No ACDC existe augmentation no dataset para todos os métodos; no VFSS, não.

---

## 9. Informações que o código NÃO permite determinar

1. **Versão exata do código de cada run.** Só os 16 runs `binary_all` registram git sha, e com `dirty: true`; os outros 76 runs referenciados pelas CSVs não registram sha. Parte relevante do código usado (`src/sweep.py`, `src/sweeps/`, mudanças em `src/experiment.py`, `src/data/base.py`, `src/data/samplers.py`) nunca foi commitada.
2. **Por que execuções nominalmente idênticas diferem** (8.5): mudança de código entre as datas, não determinismo de GPU (nenhum config liga `deterministic`), ou mudança no gerador do DataLoader — não há como separar com os artefatos existentes.
3. **Variância entre seeds.** Toda célula tem uma única seed; não existe nenhum dado para estimar dispersão entre seeds.
4. **Se a augmentation `random_generator` do ACDC estava ativa nos runs de 2026-09-10/11.** Não há flag no `hparams.yaml`; o arquivo `src/data/acdc_dataset.py` só entrou no git em 2026-09-14.
5. **Qual comportamento do gerador do DataLoader cada run teve** (seed fixa 0 × consumo do RNG global): a mudança está só na árvore suja.
6. **Se a política `same_video` caiu no fallback `global` em algum batch do X1/X1-fpv.** `draw_report` só foi gravado para o X1B.
7. **Nome anatômico das classes.** Os índices e suas frequências são verificáveis; o mapeamento índice → vértebra (C2…C8) só existe em comentários de YAML e no nome das variantes.
8. **Protocolo de anotação** (critério de escolha de `selected_labeler`, concordância entre os 5 rotuladores, como os 961 frames foram escolhidos por vídeo).
9. **Como `targets/mask/binary_all` foi gerado.** O conteúdo coincide com `multiclass_all > 0`, mas o script existente documenta outra variante como saída padrão; não há log da geração.
10. **Configuração efetiva da era das arquiteturas.** Os `hparams.yaml` desses runs registram só modelo/otimizador; dataset, split, `target_variant` e seed não foram registrados, e os YAML atuais foram editados depois (ex.: `split_mode`, `target_variant` comentados).
11. **Conteúdo e splits do dataset UNM** (não auditado aqui).
12. **Se alguma decisão de hiperparâmetro olhou o conjunto de teste.** O código seleciona checkpoint na validação; o histórico de decisões não é recuperável dos artefatos.
13. **Quais notebooks e figuras refletem qual versão das CSVs**, além das células conferidas aqui (X1 células 35/37; ACDC 15/16/19; X1-fpv/multiclass/binario-all células 9–24).
14. **Origem dos números publicados** usados na comparação do ACDC (digitados no notebook e no comentário do YAML; não verificáveis a partir do repositório).
15. **Métodos, orçamentos ou seeds que tenham rodado em outra máquina ou outro diretório de logs.** Todos os 346 arquivos tfevents encontrados vêm do host `delaunay`; não há como saber de execuções fora de `logs/`.
