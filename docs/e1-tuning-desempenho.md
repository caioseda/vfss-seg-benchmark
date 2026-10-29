# Desempenho do tuning E1: o que foi medido e o que mudou

> Escrito em 2026-10-28, durante a busca `e1-tuning-search` (interrompida em 21:03 na célula
> `unetplusplus-t11-s42`, com 56/125 células prontas). É uma auditoria de implementação, não um
> experimento: o que se mede aqui é **tempo de parede por passo**. A qualidade só entra para
> verificar que nada mudou no que o modelo vê.

## 1. Diagnóstico

Os eventos do TensorBoard da busca dão o tempo por passo de cada trial:

| trial (U-Net, 4000 passos, não podado) | s/passo | trial inteiro |
|---|---|---|
| t00 (1ª célula do processo) | 0,20 | 16,5 min |
| t02 | 0,31 | 24 min |
| t04 | 0,40 | 31 min |
| t11 (podado no passo 1000) | 0,70 | — |

Há dois problemas independentes.

**(a) Custo fixo: a virada de época.** O tuning treina num fold só (121 frames, ~15 passos por
época). A cada época o `DataLoader` recriava 8 workers, e cada frame era decodificado de um PNG
1024×1024 (~48 ms por frame). Num processo novo: passo de GPU 0,053 s, **virada de época 2,1 s**,
validação 7 s. Cerca de 75% do tempo de parede ia para a virada de época.

**(b) Custo crescente: vazamento por célula.** O `Trainer` do Lightning atribui
`callback.log = pl_module.log` a todo callback. O `E1TuningSweep` guardava o `OptunaPruning` de cada
célula em `self._pruning` e nunca o soltava, então **toda célula já treinada continuava viva**:
módulo, estado do otimizador na GPU, trainer, datamodule e datasets. Com 57 células, o processo
estava em 19 GB de GPU e 14 GB de RSS. A desaceleração linear acompanha esse vazamento. O
mecanismo exato (coleta de lixo sobre um heap crescente, fork de um pai maior) não foi isolado: um
lastro artificial de 12 GB explicou só +0,2 s por época.

Hipóteses testadas e descartadas: MLflow em SQLite (7 ms por `log_batch` mesmo no experimento com
57 runs), RSS em si via fork (+0,2 s por época com 12 GB de lastro), barra de progresso Rich num
pseudo-terminal (sem diferença).

## 2. Escada de mudanças (U-Net, 300 passos, GPU sem outro processo)

Cada degrau adiciona **uma** mudança ao anterior:

| degrau | s/passo amortizado | Δ | virada de época | validação |
|---|---|---|---|---|
| L0 baseline | 0,296 | — | 2,08 s | 7,1 s |
| L1 + `persistent_train_workers` | 0,219 | −26% | 0,99 s | 7,2 s |
| L2 + `cache_splits: [train, validation]` | 0,125 | −43% | 0,22 s | 4,7 s |
| L3 + `num_workers: 0` | 0,072 | −43% | 0,05 s | 2,8 s |
| L4 + `precision: bf16-mixed` | 0,082 | **+14%** | 0,05 s | 2,9 s |

Ruído entre réplicas idênticas: ~10–13% (U-Net L3 deu 0,0715 e 0,0799). Só efeitos acima de ~15%
contam.

## 3. Opções isoladas sobre o L3 (passo estável, mediana)

| opção | U-Net | Swin-UNet | decisão |
|---|---|---|---|
| L3 | 0,038 s | 0,099 s | — |
| `cudnn.benchmark` | = | −9% | descartada (ruído) |
| TF32 em matmul (`set_float32_matmul_precision('high')`) | = | **−19%** | não adotada, ver abaixo |
| bf16-mixed | +7% | −7% | descartada |
| channels_last | = | −13% | descartada (ruído; Swin quase não tem conv) |
| `torch.compile` | val 14 s, run 6× mais lento | val 221 s, run 17× mais lento | descartada |

- **bf16** não compensa com batch 8 em 256 px: o overhead do autocast supera o ganho do tensor
  core.
- **compile** recompila a cada formato novo (o último batch da validação tem outro tamanho), e o
  custo de compilação não se paga em 4000 passos.
- **TF32** é o único ganho real fora do pipeline de dados, e só no Swin. Ele muda a numérica das
  camadas lineares do Swin no meio da busca (os convs das CNNs já rodam em TF32 por padrão no
  PyTorch). Na busca inteira o ganho é de ~4%. Ficou de fora para não introduzir essa assimetria.

**Paralelismo de trials** não foi implementado. O `Sweep` faz read-modify-write da CSV e do
`state.json` sem lock, e o TPE com trials concorrentes precisa de *constant liar* para não sortear
pontos repetidos. Com o ganho do item 4, não compensa o risco.

## 4. O que ficou no código

| arquivo | mudança | padrão |
|---|---|---|
| `src/data/base.py` | `cache_splits`, `cache_num_workers`, `persistent_train_workers` em `DataModuleFromConfig` | desligado, X1/ACDC intocados |
| `src/sweeps/e1.py` | `E1TuningSweep.finish_cell` solta `_pruning`/`_trials` da célula | sempre |
| `configs/experiment/e1/*.yaml` | `num_workers: 0`, `cache_splits: [train, validation]`, `cache_num_workers: 8` | só E1 |
| `tests/test_datamodule_speed.py` | exatidão do cache, persistência só no treino | — |
| `tests/test_e1_protocol.py` | `test_finished_cell_is_released` | — |

`persistent_train_workers` existe, mas o E1 não o usa: com cache e `num_workers: 0` ele não tem
efeito, e, ligado sozinho, muda a ordem de embaralhamento a partir da 2ª época.

## 5. Validação no maquinário real (`Sweep` + Optuna + `evaluate` + CSV)

U-Net, trials completos de 4000 passos, validação a cada 500 passos, 3 células no mesmo processo:

| | célula inteira | FDs depois da célula | RSS |
|---|---|---|---|
| antes (busca real, t00, processo novo) | 16,5 min | — | 14 GB em 57 células |
| depois | **3,6 / 3,6 / 3,6 min** | 506 / 506 / 506 | 2,65 → 2,84 → 2,84 GB |

Swin-UNet, L0 → L3: 0,377 → 0,154 s/passo (2,4×).

## 6. A entrada do treino não mudou

- **Amostras:** o cache devolve exatamente o `__getitem__`, conferido nos 231 frames reais (treino
  fold 0 + validação): tensores iguais bit a bit, metadados com o mesmo tipo e valor.
- **Ordem:** com `num_workers: 0`, cada época ainda cria um iterador novo, que sorteia o
  `_base_seed` do mesmo gerador, como no caso com 8 workers não persistentes. Em 3 épocas (45
  batches), a config antiga e a nova produzem **a mesma sequência de frames e os mesmos pixels**.
- **Vazamento:** a correção só libera memória e não toca em nada do treino.

A única diferença que sobra entre um trial antigo e um novo é o não-determinismo da GPU, que já
existia: re-runs idênticos já divergiam.

## 7. Continuar a busca com a config nova

É válido continuar sobre os 56 trials existentes. Os trials antigos e novos medem a mesma função
objetivo (seção 6), então o TPE e o `MedianPruner` comparam valores comparáveis.

Na prática, a célula interrompida (`unetplusplus-t11-s42`) tem `last.ckpt` com o fingerprint da
config antiga, e `Experiment` recusa retomar quando o fingerprint muda. Como a mudança é só de
pipeline (seção 6), retomar com `--force-resume` é justificável, e o forçamento fica registrado no
`run.json` da célula (`forced_resume_despite_config_change`):

```bash
# a sessao tmux antiga ainda existe (shell parado): feche-a antes, ou use outro nome
~/miniconda3/envs/vfss/bin/tmux kill-session -t e1-tuning-search
scripts/submit.sh configs/sweep/e1-tuning-search.yaml e1-tuning-search --force-resume
```

Os `config.yaml` de cada run registram `num_workers`/`cache_splits`, então o corte entre trials
antigos e novos fica auditável.
