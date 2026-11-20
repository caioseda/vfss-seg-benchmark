# Treino distribuído: delaunay + DeepLearning01/05

Este guia explica como rodar sweeps na GPU1 dos servidores DeepLearning01 e DeepLearning05 (em container,
com dados em `/local/<usuario>`, até 200 GB) e, se quiser, também na delaunay. A delaunay continua sendo a
cópia central de logs, CSVs, TensorBoard e MLflow. Tudo é disparado **da delaunay** com `scripts/remote.py`.
Os hosts ficam declarados em `configs/remote_hosts.yaml`.

## Ideia em quatro peças

1. **Orquestração.** A delaunay copia o código e os dados, dá a partida com `docker run -d` via SSH e
   **puxa** os logs. Os servidores remotos nunca precisam acessar a delaunay.
2. **Mesmos caminhos.** O container monta os dados em `/data_ssd/caioseda/data` e o repositório no
   caminho da delaunay. Por isso os YAMLs não mudam, e `output_dir` e `best_checkpoint_path` continuam
   válidos depois do sync.
3. **Shards por unidade de pareamento.** `--shard K/N` é 1-indexado: uma divisão em dois usa `1/2` e `2/2`.
   Todos os métodos de um mesmo fold/seed rodam no mesmo shard, logo na mesma GPU. Como as GPUs são de
   modelos diferentes, a diferença pareada entre métodos fica livre do efeito de hardware. Pelo mesmo
   motivo, o DTC-TRUS acha a etapa 1 (o supervisionado do mesmo fold) no `state.json` do próprio shard.
   No E1 a unidade é o fold/seed com todas as arquiteturas. A busca com Optuna (`e1-tuning-search`)
   recusa shard, porque o estado do study é compartilhado entre as células.
4. **Isolamento.** Cada shard grava estado e CSVs em `logs/<familia>/_sweep/<tag>/`, nunca nos arquivos
   canônicos. Um host remoto **sempre** roda como shard: sem `--shard`, ele usa `1/1`, que é a grade
   inteira. O `sync` nunca traz arquivos canônicos, e o `merge` os consolida de forma explícita.

## Permissões

O container roda com `--user "$(id -u):$(id -g)"`, expandido **no host**, e com `umask 0002`. Todo
diretório montado é criado com o seu usuário **antes** do `docker run`. Sem isso, o Docker criaria esses
diretórios como root, e você não conseguiria mover nem apagar o que o treino gravou. O `HOME` e os caches
(`TORCH_HOME`, `HF_HOME`) ficam em `<root>/cache`.

## Uma vez por máquina

```bash
python scripts/remote.py build                           # constrói a imagem na delaunay
python scripts/remote.py setup dl05 --datasets inca      # cria as pastas, envia a imagem, copia os dados e testa a GPU
python scripts/remote.py setup dl01 --datasets inca unm  # o e2.yaml final também avalia no UNM (+50 GB)
python scripts/remote.py setup delaunay                  # só o teste de GPU com Docker
```

- **A imagem instala o lock do env `vfss`, não o `environment.yml`.** São dois arquivos:
  `docker/conda-explicit.txt` (URLs exatas, sem solve) e `docker/pip-requirements.txt`. Resolver o
  `environment.yml` do zero trouxe outro Lightning, numpy, pandas, omegaconf e torchmetrics, e a versão
  nova do Lightning chegou a quebrar o treino. Com o lock, a imagem difere do env só em duas coisas, ambas
  deliberadas: `opencv-python-headless` (mesma versão, sem libGL) e o `vfss_data_split` editável, que entra
  por `python_paths` em `configs/remote_hosts.yaml`. Se você instalar ou atualizar algo no env `vfss`, rode
  `python scripts/remote.py lock` e `build` de novo.
- A imagem é construída **uma vez** na delaunay e levada aos outros hosts com `docker save | docker load`,
  então todos têm exatamente as mesmas versões. A tag (`vfss-seg:env-<hash>`) muda sozinha quando o lock ou
  o `docker/Dockerfile` mudam. Nesse caso, rode `setup` de novo em cada host para levar a imagem nova.
- O `setup` recusa a cópia se ela fosse ultrapassar `quota_gb`. Espaço necessário: INCA ≈ 29 GB, UNM ≈ 50 GB,
  e cada run do E2 ≈ 0,35 GB.

## Disparar

```bash
# Grade inteira num host
python scripts/remote.py submit dl05 configs/sweep/e2-cv.yaml
python scripts/remote.py submit delaunay configs/sweep/e2-cv.yaml        # Docker na delaunay

# Grade dividida em 3
python scripts/remote.py submit dl01     configs/sweep/e2-cv.yaml --shard 1/3
python scripts/remote.py submit dl05     configs/sweep/e2-cv.yaml --shard 2/3
python scripts/remote.py submit delaunay configs/sweep/e2-cv.yaml --shard 3/3

# Conferir a divisão antes de treinar (não lança nada)
python scripts/run_sweep.py --config configs/sweep/e2-cv.yaml --shard 1/3 --dry-run
```

- **O que o `submit` faz.** (1) Recusa se o container já estiver rodando. (2) Confere se o host tem os
  datasets que o sweep usa e se cabe na cota. (3) Copia um **retrato** do código para
  `<root>/code/<container>/`, de modo que editar o repositório depois não muda um sweep em andamento.
  (4) Executa `docker run -d`. O container sobrevive à queda do SSH, como o tmux.
- **Retomar depois de uma queda.** Rode o **mesmo** comando. As células concluídas são puladas e a
  interrompida retoma de `last.ckpt`.
- **Na delaunay sem shard,** o container grava direto nos arquivos canônicos, como o `scripts/submit.sh`.
  Não combine isso com shards remotos do mesmo sweep: ao dividir, todos os processos usam `--shard`.
- **Uma ressalva.** O `e2.yaml` final com `seed: [42]` tem uma única unidade de pareamento e roda inteiro
  numa máquina. Só há ganho de paralelismo com mais seeds/políticas ou no `e2-cv` (5 folds).
- `--dry-run` antes do subcomando imprime o que seria executado: `python scripts/remote.py --dry-run submit ...`.

## Acompanhar

```bash
python scripts/remote.py status                               # containers, GPU e disco de cada host
python scripts/remote.py logs dl05 configs/sweep/e2-cv.yaml   # console ao vivo; Ctrl-C só para de olhar
python scripts/remote.py stop dl05 configs/sweep/e2-cv.yaml   # para o container; relançar retoma

# Em tmux na delaunay: puxa os logs a cada 5 min
python scripts/remote.py sync --family e2-cv --every 300

# TensorBoard sobre os logs centralizados (locais + remotos, com atraso de no máximo 5 min)
tensorboard --logdir logs/e2-cv --host 127.0.0.1 --port 6006
# no seu computador: ssh -L 6006:127.0.0.1:6006 delaunay  ->  http://localhost:6006
```

- **O que o `sync` traz.** Event files, `metrics.csv`, `run.json`, as CSVs e o estado de cada shard. Os
  checkpoints vêm só dos runs **concluídos**, porque o best de um run em andamento ainda muda de nome.
  O `sync` nunca usa `--delete`, então nada local é apagado.
- **Estado agregado no notebook.** `sweep_status("configs/sweep/e2-cv.yaml")` e `watch_sweep(...)`, sem
  shard, mostram a grade inteira com uma coluna `host`, a partir dos `state.json` sincronizados.

## Juntar

```bash
python scripts/remote.py sync --family e2-cv               # uma última passada
python scripts/remote.py merge configs/sweep/e2-cv.yaml    # CSVs dos shards -> logs/e2-cv/results_*.csv
python scripts/index_runs.py --backend mlflow              # runs remotos no MLflow da delaunay
```

- **O que o `merge` faz.** Acrescenta às CSVs canônicas apenas os `run_id` novos (a canônica tem
  precedência) e registra as células concluídas no `_sweep/state.json` canônico. É idempotente.
- **Por que o MLflow fica desligado nos remotos.** Os remotos rodam com `VFSS_TRACKING=off`, porque um
  banco sqlite não aceita escrita concorrente pela rede. O `index_runs.py` é idempotente e indexa os runs
  depois do sync.

## Retreinar só algumas células

Quando um bug afeta só um método (ex.: o FixMatch do E2, 2026-11-17), as células dele já estão "prontas"
nas CSVs e no `state.json` e seriam puladas. Invalide-as na delaunay **e** em cada host que rodou a família
(senão o próximo `sync` traz de volta os runs velhos e o estado do shard). Nada é apagado: vai para
`logs/_arquivo/<familia>-<rotulo>/` na delaunay e `<root>/arquivo/<familia>-<rotulo>/` no host.

```bash
FM='axes.method=[fixmatch]'
python scripts/invalidate_cells.py --config configs/sweep/e2-cv.yaml --set "$FM" --label fixmatch-bn-dreno --reason "..." --dry-run
python scripts/invalidate_cells.py --config configs/sweep/e2-cv.yaml --set "$FM" --label fixmatch-bn-dreno --reason "..."
python scripts/remote.py invalidate dl05 configs/sweep/e2-cv.yaml --set "$FM" --label fixmatch-bn-dreno --reason "..."
python scripts/remote.py submit dl05 configs/sweep/e2-cv.yaml --set "$FM"     # só as células recortadas
# depois: sync + merge como acima; os notebooks leem as CSVs canônicas e passam a mostrar o run novo
```

## Limpeza no host

Depois de `sync` e `merge`, os runs remotos já estão na delaunay. Para liberar espaço:

```bash
ssh Deeplearning05 'rm -rf /local/$USER/vfss/logs/e2-cv /local/$USER/vfss/code/vfss-e2-cv-dl05'
```
