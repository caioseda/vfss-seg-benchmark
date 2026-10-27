# Execução de experimentos, organização de logs e tracking

Registro das decisões por trás de `scripts/submit.sh`, `src/sweep.py`, `src/tracking.py` e do
layout de `logs/`. O que está aqui é o *porquê*; o *como* está nas docstrings.

## O problema

Os treinos rodavam em células de notebook sobre uma sessão SSH. Uma queda de conexão matava o
processo e, como `trainer.fit()` nunca era chamado com `ckpt_path=`, a célula recomeçava do zero —
um run de DiffRect no ACDC (~2 h) perdido inteiro. Em paralelo, `logs/` tinha 85 entradas planas e
54 GB, e `tensorboard --logdir logs/` misturava todos os experimentos de todos os meses.

## O layout

```
logs/<família>/
    results_*.csv          o registro científico (versionado no git)
    predictions/*.npz      cache de previsões, indexado por run_id
    runs/<nome>/<uid>/      config.yaml  run.json  console.log
                            tensorboard/  csv/metrics.csv  checkpoints/
    _sweep/                 state.json + console.log da grade
logs/_archive/             eras encerradas; nada apagado
logs/_tracking/            stores dos trackers
```

**Invariante:** `logs/<família>/runs/` contém *exatamente* os runs que a CSV de resultado daquela
família referencia. É o que faz `tensorboard --logdir logs/x1/runs` mostrar as 16 células da grade
publicada, e não também as tentativas anteriores e os orçamentos de um desenho que o próprio
notebook declara não comparável. O resto foi para `_archive/unclaimed/` (treinou inteiro, nenhuma
CSV referencia) e `_archive/incomplete/` (sem `checkpoints/`).

Dois experimentos ao mesmo tempo, com legenda própria para cada árvore:

```bash
tensorboard --logdir logs/x1/runs
tensorboard --logdir_spec 4cls:logs/x1-frames-per-video/runs,8cls:logs/x1-frames-per-video-multiclass-all/runs
```

### A colisão que motivou o invariante

`experiment_1_does_ssl_helps_frames_per_video.ipynb` (4 classes) e a variante `multiclass_all`
(8 classes) usavam o **mesmo** template de `name`. Resultado medido em disco:
`logs/x1-frames-diffrect-fpv01-s42/` guardava um run com 46 tags e outro com 76, distinguíveis só
pelo timestamp. O TensorBoard sobrepunha curvas incomparáveis e `find_checkpoint()` podia devolver o
checkpoint errado. A migração os separou **por timestamp**, via a coluna `output_dir` das CSVs — o
nome do diretório, sozinho, era ambíguo. Cada grade tem agora um `run_name_prefix` próprio.

## Execução: CLI + tmux, retomada em dois níveis

```bash
scripts/submit.sh configs/sweep/x1.yaml       # sessão tmux destacada
tmux attach -t x1                             # Ctrl-b d sai sem matar
tail -f logs/x1/_sweep/console.log
```

- **Nível 1** — células concluídas são puladas. A fonte de verdade é a CSV de resultado da família,
  a *mesma* que os notebooks já escreviam; só depois o `_sweep/state.json`. É por isso que lançar um
  sweep sobre uma grade já rodada não retreina nada.
- **Nível 2** — uma célula interrompida retoma de `checkpoints/last.ckpt`, escrito a cada
  `resume_every_n_train_steps` (500 por padrão) e **apagado ao concluir**, então o custo de disco é
  zero no fim. Depende de o `run_uid` ser atribuído uma vez e reusado; cunhar um timestamp novo a
  cada tentativa é o que produzia os diretórios órfãos que a migração varreu.

Um `status: running` encontrado no início **não** significa que algo está rodando: um SIGKILL não dá
chance de escrever `failed`. Um `running` obsoleto é tratado como retomável.

`conda run` não preserva o diretório de trabalho, então `submit.sh` faz um `cd` explícito dentro do
comando; e `--no-capture-output` é obrigatório, senão o `console.log` só aparece no fim.

## Retomada fiel

O requisito é que os hiperparâmetros voltem e progridam como se não houvesse interrupção.

**Já garantido pelo checkpoint do Lightning** (`loops`, `optimizer_states`, `lr_schedulers`,
`callbacks`, `state_dict` do módulo e do DataModule): `global_step`, momentos do AdamW, o LR do
scheduler, os pesos EMA do teacher (submódulo), e — o que mais assustaria — a rampa de consistência
e o decaimento EMA, porque ambos são **função de `global_step`** e não guardam contador próprio.

**Três lacunas, fechadas:**

1. **RNG.** `_collect_rng_states` existe em `lightning_fabric/utilities/seed.py` mas só o launcher de
   multiprocessing o usa — `dump_checkpoint` nunca o grava. Sem isso divergem as augmentations, o
   ruído do teacher, o cutout do FixMatch e o `q_sample` do DiffRect. → `ResumeFidelity`.
2. **Posição do sampler.** `TwoStreamBatchSampler.state_dict()` guarda `(época, próximo batch)`, e o
   `__iter__` **reproduz** os sorteios já feitos ao retomar — não basta pular, porque o `rng` é
   consumido a cada batch.
3. **Deriva de configuração.** `Experiment` grava o sha256 da config resolvida e **aborta** a
   retomada se mudou. O caso silencioso é o perigoso: alterar `max_steps` reescreve a rampa de
   consistência e o `T_max` do cosseno *só para o trecho restante*.

Dois detalhes descobertos na prática, ambos invisíveis sem o teste:

- **`on_train_epoch_start` não dispara** quando o run retoma no meio de uma época. O restore do RNG
  tem de ir em `on_train_start`, senão pousa uma época tarde e o fluxo fica defasado.
- **O DataLoader precisou de um `generator` próprio.** `_BaseDataLoaderIter.__init__` sorteia um
  `_base_seed` do RNG global a cada iterador criado, e um run retomado cria um iterador onde o
  ininterrupto não criou. Sem isolar esse sorteio, a retomada fica permanentemente um sorteio adiante
  e nenhuma augmentation coincide.

`tests/test_resume.py` prova o resultado e **ablaciona** cada peça: sem `ResumeFidelity` 6 losses
divergem; sem o estado do sampler 2 batches divergem e o erro é 45x maior. O escopo é explícito: o
cronograma é exato sempre; a trajetória completa é bit-exata com `num_workers=0`.

## Tracking: índice, nunca cópia

O diretório de run é a única cópia canônica. O TensorBoard lê a árvore direto; MLflow e Aim recebem
params, a série escalar e uma tag `run_dir` apontando de volta — e **nenhum artefato**
(`log_model=False`, nunca `log_artifact`).

Ser honesto sobre o limite: um tracker *é* um índice, e um índice guarda a série escalar. O que o
desenho garante é que nada **pesado** é duplicado — os ~53 GB de checkpoints, os event files e os
`.npz` ficam exatamente uma vez.

Números medidos no acervo (186 runs, 4 layouts históricos):

| | |
|---|---|
| pontos de métrica indexáveis | 542.491 |
| dos quais NaN genuínos | 18.122 |
| `DeviceStatsMonitor.*` excluído | corta 27% dos pontos |

`DeviceStatsMonitor.*` fica de fora: são as chaves mais longas do acervo e não respondem nenhuma
pergunta de pesquisa. Ficam no TensorBoard, que é o lugar delas.

`scripts/index_runs.py` lê `metrics.csv` com o módulo `csv` e **não** com pandas, de propósito: uma
célula vazia significa "nunca logado" e a string `nan` significa um NaN de verdade. O pandas colapsa
as duas, e o artefato de NaN em ASSD/HD95 é conhecido neste projeto — confundi-las corromperia o
registro. Para os runs que só têm event files, o fallback usa
`tensorboard.backend.event_processing`, que já está instalado; `aim convert tensorboard` foi
descartado porque exige `import tensorflow` e, sem ele, **falha em silêncio**.

### Estado dos dois backends

**MLflow — instalado, ligado por padrão, ao vivo e retroativo.** O bloqueador nunca foi o MLflow:
era o pin `protobuf==3.20.*`, que `mlflow>=3` forçaria a subir via `opentelemetry-proto`. Removido do
`environment.yml`. Os 230 runs do acervo foram indexados (638.259 pontos), e os runs novos chegam ao
MLflow **durante o treino**, sem passo manual.

```bash
mlflow ui --backend-store-uri sqlite:///logs/_tracking/mlflow.db
python scripts/index_runs.py --verify
```

Quais trackers acompanham o treino é decidido em **um** arquivo, `configs/tracking.yaml`, com
precedência `defaults em src/tracking.py < configs/tracking.yaml < bloco logging: do YAML do
experimento`. `VFSS_TRACKING=off` desliga tudo sem editar nada.

Três propriedades do tracking ao vivo, cada uma com um teste que **falha se ela for removida**
(`tests/test_tracking.py`, verificado por ablação):

- **Mesma identidade do indexador.** O tag `run_uid` é `sha1(caminho relativo a logs/)` nos dois
  lados, calculado pela mesma função. Se divergissem, o backfill não reconheceria o run logado ao
  vivo e o reindexaria: o mesmo treino apareceria duas vezes.
- **Retomada continua o mesmo run.** Uma retomada reabre o mesmo diretório e portanto o mesmo
  `run_uid`; o logger procura o run existente por tag e o reaproveita. Sem isso cada tentativa abriria
  um run novo para o que é um único treino.
- **O tracker nunca derruba o treino.** Uma exceção do backend (banco travado, disco cheio) dentro de
  `log_metrics` interromperia um treino de horas por algo que não é o experimento. Cada falha é
  avisada (as três primeiras) e contada, e o run segue; a fonte da verdade continua sendo o
  `csv/metrics.csv`. Um tracker que não consegue nem ser *construído* (não instalado, URI inválida) é
  pulado com um aviso. Para reparar um run cujo tracker falhou no meio:
  `python scripts/index_runs.py --replace --family <família>`.

**O número publicado.** Só nasce no `evaluate()`, depois de os loggers fecharem, então um run logado
ao vivo não o teria. O `Sweep` o anexa ao run do MLflow assim que a linha entra na CSV de resultado
(`attach_published_metrics`), e `index_runs.py` o completa em qualquer run que ainda não o tenha — o
que também cobre runs cuja CSV foi escrita depois.

Nos notebooks, use `mlflow.search_runs(..., output_format="list")` ou `MlflowClient.search_runs()`
enquanto `mlflow/mlflow#22349` (suporte a pandas 3.x) estiver aberto.

### Abrindo a interface do MLflow (principalmente por túnel)

```bash
python scripts/mlflow_ui.py            # sobe o mlflow ui + um proxy; imprime um link por experimento
# no seu computador:  ssh -L 5000:127.0.0.1:5000 usuario@servidor   ->   http://localhost:5000
```

Três coisas, todas reproduzidas contra o MLflow 3.16.1 desta máquina, fazem a interface parecer
quebrada quando não está:

1. **Ela abre em "GenAI", que está vazia para estes experimentos.** O MLflow 3.x tem uma chave no
   canto superior esquerdo, "GenAI | Model training", e abre na primeira: `Overview` com "Traces 0 ·
   No data available". Os runs de treino estão em **"Model training"** (URL `…/runs?workflowType=
   machine_learning`). A escolha fica no `localStorage` do navegador, então vale por navegador;
   o lançador imprime links que já caem na visão certa.
2. **O servidor não comprime nada.** A página de runs baixa ~8,5 MB (`main.js` 4,4 MB, um chunk de
   1,15 MB, 0,83 MB de CSS). Por um túnel lento isso é o "carregando para sempre"; os estáticos têm
   cache de 28 dias, então o peso é só da primeira visita — e se ela nunca termina o cache nunca
   enche. Pelo proxy a mesma página baixa **2,1 MB**.
3. **POSTs com `Origin` que não seja `localhost` levam 403.** O middleware de segurança responde
   "Cross-origin request blocked" e `runs/search` e `experiments/search` são POSTs. Num túnel
   `ssh -L` o Origin é `http://localhost:PORTA` e passa; num túnel público (`https://…devtunnels.ms`,
   ngrok, cloudflared) a lista de experimentos carrega (GET) e a tabela de runs fica vazia. O proxy
   reescreve `Origin`/`Referer`, então funciona seja qual for o túnel. Para confirmar sem o proxy:
   aba *Network* do navegador, status 403 em `runs/search`.

O proxy (`src/mlflow_proxy.py`) só repassa HTTP e não conhece o MLflow, então não depende de nenhum
detalhe interno dele. Os testes (`tests/test_mlflow_proxy.py`) rodam contra um servidor falso que
**reproduz o 403**, de modo que um teste prova que é o proxy que o contorna, e cada proteção foi
verificada por ablação.

**As datas dos runs retro-indexados.** O backfill cria o run no instante em que indexa. Sem tratamento,
a interface mostrava "Created: 5 hours ago" e "Duration: 1.7s" para um treino que durou 48 min e
aconteceu semanas antes, e o eixo de tempo dos gráficos caía em 1970. `run_time_span` reconstrói o
início de `run.json` (runs novos) ou do timestamp no nome do diretório, e o fim do `finished_at` ou
da data do último arquivo de métrica; os timestamps das métricas são interpolados entre os dois
(aproximação: serve para ordenar e dar a escala, não para medir a duração de um passo). Runs
indexados antes dessa correção continuam com a data errada até serem reindexados:

```bash
python scripts/index_runs.py --replace --family x1    # um experimento
python scripts/index_runs.py --replace                # todos
```

`--replace` faz soft-delete do run antigo e cria um novo, então o banco cresce até um `mlflow gc`.
Rode `mlflow gc` com cuidado: ele apaga **todos** os runs soft-deletados do banco, inclusive os que você
tenha apagado na interface de propósito; `mlflow gc --run-ids <ids>` apaga só os listados.

**Aim — não funciona nesta versão, e fica desligado.** Medido no `aim 3.29.1` / `aimrocks 0.5.2`: o
run é criado (o hash e o `name` persistem) mas **params e métricas não**. `run[chave] = valor` é um
no-op silencioso, porque `aim/ext/exception_resistant.py` engole a exceção; `repo.iter_runs()`
devolve zero, e `Run.metrics()` volta vazio após reabrir. Consequência prática: indexar *parecia*
funcionar, escrevia ~76 MB de chunks por 16 runs e produzia um índice vazio — e, como a idempotência
não via nada, cada passada duplicava tudo.

`assert_aim_persists()` escreve um run descartável e o lê de volta **antes** de indexar, e aborta com
esse diagnóstico; ligar `aim.enabled` no treino ao vivo imprime o mesmo aviso. Se um Aim futuro
corrigir o problema, a checagem passa e o caminho volta a funcionar sem mudança. Contexto: o último
release do Aim é de 2025-05-08 e a questão de manutenção (`aimhubio/aim#3414`) segue sem resposta.

## Notebooks: submeter e observar

O treino não roda mais no kernel. Cada notebook de experimento perdeu a célula do loop e ganhou:

```python
sweep_status(SWEEP_CONFIG)    # o que está pronto, o que falta, o que é retomável
submit_sweep(SWEEP_CONFIG)    # lança destacado e devolve na hora
watch_sweep(SWEEP_CONFIG)     # interromper para de OLHAR, não de treinar
```

Toda a Parte 1 (auditoria do split) e a Parte 3 (tabelas e figuras) seguem idênticas, lendo
`load_results()` da mesma CSV. A migração alterou nas CSVs **apenas** a coluna `output_dir`: a tabela
macro do X1 tem hash idêntico antes e depois.

A grade em si vive em `configs/sweep/<nome>.yaml` (os valores) e `src/sweeps/` (a lógica que depende
do regime). `tests/test_sweeps.py` afirma, contra o `hparams.yaml` de cada run já concluído, que a
configuração gerada hoje é a mesma que produziu aquela linha da CSV — 83 células conferidas. Sem
isso, mover a grade do notebook para o código poderia tornar os resultados novos incomparáveis com os
antigos em silêncio.

Exceções documentadas no teste, ambas estreitas: o `name` da grade `multiclass-all` mudou de propósito
(é a correção da colisão, e um teste separado afirma que **nada além do `name`** diverge), e um único
run do ACDC foi gravado antes de o template incluir o backbone.

## Embasamento

- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015 — configuration
  debt e glue code como as dívidas dominantes; a grade em config/código, não em constante de notebook.
- Rule et al., *Ten Simple Rules for Reproducible Research in Jupyter Notebooks*, PLOS Comp. Biol.
  2019 — o notebook como narrativa, a computação pesada fora dele.
- Pineau et al., *Improving Reproducibility in Machine Learning Research*, JMLR 2021 (checklist do
  NeurIPS); Gundersen & Kjensmo, AAAI 2018; Sandve et al., PLOS Comp. Biol. 2013 — o que `run.json`
  registra: seed, config resolvida, git sha, host.
- Zaharia et al., *Accelerating the ML Lifecycle with MLflow*, IEEE Data Eng. Bull. 2018 — o modelo
  run/params/métricas/artefatos; Isdahl & Gundersen, IEEE eScience 2019 — comparação de plataformas.
- Wilkinson et al., *FAIR Guiding Principles*, Scientific Data 2016 — identificador estável por run,
  com os artefatos localizáveis a partir dele.
