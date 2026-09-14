# DiffRect e a reprodução no ACDC — decisões e prevenção de bugs

Registro das escolhas feitas ao portar o DiffRect (Liu, Li & Yuan, MICCAI 2024, arXiv:2407.09918)
para este repositório e ao montar a validação no ACDC. O critério para uma entrada estar aqui é:
**alguém lendo o código depois não conseguiria reconstruir a decisão a partir dele**, ou a decisão
protege contra um bug que roda sem erro.

Referências consultadas linha a linha: o artigo e o repositório oficial
[`CUHK-AIM-Group/DiffRect`](https://github.com/CUHK-AIM-Group/DiffRect) (MIT) —
`train_diffrect_ACDC.py`, `networks/unet_de.py`, `networks/unet.py`, `dataloaders/dataset.py`.

---

## Sumário das decisões

| # | Assunto | Decisão |
|---|---|---|
| 1 | Integração do ACDC | O dataset ACDC se adapta ao contrato do VFSS; código compartilhado quase não muda |
| 2 | Hiperparâmetros | X1/X1B constantes entre métodos; ACDC fiel à referência. São dois regimes separados |
| 3 | `semi_supervised_base` | `reference` (fiel) e `canonical` (isola o módulo de retificação), via flag |
| 4 | Backbone | Não é o mesmo do artigo. Rodamos os dois no ACDC como ablação |
| 5 | Dois `param_groups` | DiffRect treina duas redes; um otimizador só, um backward fundido |
| 6 | Difusão | Reimplementada (~80 linhas), não vendorizada (~50 KB) |
| 7 | Métricas | HD95/ASD sobre scipy, não medpy (GPL-3.0); avaliação volumétrica |
| 8 | Discrepâncias artigo × código | Onde discordam, o código vence e um teste fixa |

---

## 1. Como o ACDC entrou num pipeline escrito para VFSS

**Pergunta:** o código foi alterado, ou o dataset foi adaptado para simular a estrutura do VFSS?

**Resposta: o dataset se adaptou.** Foi a escolha deliberada, porque o notebook do ACDC só vale
como teste de aceitação se exercitar **o mesmo caminho de código** que o X1 vai rodar. Se o ACDC
tivesse um treinador próprio, ele validaria esse treinador e não o que roda no VFSS.

O que `ACDCSliceDataset` (`src/data/acdc_dataset.py`) imita do VFSS, e por quê:

| Contrato do VFSS | Onde é lido | O que o ACDC expõe |
|---|---|---|
| `dataset.video_frame_df` com `is_labeled` | `src/data/samplers.py:28` | DataFrame alinhado à ordem de indexação |
| `video_id`, `frame_id` inteiros | `src/data/samplers.py:37` | paciente e índice de fatia |
| `{'image','segmentation','metadata'}` | `LitWrapper.shared_step` | idem |
| linha não rotulada = máscara de zeros | `tests/test_ssl_methods.py:53` | idem, GT real em `hidden_segmentation` |

Consequência prática: `TwoStreamBatchSampler`, `DataModuleFromConfig`, `Experiment` e os quatro
wrappers SSL rodam no ACDC **sem uma linha de alteração**.

Duas armadilhas reais nesse contrato, ambas descobertas ao construir o dataset:

- `is_labeled` precisa ser `numpy.bool_`. `labeled_unlabeled_indices` faz `~is_labeled`; num array
  de objeto com `bool` do Python isso dá `-2`/`-1`, **ambos truthy**, e as duas listas de índices
  saem erradas sem erro nenhum. Fixado em `tests/test_acdc_dataset.py::test_is_labeled_is_numpy_bool`.
- `frame_metadata` é injetado sempre que a classe do sampler declara o parâmetro
  (`src/data/base.py:149`) — e `TwoStreamBatchSampler` declara. Então `video_id`/`frame_id` são
  obrigatórios **mesmo com `unlabeled_policy: global`**, que é o que o ACDC usa.

### O que mudou em código compartilhado (o mínimo possível)

| Arquivo | Mudança | Por quê |
|---|---|---|
| `src/models/base.py` | `optimizer_param_groups()` extraído | ponto de extensão para o retificador — ver §5 |
| `src/models/ssl/base.py` | `log_training()` e `budget_fraction_steps()` extraídos | DiffRect emite as mesmas chaves de log por construção, não por cópia |
| `src/data/augmentations.py` | `weak_augment_multi()` | DiffRect precisa de `segmentation` **e** `hidden_segmentation` sob o mesmo sorteio |
| `src/metrics.py` | métricas volumétricas | as 2-D não servem para ACDC — ver §7 |
| `src/third_party/ssl4mis/losses.py` | `oh_input` | Dice entre dois mapas duros, para a calibration guidance |

Nada disso muda o comportamento de X1/X1B: as adições são novas funções, e `weak_augment` virou um
wrapper de uma linha sobre `weak_augment_multi` com equivalência numérica fixada em teste.

### O que é exclusivo do ACDC

`src/data/acdc_dataset.py`, `scripts/prepare_acdc.py`, `src/evaluation_volume.py`,
`src.callbacks.VolumetricValidation`, `configs/experiment/acdc/` e `src/third_party/ssl4mis/unet.py`.

---

## 2. Hiperparâmetros: dois regimes com objetivos diferentes

**Pergunta:** X1 e X1B precisam de hiperparâmetros constantes para serem comparáveis, mas alguns
parâmetros dos métodos SSL fogem do original. Como isso foi tratado?

**Separando as duas perguntas em dois conjuntos de configs**, porque elas não são a mesma:

- **X1/X1B** perguntam *"qual método ajuda mais no VFSS?"*. Aí o que precisa ser constante é tudo
  menos o método. Os quatro configs de `configs/experiment/x1/` são idênticos exceto pelo bloco
  `lit_module`.
- **ACDC** pergunta *"a implementação está certa?"*. Aí o que precisa bater é a referência.

| | X1 / X1B | ACDC | Referência |
|---|---|---|---|
| otimizador | AdamW, lr 1e-3, wd 1e-4 | SGD, lr 0.01, momentum 0.9, wd 1e-4 | idem ACDC |
| scheduler | `CosineAnnealingLR(T_max=8000)` | `PolynomialLR(total_iters=30000, power=0.9)` | `lr*(1-it/max)**0.9` |
| passos | 8.000 | 30.000 | 30.000 |
| batch / rotulados | 8 / 4 | 6 / 3 | 6 / 3 |
| backbone | `src.models.unet.UNet` (17.26M) | ambos — ver §4 | `SSL4MISUNet` (1.81M) |
| seleção de modelo | `val/dice_score` (fatia) | `val/dice_volume` | volumétrico |
| política não rotulada | `same_video` | `global` | global |

O `PolynomialLR` do torch 2.4 telescopa exatamente para a fórmula da referência, então não há
scheduler custom.

### O cronograma de consistência é fração do orçamento, não passo absoluto

Esta é a decisão que mais silenciosamente daria errado se copiada literalmente. No SSL4MIS a rampa
é avaliada como `sigmoid_rampup(iter_num // 150, 200.0)` e `150 * 200 == 30000 == max_iterations`:
o divisor existe para a rampa completar no **último** passo. Copiar `150` e `200` para um run de
8.000 passos faria o peso parar em `exp(-5*(1-53/200)^2) ≈ 0.067` — 6.7% do configurado. Os métodos
semi-supervisionados seriam indistinguíveis do supervisionado **por aritmética, não por dados**.

Por isso `src/models/ssl/base.py` ancora tudo em fração do orçamento
(`REFERENCE_RAMPUP_STEPS / REFERENCE_TOTAL_STEPS = 1.0`, warm-up `1000/30000 ≈ 0.0333`), o que
preserva a *forma* do cronograma sob qualquer `max_steps`.

**Exceção conhecida, e é uma dívida honesta.** `rectification_start_fraction` usa a mesma mecânica,
mas o que ele controla é diferente: não é "equilibrar o termo não supervisionado", é *"o retificador
já aprendeu alguma coisa?"* — que é uma quantidade **absoluta**. Em 8.000 passos a retificação liga
no passo 267 com o retificador tendo treinado 267 passos, contra 1.000 na referência. Uma célula X1
fraca do DiffRect pode ser artefato disso. Está comentado no YAML, e o diagnóstico para distinguir
é o gap `train/rectified_dice` − `train/pseudo_dice` (§8).

---

## 3. `semi_supervised_base`: canônica × referência

**Pergunta:** quais as diferenças entre as duas implementações?

O `L_Semi` que o DiffRect publicou **não** é o FixMatch canônico. É a variante do SSL4MIS:

| | canônica (`FixMatchLitWrapper`) | referência (DiffRect) |
|---|---|---|
| confiança | `max(softmax) >= 0.95` | `(p - min)/max > 0.8` sobre o eixo de classes |
| pixels reprovados | descartados da loss | viram **classe 0 (fundo)** e treinam com peso cheio |
| loss | CE mascarada | CE + Dice |
| termo extra | — | complementary loss ponderada por entropia |
| perturbação forte | fotométrica + cutout | fotométrica, sem cutout |

Dois detalhes que parecem bug e não são, ambos documentados em `normalized_pseudo_label`:

1. A normalização divide por `max`, **não** por `(max - min)`. Logo o limiar `> 0.8` pergunta na
   verdade "a classe menos provável está abaixo de um quinto da mais provável?".
2. Pixels onde nenhuma classe passa do limiar são zerados e depois `argmax`-ados — voltam como
   fundo e **são supervisionados**. O FixMatch descartaria. Supervisionar fundo com confiança em
   pixel incerto é parte do que o método faz, não descuido a "limpar".

Ambas as variantes estão implementadas. `reference` é o default (necessário para o ACDC bater);
`canonical` existe porque com ela DiffRect e FixMatch passam a diferir em **exatamente uma coisa** —
o módulo de retificação — o que torna a célula X1 uma leitura limpa do módulo sozinho.

Mesma lógica em `supervise_weak_view`: a referência supervisiona a view fracamente aumentada,
enquanto os outros três métodos do X1 supervisionam a imagem crua. Default fiel (`True`), com a
flag disponível para isolar a comparação.

---

## 4. Backbone: **não** é o mesmo do artigo

**Pergunta:** o modelo base usado nos artigos é o mesmo daqui? Uma UNet?

São as duas UNets, mas **não a mesma UNet** — e a diferença é grande:

| | `src/third_party/ssl4mis/unet.py` | `src/models/unet.py` |
|---|---|---|
| larguras | 16, 32, 64, 128, 256 | 64, 128, 256, 512, 512 |
| **parâmetros** (1 canal, 4 classes) | **1.813.764** | **17.262.020** |
| dropout | 0.05 / 0.1 / 0.2 / 0.3 / 0.5 | nenhum |
| ativação | LeakyReLU | ReLU |

**9.5× mais parâmetros e nenhum dropout.** Num orçamento de 32 fatias rotuladas (1% do ACDC) isso
não é um detalhe: é outro experimento. Comparar o nosso número contra um publicado trocando o
backbone em silêncio não seria reprodução.

**Decisão: rodar os dois no ACDC.** `ssl4mis` é o teste de aceitação propriamente dito — arquiteturas
iguais, então uma diferença aponta para o port. `repo` mede quanto do resultado é arquitetura, e é
o backbone que o X1/X1B usa de fato. O veredito do notebook é calculado **só** sobre `ssl4mis`.

X1/X1B continuam na UNet do repo: lá a pergunta é comparar métodos sob arquitetura fixa.

> Nota de transcrição: o `Decoder` da referência lê `params['bilinear']` e **nunca** o repassa aos
> up-blocks, então todo upsampling é bilinear e o parâmetro é código morto. Transcrito como está —
> honrar a flag colocaria convoluções transpostas e levaria o modelo de 1.81M para 1.94M, ou seja,
> outra rede. Fixado em `tests/test_reference_equivalence.py::TestSSL4MISBackbone`.

---

## 5. Por que dois `param_groups`

**Pergunta:** por que passar dois `param_groups` ao otimizador? DiffRect treina duas redes?

**Sim, duas redes.** Além da UNet de segmentação, o DiffRect treina um **retificador**: um modelo de
difusão latente que aprende a corrigir pseudo-rótulos (18.48M parâmetros na configuração da
referência — maior que a própria rede de segmentação quando esta é a UNet do SSL4MIS). Na inferência
só a rede de segmentação roda; o retificador existe para produzir supervisão durante o treino.

`LitWrapper.configure_optimizers` só otimiza `self.model.parameters()` — e isso é **load-bearing**:
é exatamente assim que o teacher do Mean Teacher fica de fora do otimizador (fixado em
`tests/test_ssl_methods.py:179`). Sem sobrescrever, `self.rectifier` seria construído, forwardado,
salvo no checkpoint e **nunca treinado** — um bug que roda perfeitamente e produz curva plausível.
Daí o hook `optimizer_param_groups()`: grupo 0 = segmentação, grupo 1 = retificador, com lr própria
opcional.

### Por que um otimizador e um backward, e não otimização manual

A referência dá **três** passos de otimizador por iteração. Reproduzir isso no Lightning exigiria
`automatic_optimization = False`, e aí duas coisas quebram em silêncio (verificado no
`pytorch_lightning` 2.6.1 instalado):

- `loops/training_epoch_loop.py:114-117` — sob otimização manual `global_step` conta **passos de
  otimizador**, não batches. Com 3 passos/iteração, `max_steps: 8000` pararia em ~2.667 batches:
  DiffRect veria **um terço** das imagens dos outros métodos. Como o terceiro passo é condicional ao
  warm-up da retificação, o mapeamento ainda seria *por partes*. Isso sozinho invalidaria o X1.
- `loops/training_epoch_loop.py:475` — `if not ... automatic_optimization: return`, **sem aviso**. O
  `CosineAnnealingLR` do YAML seria ignorado e DiffRect treinaria a lr constante enquanto os outros
  recozem até 1e-6.

Além disso, otimização manual proíbe `gradient_clip_val` e `accumulate_grad_batches`
(`configuration_validator.py:121-133`), e um `training_step` que retorna `None` sai de
`tests/test_learning.py`.

**Saída:** os três backwards da referência já são **disjuntos em gradiente** — todo tensor que cruza
entre as duas redes passa por `argmax`, que é barreira não diferenciável, além de estar detached.
Logo `loss = seg + refine + rect; loss.backward()` com um otimizador de dois grupos produz **os
mesmos gradientes**, preservando orçamento de passos, scheduler e acumulação.

Divergência declarada: muda a *ordem de atualização* (na referência o passo 3 re-forwarda a rede de
segmentação com pesos já atualizados pelo passo 1). Está no docstring do módulo.

---

## 6. Difusão reimplementada, não vendorizada

O retificador depende do `guided_diffusion` da OpenAI (~50 KB), do qual o caminho de código
efetivamente alcançado são ~80 linhas: schedule cosseno, `q_sample`, e amostragem DDIM com
parametrização `START_X`. `src/third_party/diffrect/diffusion.py` reimplementa esse subconjunto com
cabeçalho de procedência, seguindo o precedente de `src/third_party/ssl4mis/`.

`ModelVarType.FIXED_LARGE` + `LossType.MSE` significam que a cabeça de variância nunca é lida —
dito no docstring para que a omissão seja visivelmente deliberada, e não um esquecimento.

---

## 7. Métricas: volumétricas, e sem medpy

**Avaliação volumétrica.** Os números publicados de ACDC são por **volume**, não por fatia. Dice
sobre uma pilha de D fatias faz a média de D escores; uma fatia basal onde o ventrículo direito
simplesmente não existe pontua 0 e puxa a média, enquanto o Dice volumétrico nunca vê aquela fatia
como observação separada. HD95 e ASD são piores: distância entre *superfícies*, e a superfície por
fatia é um contorno, não uma casca.

Medido na prática neste projeto: no supervisionado @10%, no mesmo checkpoint,
`val/dice_volume` = 0.8678 contra `val/dice_score` (fatia) = 0.7892 — **7.9 pontos**. Quem comparar
um número deste repositório com um da literatura de ACDC precisa saber qual dos dois está olhando.

Daí `src/evaluation_volume.py` e o callback `VolumetricValidation`, que loga `val/dice_volume` para
a seleção de checkpoint bater com a referência. Foi feito como *callback* de propósito: `Experiment`
anexa o `ModelCheckpoint` **depois** dos callbacks do config, e este lê `trainer.callback_metrics`
em `on_validation_end` — então `monitor: val/dice_volume` resolve sem tocar em `LitWrapper`,
`DataModuleFromConfig` ou `Experiment`.

**Sem medpy.** A referência calcula `dc`/`jc`/`hd95` com `medpy`, que é GPL-3.0. Este repositório já
recusou dependência de licença restritiva antes — `sigmoid_rampup` foi reimplementado a partir do
artigo para não puxar a CC BY-NC do `ramps.py` do SSL4MIS. Então HD95/ASD foram reimplementados
sobre `scipy.ndimage` (BSD). Consequência declarada no notebook: Dice e Jaccard são operações de
conjunto exatas e devem bater; HD95/ASD são *comparáveis mas não bit-idênticos*.

**Container de dados.** `.npz` por volume em vez de `.h5` por fatia: h5py não estava no ambiente, e
um handle HDF5 aberto no `__init__` e lido de workers forkados é fonte clássica de corrupção
silenciosa. O *conteúdo* (float32 normalizado min-max por volume, label uint8, indexação por fatia)
é transcrito do `acdc_data_processing.py`.

---

## 8. Onde artigo e código discordam, o código vence

Três discrepâncias, todas fixadas em teste:

1. **O retificador vê a imagem.** O artigo sugere que a retificação opera só sobre o rótulo, e
   `docs/hipoteses_experimentos.md` afirmava isso. O código refuta: `UNet_LDMV2.forward` concatena a
   imagem à máscara colorida e ainda soma features multiescala de um **segundo** encoder sobre a
   imagem crua. Só a U-Net de denoising interna é puramente latente. Vira a flag
   `condition_on_image` (default `True`). **Isso afeta H4, H6 e X7** — o doc foi corrigido.
2. **A calibration guidance é a Dice _loss_, não o Dice score.** A Eq. 6 diz `τ = Dice(y_s, y_w)`;
   o código faz `dice_loss(...) * 999`, ou seja `1 - dice`. Direção invertida.
3. **A CG é o timestep.** Não é um embedding auxiliar: ela entra como o passo de tempo sinusoidal da
   rede externa. O acoplamento vive num lugar só, dentro de `TimestepEmbedding`.

---

## 9. O que foi feito para prevenir bugs

O princípio: **todo bug que importa aqui roda perfeitamente e produz uma curva de loss plausível.**
Um teste de fumaça não pega nenhum deles.

### Bugs reais encontrados durante a construção

| Bug | Como apareceu | Teste que o fixa |
|---|---|---|
| Dataset ACDC **sem augmentação** | supervisionado @10% com train Dice 0.987 contra val 0.696 **e caindo**; ~15 pontos abaixo do publicado | `test_acdc_dataset.py::test_training_split_is_augmented_and_validation_is_not` |
| `ACDCVolumeDataset` ignorava `image_channels` e pulava `[0,1]→[-1,1]` | crash de shape no `fast_dev_run`; a parte da normalização não teria dado erro nenhum | `test_volume_layout_and_range_match_the_slice_dataset` |
| Mirror do HuggingFace **incompleto** | `snapshot_download` retornou sucesso tendo pulado 2 arquivos: 199 imagens e 199 máscaras que não pareavam. Contar arquivos não pegaria — os totais batiam | `_verify_download` (verifica o **pareamento**, não a contagem) |
| SSL4MIS **renumera** os frames | `frame01`/`frame02` nas listas contra `frame01`/`frame12` no ACDC cru | assert em `_canonical_cases`, lido do `Info.cfg` |

Sobre o último: inverter ED/ES não falharia, porque os dois frames de um paciente sempre caem no
mesmo split (verificado: 0 de 100 divididos). O que mudaria em silêncio é `train_slices.list`, cujas
primeiras N linhas **são** o subconjunto rotulado de 1%/5%/10%.

### Invariantes fixados

- **Isolamento de supervisão** — rótulos escondidos nunca são supervisionados. Testado por
  envenenamento: as linhas não rotuladas recebem lixo e uma implementação correta não percebe.
- **Disjunção de gradiente** — `refine_loss.backward()` deixa todo `.grad` da rede de segmentação em
  None/zero e vice-versa. **É este par que licencia o backward fundido**, e sem ele a equivalência
  seria afirmada em vez de verificável.
- **Igualdade ponta a ponta** — com o peso de consistência em 0 e a retificação desligada, DiffRect
  treina a rede de segmentação **bit-idêntica** ao `SupervisedLitWrapper`. É uma igualdade exata,
  sem limiar para calibrar.
- **O retificador realmente treina** — a LFR é disjunta, então *nenhuma métrica de segmentação diz
  se ela aprendeu*. Um retificador congelado deixaria todo o resto verde. Fixado sobre a MSE latente.
- **Paridade numérica com a referência** — `get_comp_loss`, a regra de pseudo-rótulo min-max, os
  coeficientes de `q_sample`, a direção da CG e a contagem de parâmetros do backbone, todos contra
  trechos transcritos.

### Disciplina de mutação

Cada teste novo foi validado injetando o bug que ele deveria pegar. Resultados:

| Mutação | Resultado |
|---|---|
| `refine_sup` supervisiona `logits_weak` (vazamento real) | **pega** por dois testes independentes |
| `pred_x_start` detached (LFR nunca aprende) | **pega** — e os dois testes pré-existentes do retificador deixavam passar |
| augmentação desligada | **pega** |
| label rotacionada diferente da imagem | **pega** |
| volume sem o mapeamento `[0,1]→[-1,1]` | **pega** (2 testes) |
| `pl_weak` sem `.detach()` | **sobrevive** — mutante equivalente: `normalized_pseudo_label` já faz `argmax`, que é barreira não diferenciável, então o detach externo é redundante |

O último é informativo: mostra que o desenho é à prova de vazamento **por construção** (a barreira
do argmax), e não apenas por disciplina de `detach()`.

### Guardas fora da suíte de testes

Verificados num run curto de verdade, porque nenhum teste unitário os alcança:

- `trainer.global_step == batches vistos` (40 == 40) — guarda direta contra o hazard do `global_step`
- a lr **anda** (0.010000 → 0.009988 em 40 passos) — guarda contra o scheduler silenciosamente pulado
- 2 `param_groups` com 56 e 202 tensores — retificador de fato no otimizador

### Custo medido

RTX 6000 Ada, batch 6 a 256×256, backbone do repo: supervisionado 0.045 s/passo, Mean Teacher 0.051,
FixMatch 0.076, **DiffRect 0.243 (5.4×)** — 2.0 h para 30k iterações, contra as "~2 GPU hours" numa
4090 declaradas no artigo. A concordância é corroboração independente de que a implementação está na
escala certa.

---

## 10. O que ainda não está respondido

- A grade do ACDC está rodando. **O port só está aceito quando o `diffrect` com backbone `ssl4mis`
  cair dentro de 2.0 pontos de Dice do publicado.** Até lá, nenhuma conclusão do X1 sobre o DiffRect
  é interpretável — um retificador quebrado produz só uma célula um pouco pior, indistinguível de
  "DiffRect não ajuda no VFSS".
- O warm-up da retificação como fração do orçamento (§2) é uma dívida conhecida, não uma escolha
  defendida.
- Os baselines CPS, ICT, MCNetV2 e INCL da Tabela 1 não estão implementados; entram no notebook só
  como referência de escala.
