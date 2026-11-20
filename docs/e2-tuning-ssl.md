# E2: tuning dos pesos SSL (protocolo registrado antes de rodar)

Este protocolo foi decidido em 2026-11-20, **antes** de qualquer trial. Mudar algo nele depois de ver
resultados tem de ficar registrado aqui, com data e motivo.

## Por que ajustar

No E2 "padrão" (`configs/sweep/e2.yaml`), os métodos SSL rodaram com os pesos dos YAMLs base:

- **Mean Teacher e FixMatch:** `consistency_weight` = 0,1, herdado do X1.
- **MT temporal:** `temporal_weight` = 0,1.
- **DTC-TRUS:** os pesos 3/1/2/0,1 do **código** original.

Esses valores não são neutros:

- O FixMatch publicado usa λ_u = 1 (Sohn et al., 2020).
- O próprio artigo do DTC-TRUS usa 1/1/0,25/1 (arXiv 2606.31198), não os pesos do código. Os autores não
  dizem como os escolheram, não fazem análise de sensibilidade e não têm conjunto de validação.
- O DTC-TRUS colapsa para "tudo fundo" no E2. Com FG em cerca de 2,7% da imagem, o KD quase não pune o
  colapso, e os pesos foram calibrados para próstata em TRUS, uma estrutura grande.

Ajustar só o método que falhou criaria um orçamento de tuning desigual. Por isso, **todos os métodos SSL
são ajustados com o mesmo orçamento e o mesmo procedimento**, como em Oliver et al. (2018, *Realistic
Evaluation of Deep SSL*), no molde do tuning A.1 do E1.

## O que é ajustado e o que não é

| classe | exemplos | tratamento |
|---|---|---|
| (a) protocolo comum | lr, wd, passos, batch 8+8, augmentation fraca | Ajustado **uma vez** no E1, por arquitetura, e igual para todos. Reajustar por método confundiria método com otimizador. |
| (b) pesos das perdas sobre não rotulados | `consistency_weight`, `temporal_weight`, `kd_weight`, `con_weight`, `proto_weight` | **Ajustados.** Dão a força do sinal não supervisionado frente à âncora, que depende do domínio. |
| (c) constantes do mecanismo | EMA 0,99, τ 0,95, T 4, ruído do MT, percentil 70, rampas | **Fixas** nos valores publicados. Fazem parte do método, que os autores apresentam como independente do conjunto de dados. Ajustá-las aumentaria a dimensão da busca sobre uma validação de 11 pacientes. |

Por método:

- **Mean Teacher:** só `consistency_weight` (1-D).
- **FixMatch:** só λ_u (`consistency_weight`), com τ fixo (1-D).
- **MT temporal (`mt-temporal-prev`, o método base): tuning incremental.** O `consistency_weight` é
  **herdado** do vencedor da confirmação do Mean Teacher, e só o `temporal_weight` é ajustado (1-D). Assim
  a diferença MT temporal − MT isola o termo temporal. As variantes (`bi`, `prev3`, `proto-*`, `nomt`) não
  são ajustadas: herdam os pesos do `prev`. O `nomt` herda só o `temporal_weight`, porque o termo MT dele é
  0 por definição.
- **DTC-TRUS:** `kd_weight`, `con_weight` e `proto_weight` (3-D), com `seg_weight` = 3 como referência de
  escala. O AdamW é praticamente invariante à escala global da perda e o weight decay é desacoplado, então
  só as razões importam e fixar um peso não perde generalidade. T, o percentil e o PL por flip seguem o
  original, assim como a inicialização ImageNet do aluno e a ausência de GT na etapa 2.

**O orçamento é o mesmo para todos (25 trials), embora a dimensão varie de 1 a 3.** A medida de justiça é
o esforço computacional. Para mostrar que um ganho não veio de "mais botões", o artigo traz a curva da
melhor validação esperada em função do número de trials, por método (Dodge et al., 2019).

## Espaço de busca

A regra é a mesma para todos: log-uniforme de valor atual ÷10 a ×10 (2 décadas, como lr/wd no E1).

**Exceção declarada:** o teto do `proto_weight` sobe para 3, para conter o ponto do artigo.

| método | parâmetro | trial 0 (E2 padrão) | intervalo | 2º ponto fixo |
|---|---|---|---|---|
| Mean Teacher | `consistency_weight` | 0,1 | 0,01 a 1 | — |
| FixMatch | `consistency_weight` (λ_u) | 0,1 | 0,01 a 1 | 1,0 (Sohn et al.) |
| MT temporal | `temporal_weight` (`consistency_weight` herdado) | 0,1 | 0,01 a 1 | — |
| DTC-TRUS | `kd_weight` | 1,0 | 0,1 a 10 | 3,0 |
| | `con_weight` | 2,0 | 0,2 a 20 | 0,75 |
| | `proto_weight` | 0,1 | 0,01 a 3 | 3,0 |

O 2º ponto fixo do DTC-TRUS é o do artigo (1/1/0,25/1) reescalado para seg = 3. Os pontos fixos saem dos 25
trials, não se somam a eles.

### Os 10 primeiros trials (sorteio do TPE: seed 42 + n, `multivariate`)

| trial | MT λ | FixMatch λ_u | temporal | DTC kd / con / proto |
|---|---|---|---|---|
| 0 | 0,1 | 0,1 | 0,1 | 1 / 2 / 0,1 |
| 1 | 0,017 | **1** | 0,017 | **3 / 0,75 / 3** |
| 2 | 0,467 | 0,467 | 0,467 | 4,67 / 0,324 / 0,699 |
| 3 | 0,951 | 0,951 | 0,951 | 9,51 / 2,51 / 0,050 |
| 4 | 0,37 | 0,37 | 0,37 | 3,7 / 3,72 / 0,041 |
| 5 | 0,0169 | 0,0169 | 0,0169 | 0,169 / 17,8 / 0,639 |
| 6 | 0,0108 | 0,0108 | 0,0108 | 0,108 / 12,1 / 0,051 |
| 7 | 0,04 | 0,04 | 0,04 | 0,4 / 0,624 / 1,97 |
| 8 | 0,0975 | 0,0975 | 0,0975 | 0,975 / 0,572 / 0,043 |
| 9 | 0,225 | 0,225 | 0,225 | 2,25 / 0,246 / 0,071 |

Os pontos são os **mesmos múltiplos do valor atual** em todos os métodos, como entre as arquiteturas no
E1. Os trials 10 a 24 vêm do TPE e dependem dos resultados.

## Regime e seleção (espelho do E1)

| | E1 (A.1) | E2 SSL |
|---|---|---|
| treino | fold 0 do desenvolvimento | fold 0 + pool não rotulado **só** dos vídeos do fold 0 |
| seleção | `val/dice_patient`, validação fixa | igual |
| busca | TPE multivariado, seed 42, 10 de startup, MedianPruner (5 de startup, warm-up de 1000 passos), 25 × 4000 passos | igual |
| confirmação | top-3 não podadas × 8000 passos | igual |
| teste | nenhum | nenhum (nem o INCA, nem o UNM) |
| congelamento | `configs/frozen/e1-hparams.yaml` | `configs/frozen/e2-ssl-hparams.yaml` |

- **Referência e professor.** O `supervised` roda uma vez na busca, pelo cronograma completo (8000 passos),
  no fold 0. Ele é a referência supervisionada do regime de tuning e a etapa 1 do DTC-TRUS, a mesma em todos
  os trials da busca e da confirmação.
- **Desempate.** Vale o maior Dice médio por paciente na validação. A confirmação recusa uma busca
  incompleta, e o congelamento recusa qualquer método não confirmado.

## Como rodar

```bash
# Fase 1: referência + MT + FixMatch + DTC-TRUS. Pode ser dividida por método (o supervised vai junto do DTC-TRUS)
python scripts/run_sweep.py --config configs/sweep/e2-ssl-tuning-search.yaml --dry-run
scripts/submit.sh configs/sweep/e2-ssl-tuning-search.yaml
#   ou: python scripts/remote.py submit dl05 configs/sweep/e2-ssl-tuning-search.yaml --shard 1/3   (e 2/3, 3/3)
#   depois: python scripts/remote.py sync --family e2-ssl-tuning-search
#           python scripts/remote.py merge configs/sweep/e2-ssl-tuning-search.yaml

# Confirmação da fase 1 (lê a CSV canônica da busca: rodar onde ela está, a delaunay)
scripts/submit.sh configs/sweep/e2-ssl-tuning-confirm.yaml

# Fase 2: MT temporal, herdando o consistency_weight confirmado do MT
python scripts/run_sweep.py --config configs/sweep/e2-ssl-tuning-search.yaml --set "axes.method=[mt-temporal-prev]"
python scripts/run_sweep.py --config configs/sweep/e2-ssl-tuning-confirm.yaml --set "axes.method=[mt-temporal-prev]"

# Congelar (commitar o arquivo É o congelamento)
python scripts/freeze_e2_ssl_hparams.py --dry-run
python scripts/freeze_e2_ssl_hparams.py

# Linha "ajustado": famílias próprias, nada do e2/e2-cv é tocado
python scripts/remote.py submit dl05 configs/sweep/e2-cv-tuned.yaml
python scripts/remote.py submit dl01 configs/sweep/e2-tuned.yaml
```

**Ordem e onde rodar.** A confirmação e a fase 2 leem a CSV **canônica** da busca e o `state.json` dela
(professor do DTC-TRUS). Por isso rodam na delaunay, depois do `sync` e do `merge`. Só a busca da fase 1 é
distribuída.

**Custo estimado**, com cerca de 0,09 s por passo nos MT/FixMatch:

| etapa | sem pruning | dividida em 3 |
|---|---|---|
| fase 1, busca | cerca de 6 h | cerca de 3 h |
| confirmação da fase 1 | cerca de 1,5 h | — |
| fase 2, busca + confirmação | cerca de 3,5 h | — |
| `e2-tuned` + `e2-cv-tuned` | o mesmo custo do E2 e do E2-CV | — |

## O que vai para o artigo

- **Duas linhas por método:**
  - **"padrão":** e2/e2-cv, incluindo o colapso do DTC-TRUS com os pesos publicados no código;
  - **"ajustado":** e2-tuned/e2-cv-tuned.
- **A curva de orçamento por método** e o histórico de trials (`E2TuningSweep.trials_dataframe()`).
- **Análise de sensibilidade das constantes da classe (c)** (τ, EMA, T), se for feita: depois do
  congelamento, só nos valores congelados, no apêndice. **Nunca é usada para seleção.**

## Ameaças à validade (declarar)

1. **Seed única e validação pequena (11 pacientes).** Re-runs idênticos já diferiram até cerca de 0,10 de
   Dice no X1. Daí a grade grossa (2 décadas), a confirmação e a linha "padrão" ao lado da "ajustada".
2. **O fold 0 entra no tuning como treino.** No E2-CV ele também é fold de avaliação. Não há seleção sobre
   ele, e é a mesma limitação do E1-CV.
3. **A augmentation fraca entrou depois do colapso do MT** (e2_03). Foi uma decisão informada pelos
   resultados de treino e validação, aplicada a todos os métodos e anterior a qualquer teste.
4. **A dimensão difere entre métodos** (1 a 3, com orçamento igual). Ver a curva de orçamento.
5. **DTC-TRUS: o artigo e o código divergem.** O padrão é o do código. O artigo entra como ponto fixo da
   busca.
