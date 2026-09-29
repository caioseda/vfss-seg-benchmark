#!/usr/bin/env bash
#
# Lanca um sweep numa sessao tmux destacada, para que o treino sobreviva a queda do SSH.
#
#   scripts/submit.sh configs/sweep/x1.yaml                      # nome da sessao = x1
#   scripts/submit.sh configs/sweep/x1.yaml minha-run
#   scripts/submit.sh configs/sweep/x1.yaml conferir --dry-run   # argumentos extras vao ao runner
#
#   tmux ls                    # o que esta rodando
#   tmux attach -t x1          # acompanhar (Ctrl-b d para sair sem matar)
#   tail -f logs/x1/_sweep/console.log
#
# Relancar apos uma queda e seguro: celulas concluidas sao puladas e a interrompida retoma de
# `last.ckpt`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${VFSS_CONDA_ENV:-vfss}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
TMUX_BIN="${TMUX_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/tmux}"

if [ $# -lt 1 ]; then
    echo "uso: $0 <configs/sweep/*.yaml> [nome-da-sessao]" >&2
    exit 2
fi

SWEEP_CONFIG="$1"
SESSION="${2:-$(basename "$SWEEP_CONFIG" .yaml)}"
shift $(( $# > 1 ? 2 : 1 ))
EXTRA_ARGS=("$@")

[ -f "$REPO_ROOT/$SWEEP_CONFIG" ] || [ -f "$SWEEP_CONFIG" ] || {
    echo "config nao encontrada: $SWEEP_CONFIG" >&2; exit 1; }
command -v "$TMUX_BIN" >/dev/null 2>&1 || {
    echo "tmux nao encontrado em $TMUX_BIN" >&2
    echo "instale com: conda install -n $CONDA_ENV -c conda-forge tmux" >&2; exit 1; }

if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    echo "a sessao '$SESSION' ja existe -- ela pode estar treinando agora." >&2
    echo "  acompanhar: $TMUX_BIN attach -t $SESSION" >&2
    echo "  encerrar:   $TMUX_BIN kill-session -t $SESSION" >&2
    exit 1
fi

# Duas coisas obrigatorias aqui, ambas descobertas na pratica:
#
#   `cd "$REPO_ROOT"` dentro do comando, e nao apenas o `-c` do tmux: `conda run` nao preserva o
#   diretorio de trabalho, e o painel acaba em `/` -- o script nao e encontrado e o `--config`
#   relativo nao resolve.
#
#   `--no-capture-output`: sem ele o `conda run` bufferiza a saida, e o console.log so aparece
#   quando o sweep termina -- inutil justamente enquanto se quer acompanhar.
EXTRA_QUOTED=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    printf -v EXTRA_QUOTED '%q ' "${EXTRA_ARGS[@]}"
fi
"$TMUX_BIN" new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "cd $(printf '%q' "$REPO_ROOT") && \
     $CONDA_BASE/bin/conda run --no-capture-output -n $CONDA_ENV \
     python scripts/run_sweep.py --config $(printf '%q' "$SWEEP_CONFIG") ${EXTRA_QUOTED:-}; \
     echo; echo '--- sweep encerrado. Ctrl-b d para sair, ou exit para fechar a sessao. ---'; \
     exec bash"

echo "sessao tmux '$SESSION' iniciada."
echo "  acompanhar: $TMUX_BIN attach -t $SESSION"
echo "  ou:         tail -f logs/<familia>/_sweep/console.log"
