'''
Bateria de diagnóstico do colapso da validação nos métodos MT do E2.

    python scripts/e2_diag.py make                 # (re)gera configs/experiment/e2-diag/*.yaml
    python scripts/e2_diag.py run                  # roda todas as runs, na ordem de RUNS
    python scripts/e2_diag.py run D0 D1 D5         # só algumas
    python scripts/e2_diag.py run --smoke D0       # 200 passos, val a cada 100 (checagem de chaves)
    python scripts/e2_diag.py run --seed 43 D0 D5  # outra seed (saída em .../s43)

Cada run é o config do E2 (DeepLabV3+, seed 42, 8000 passos, 8+8) com UMA mudança (tabela abaixo) e
a mesma instrumentação:
  - `MTEvalProbe` (src/callbacks.py): student/teacher em eval, teacher com BN recalibrado, etc.;
  - `PatientValidation` extra com `forward_attr: forward` -> `val_student/dice_patient`;
  - snapshots persistentes a cada 1000 passos (`trainer.snapshots`, src/experiment.py);
  - SEM split de teste: diagnóstico não olha o teste INCA.

Saídas em logs/e2-diag/runs/<name>/s42/. Analisadas por
notebooks/artigo/e2_ssl_temporal/e2_03_colapso_mt.ipynb.
'''

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

E2_DIR = PROJECT_ROOT / "configs" / "experiment" / "e2"
OUT_DIR = PROJECT_ROOT / "configs" / "experiment" / "e2-diag"
FAMILY = "e2-diag"
DEFAULT_SEED = 42
SNAPSHOT_EVERY = 1000

# id -> (nome, config base do E2, overrides em `lit_module.params`, overrides extras, o que testa)
RUNS = {
    "D0": ("mt-probe", "e2-meanteacher.yaml", {}, {},
           "reproduz o colapso com a sonda ligada"),
    "D1": ("mt-lambda0", "e2-meanteacher.yaml", {"consistency_weight": 0.0}, {},
           "supervisionado + teacher EMA: colapsa sem consistência? (se sim, não é o lambda)"),
    "D5": ("mt-aug", "e2-meanteacher.yaml", {"student_augment": "weak"}, {},
           "augmentation (flip + afim leve) no student e no teacher evita o colapso?"),
    "D6": ("mt-aug-reg", "e2-meanteacher.yaml",
           {"student_augment": "weak", "consistency_weight": 0.03, "ema_decay": 0.999}, {},
           "pacote anti-overfitting: augmentation + lambda baixo + teacher mais lento"),
    "D7": ("sup-aug", "e2-supervised.yaml", {"student_augment": "weak"}, {},
           "controle: ganho da augmentation sem MT"),
    "D2": ("mt-ramp-half", "e2-meanteacher.yaml", {"consistency_rampup_fraction": 0.5}, {},
           "se o lambda causa o colapso, ele chega mais cedo"),
    "D3": ("mt-lambda1", "e2-meanteacher.yaml", {"consistency_weight": 1.0}, {},
           "dose-resposta: lambda 10x acelera/aprofunda o colapso?"),
    "D4": ("mt-constlr", "e2-meanteacher.yaml", {},
           {"lr_scheduler": {"params": {"T_max": 1_000_000_000}}},
           "lr ~constante: o colapso segue o passo/LR em vez do lambda?"),
    # Correções SEM augmentation: as variantes temporais recebem fluxo pré-computado do dataset, e
    # aplicar geometria exigiria transformar também o campo vetorial.
    "D8": ("mt-lambda003", "e2-meanteacher.yaml", {"consistency_weight": 0.03}, {},
           "lambda 0,03 sem augmentation basta para evitar o colapso?"),
    "D9": ("mt-ema0999", "e2-meanteacher.yaml", {"ema_decay": 0.999}, {},
           "teacher mais lento (menos acoplado ao student) sem augmentation evita o colapso?"),
}


def build_config(run_id: str, seed: int = DEFAULT_SEED):
    name, base_file, lit_overrides, extra_overrides, purpose = RUNS[run_id]
    config = OmegaConf.load(E2_DIR / base_file)
    config.name = f"e2diag-{run_id}-{name}"
    # O diagnóstico parte do E2 SEM augmentation e sem a validação extra do student que o E2 passou a
    # ter (2026-11-16): cada run muda só o que a tabela diz, e D0 continua reproduzindo o colapso.
    config.lit_module.params.pop("student_augment", None)
    config.trainer.callbacks = [c for c in config.trainer.callbacks
                                if c.get("params", {}).get("prefix") != "val_student"]

    for key, value in lit_overrides.items():
        config.lit_module.params[key] = value
    if extra_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(extra_overrides))

    has_teacher = "MeanTeacher" in str(config.lit_module.target)
    callbacks = list(config.trainer.callbacks)
    if has_teacher:
        callbacks.append(OmegaConf.create({
            "target": "src.callbacks.PatientValidation",
            "params": {"num_classes": 2, "class_ids": {1: "FG"}, "prefix": "val_student",
                       "forward_attr": "forward"},
        }))
        callbacks.append(OmegaConf.create({
            "target": "src.callbacks.MTEvalProbe",
            "params": {"num_classes": 2, "class_ids": {1: "FG"}, "recal_batches": 32,
                       "train_eval_batches": 8, "unlabeled_probe_batches": 4},
        }))
    config.trainer.callbacks = callbacks
    config.trainer.snapshots = {"every_n_train_steps": SNAPSHOT_EVERY}

    # Diagnóstico não olha o teste INCA (o protocolo só o abre no treino final).
    del config.data.params["test"]
    config.data.params.cache_splits = ["validation"]
    # A seed também muda a ordem dos lotes; rótulos (label_seed) e pool de não rotulados ficam fixos.
    config.data.params.train_batch_sampler.params.seed = seed
    return config, purpose


def make() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_id in RUNS:
        config, purpose = build_config(run_id)
        path = OUT_DIR / f"{config.name}.yaml"
        header = (f"# GERADO por scripts/e2_diag.py -- não editar à mão.\n"
                  f"# {run_id}: {purpose}\n")
        path.write_text(header + OmegaConf.to_yaml(config))
        print(f"escrito {path.relative_to(PROJECT_ROOT)}")


def run(run_ids, smoke: bool, seed: int = DEFAULT_SEED) -> None:
    from src.experiment import Experiment
    make()
    for run_id in run_ids:
        config, purpose = build_config(run_id, seed)
        path = OUT_DIR / f"{config.name}-s{seed}.yaml"
        if seed != DEFAULT_SEED:
            path.write_text(f"# GERADO por scripts/e2_diag.py (seed {seed}) -- não editar à mão.\n" + OmegaConf.to_yaml(config))
        else:
            path = OUT_DIR / f"{config.name}.yaml"
        overrides = None
        run_uid = f"s{seed}"
        if smoke:
            overrides = {"trainer": {"params": {"max_steps": 200, "val_check_interval": 100},
                                     "snapshots": {"every_n_train_steps": 100}},
                         "lr_scheduler": {"params": {"T_max": 200}}}
            run_uid = f"{run_uid}-smoke"
        run_json = PROJECT_ROOT / "logs" / FAMILY / "runs" / config.name / run_uid / "run.json"
        if run_json.exists() and json.loads(run_json.read_text()).get("status") == "done":
            print(f"\n===== {run_id} {config.name}: já concluída, pulando ({run_json.parent})")
            continue
        # Uma run interrompida (status "running"/"failed") retoma de checkpoints/last.ckpt.
        print(f"\n===== {run_id} {config.name}: {purpose}")
        experiment = Experiment(config_path=str(path), overrides=overrides,
                                log_dir=str(PROJECT_ROOT / "logs"), family=FAMILY, run_uid=run_uid)
        experiment.run(seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["make", "run"])
    parser.add_argument("runs", nargs="*", help=f"ids ({', '.join(RUNS)}); vazio = todas")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed (run_uid = s<seed>)")
    parser.add_argument("--smoke", action="store_true", help="200 passos, para checar as chaves logadas")
    args = parser.parse_args()
    unknown = set(args.runs) - set(RUNS)
    if unknown:
        parser.error(f"ids desconhecidos: {sorted(unknown)}")
    if args.command == "make":
        make()
    else:
        run(args.runs or list(RUNS), smoke=args.smoke, seed=args.seed)
