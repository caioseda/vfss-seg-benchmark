#!/usr/bin/env python3
'''
Roda sweeps em container nos servidores DeepLearning (ou na delaunay) e centraliza os logs na delaunay.
Tudo e disparado DA DELAUNAY. Hosts em configs/remote_hosts.yaml.

Uma vez (e de novo so quando o env `vfss` mudar):
    python scripts/remote.py lock                               # congela o env vfss em docker/*.txt
    python scripts/remote.py build                              # imagem na delaunay, a partir do lock
    python scripts/remote.py setup dl05 --datasets inca         # pastas, imagem e dados no host
    python scripts/remote.py setup dl01 --datasets inca unm     # o e2.yaml final tambem avalia no UNM

Treino inteiro num host:
    python scripts/remote.py submit dl05 configs/sweep/e2-cv.yaml
    python scripts/remote.py submit delaunay configs/sweep/e2-cv.yaml          # Docker na delaunay

Um sweep dividido (1-indexado; cada fold/seed inteiro fica numa GPU so):
    python scripts/remote.py submit dl01     configs/sweep/e2-cv.yaml --shard 1/3
    python scripts/remote.py submit dl05     configs/sweep/e2-cv.yaml --shard 2/3
    python scripts/remote.py submit delaunay configs/sweep/e2-cv.yaml --shard 3/3

Acompanhar e juntar:
    python scripts/remote.py status                             # containers, GPU e disco de cada host
    python scripts/remote.py logs dl05 configs/sweep/e2-cv.yaml # console ao vivo (Ctrl-C so para de olhar)
    python scripts/remote.py sync --family e2-cv --every 300    # em tmux: puxa os logs a cada 5 min
    python scripts/remote.py merge configs/sweep/e2-cv.yaml     # CSVs dos shards -> CSV canonica
    python scripts/index_runs.py --backend mlflow               # runs remotos no MLflow da delaunay
    tensorboard --logdir logs/e2-cv --host 127.0.0.1 --port 6006

`--dry-run` (antes do subcomando) imprime os comandos sem executar nada.
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shlex
import subprocess
import tempfile
import time

from src import remote
from src.remote import Runner, get_host, load_hosts

DATASET_ALIASES = {"inca": "dataset_inca_all_frames", "unm": "dataset_unm", "acdc": "ACDC"}


def _fail(message: str) -> int:
    print(f"erro: {message}", file=sys.stderr)
    return 1


def _relative_config(config_path: str) -> str:
    path = Path(config_path)
    path = path if path.is_absolute() else (PROJECT_ROOT / path)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config nao encontrada: {config_path}")
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        raise ValueError(f"a config precisa estar dentro do repositorio ({PROJECT_ROOT}): {path}")


def _remote_hosts(config: dict, names) -> list:
    if names:
        return [get_host(name, config) for name in names]
    return [get_host(name, config) for name, entry in config["hosts"].items() if not (entry or {}).get("local")]


# ---------------------------------------------------------------------- build / setup

def cmd_lock(args, runner: Runner, config: dict) -> int:
    '''
    Congela o env conda (`vfss`) em docker/conda-explicit.txt (URLs exatas) e docker/pip-requirements.txt
    (versoes exatas). A imagem instala isso, nao o environment.yml.
    '''
    import os
    import yaml

    conda = os.environ.get("CONDA_EXE") or str(Path.home() / "miniconda3" / "bin" / "conda")
    explicit = subprocess.check_output([conda, "list", "-n", args.env, "--explicit"], text=True)
    exported = yaml.safe_load(subprocess.check_output([conda, "env", "export", "-n", args.env, "--no-builds"],
                                                      text=True))
    pip_section = next((entry["pip"] for entry in exported["dependencies"] if isinstance(entry, dict)), [])
    prefix = Path(exported["prefix"])
    editable = subprocess.check_output([str(prefix / "bin" / "python"), "-m", "pip", "list", "--editable",
                                        "--format=freeze"], text=True).split()
    lines = remote.pip_lock_lines(pip_section, editable)
    header = (f"# Gerado por `python scripts/remote.py lock --env {args.env}` em {time.strftime('%Y-%m-%d')}. "
              "Nao edite a mao.\n")
    if runner.dry_run:
        print(f"conda: {explicit.count(chr(10))} linhas; pip: {len(lines)} pacotes; editaveis fora: {editable}")
        return 0
    remote.CONDA_LOCK.write_text(header + explicit)
    remote.PIP_LOCK.write_text(header + "\n".join(lines) + "\n")
    print(f"{remote.CONDA_LOCK.relative_to(PROJECT_ROOT)}: {sum(1 for l in explicit.splitlines() if l.startswith('http'))} pacotes conda")
    print(f"{remote.PIP_LOCK.relative_to(PROJECT_ROOT)}: {len(lines)} pacotes pip (editaveis fora: {editable})")
    print(f"nova tag da imagem: {remote.image_tag(config)}  ->  python scripts/remote.py build")
    return 0


def cmd_build(args, runner: Runner, config: dict) -> int:
    tag = remote.image_tag(config)
    print(f"imagem: {tag}")
    runner.local(["docker", "build", "-t", tag, "-"], input=remote.build_context())
    return 0


def _image_exists_locally(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def cmd_setup(args, runner: Runner, config: dict) -> int:
    host = get_host(args.host, config)
    tag = remote.image_tag(config)
    datasets = [DATASET_ALIASES.get(name, name) for name in args.datasets]

    if not runner.dry_run and not _image_exists_locally(tag):
        return _fail(f"a imagem {tag} nao existe na delaunay: rode  python scripts/remote.py build")

    runner.script(host, "set -e\numask 0002\nmkdir -p " + " ".join(shlex.quote(f"{host.root}/{sub}") for sub in
                  ("data", "code", "logs", "cache/home")) + "\n")

    if host.local:
        print("host local: dados e codigo sao os da delaunay; nada a copiar.")
    else:
        # Cota: o que ja esta no host + o que os datasets ainda vao ocupar.
        current = remote.remote_usage_gb(runner, host)
        missing = 0.0
        for dataset in datasets:
            local = remote.local_size_gb(Path(remote.data_path(config)) / dataset)
            already = 0.0
            if not runner.dry_run:
                out = runner.script(host, f"du -sk {shlex.quote(f'{host.root}/data/{dataset}')} 2>/dev/null "
                                    "| cut -f1 || echo 0\n", check=False, capture=True).stdout or "0"
                try:
                    already = int(out.strip().splitlines()[-1]) / 1024 ** 2
                except (ValueError, IndexError):
                    already = 0.0
            missing += max(local - already, 0.0)
        projected = current + missing
        print(f"disco em {host.root}: {current:.1f} GB hoje, ~{projected:.1f} GB depois "
              f"(limite {host.quota_gb:g} GB)")
        if projected > host.quota_gb and not args.force:
            return _fail("ultrapassaria o limite de disco. Use menos datasets ou --force.")

        # A MESMA imagem da delaunay (docker save | docker load), so se o host ainda nao a tiver.
        has_image = runner.script(host, f"docker image inspect {shlex.quote(tag)} >/dev/null 2>&1 "
                                  "&& echo yes || echo no\n", check=False, capture=True)
        if runner.dry_run or (has_image.stdout or "").strip() != "yes":
            print(f"enviando a imagem {tag} para {host.name} (alguns GB; uma vez por versao do ambiente)...")
            runner.local(["bash", "-o", "pipefail", "-c",
                          f"docker save {shlex.quote(tag)} | gzip -1 | ssh -o BatchMode=yes "
                          f"{shlex.quote(host.target)} 'gunzip | docker load'"])
        else:
            print(f"{host.name} ja tem a imagem {tag}.")

        for dataset in datasets:
            print(f"copiando {dataset} ...")
            runner.local(remote.rsync_data_command(host, dataset, config))

    # Teste de fumaca: o container enxerga exatamente 1 GPU e le os dados no caminho da delaunay.
    check = ("import torch, os; n = torch.cuda.device_count(); "
             "print('GPUs visiveis:', n, '|', torch.cuda.get_device_name(0) if n else '-'); "
             f"d = {remote.data_path(config)!r}; print('datasets:', sorted(os.listdir(d))); "
             "assert n == 1, 'esperava 1 GPU visivel'")
    script = remote.docker_run_script(host, "vfss-smoke", tag, ["python", "-c", check], tag=host.name, config=config)
    script = script.replace("docker run -d", "docker run --rm").replace(f"--name vfss-smoke", "")
    runner.script(host, script)
    if not host.local:
        runner.script(host, f"rm -rf {shlex.quote(host.root + '/code/vfss-smoke')}\n", check=False)
    return 0


# ---------------------------------------------------------------------- submit

def cmd_submit(args, runner: Runner, config: dict) -> int:
    host = get_host(args.host, config)
    config_rel = _relative_config(args.config)
    tag = args.tag or host.name
    # Um host remoto sempre roda isolado como shard (1/1 = grade inteira): o sync nunca traz arquivos
    # canonicos de familia, entao nada do que ja esta na delaunay pode ser sobrescrito.
    shard = args.shard or (None if host.local else "1/1")
    if shard:
        from src.sweep import parse_shard
        parse_shard(shard)
    name = remote.container_name(config_rel, tag)
    image = remote.image_tag(config)

    state = runner.script(host, remote.container_state_script(name), check=False, capture=True)
    status = (state.stdout or "").strip() or "absent"
    if status in {"running", "restarting", "paused"}:
        return _fail(f"o container {name} ja esta '{status}' em {host.name}. Ele pode estar treinando agora.\n"
                     f"  acompanhar: python scripts/remote.py logs {host.name} {name}\n"
                     f"  parar:      python scripts/remote.py stop {host.name} {name}")

    if remote.git_dirty():
        print("aviso: ha alteracoes nao commitadas -- o run.json gravara `dirty: true` e o retrato de codigo "
              "leva o working tree como esta.")

    if not host.local:
        needed = remote.referenced_datasets(PROJECT_ROOT / config_rel, config)
        if needed and not runner.dry_run:
            checks = " ".join(f"[ -d {shlex.quote(f'{host.root}/data/{d}')} ] || echo {shlex.quote(d)};" for d in needed)
            missing = (runner.script(host, checks + "\n", check=False, capture=True).stdout or "").split()
            if missing and not args.force:
                return _fail(f"{host.name} nao tem {missing}, que este sweep usa. Rode:\n"
                             f"  python scripts/remote.py setup {host.name} --datasets {' '.join(missing)}")
        usage = remote.remote_usage_gb(runner, host)
        if usage > host.quota_gb and not args.force:
            return _fail(f"{host.root} em {host.name} ja usa {usage:.1f} GB (> {host.quota_gb:g} GB).")
        if usage > 0.9 * host.quota_gb:
            print(f"aviso: {host.root} em {host.name} usa {usage:.1f} GB de {host.quota_gb:g} GB.")
        print(f"retrato do codigo -> {host.name}:{host.root}/code/{name}/")
        runner.script(host, f"umask 0002; mkdir -p {shlex.quote(f'{host.root}/code/{name}')}\n")
        runner.local(remote.rsync_code_command(host, name))
        for command in remote.rsync_deps_commands(host, name, config):
            runner.local(command)

    extra = []
    if args.fast_dev_run:
        extra.append("--fast-dev-run")
    if args.force_resume:
        extra.append("--force-resume")
    if args.sweep_dry_run:
        extra.append("--dry-run")
    command = remote.sweep_command(config_rel, shard, tag, overrides=args.set or (), extra=extra)
    labels = {"vfss.sweep": config_rel, "vfss.shard": shard or "-", "vfss.tag": tag}
    runner.script(host, remote.docker_run_script(host, name, image, command, tag, config=config, labels=labels))

    family_hint = Path(config_rel).stem
    print(f"\ncontainer '{name}' iniciado em {host.name} (GPU {host.gpu}, shard {shard or 'nenhum'}).")
    print(f"  ao vivo:   python scripts/remote.py logs {host.name} {config_rel}" + (f" --tag {tag}" if args.tag else ""))
    print(f"  estado:    python scripts/remote.py status {host.name}")
    if not host.local:
        print(f"  trazer:    python scripts/remote.py sync {host.name} --family <familia>  (ex.: {family_hint})")
    print("  relancar o MESMO comando depois de uma queda retoma de onde parou.")
    return 0


# ---------------------------------------------------------------------- sync / merge

def _sync_once(runner: Runner, hosts, families, checkpoints: bool) -> None:
    for host in hosts:
        print(f"[{time.strftime('%H:%M:%S')}] {host.name}: logs")
        done = runner.local(remote.rsync_logs_command(host, families), check=False)
        if done.returncode not in (0, 24):   # 24 = arquivos sumiram durante a copia (normal em treino)
            print(f"  aviso: rsync terminou com codigo {done.returncode}")
        if not checkpoints:
            continue
        dirs = remote.finished_checkpoint_dirs(PROJECT_ROOT / "logs", families, tag=host.name)
        if not dirs:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("\n".join(dirs) + "\n")
            listing = Path(handle.name)
        try:
            print(f"  checkpoints de {len(dirs)} runs concluidos")
            runner.local(remote.rsync_checkpoints_command(host, listing), check=False)
        finally:
            listing.unlink(missing_ok=True)


def cmd_sync(args, runner: Runner, config: dict) -> int:
    hosts = _remote_hosts(config, args.hosts)
    if args.every is None:
        _sync_once(runner, hosts, args.family, not args.no_checkpoints)
        return 0
    print(f"sincronizando a cada {args.every}s. Ctrl-C para parar (o treino nos hosts nao e afetado).")
    try:
        while True:
            _sync_once(runner, hosts, args.family, not args.no_checkpoints)
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nsync encerrado.")
    return 0


def cmd_merge(args, runner: Runner, config: dict) -> int:
    from src.sweep import Sweep, merge_shards

    target = args.target
    if target.endswith((".yaml", ".yml")):
        family_dir = Sweep(target).family_dir
    else:
        family_dir = PROJECT_ROOT / "logs" / target
    report = merge_shards(family_dir, dry_run=args.dry_run or runner.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["duplicates"]:
        print("aviso: run_id em mais de um shard -- ficou o do primeiro (ordem alfabetica do host).")
    if report["missing_output_dirs"]:
        print("aviso: output_dir ausente na delaunay -- rode `remote.py sync` antes do merge.")
    if not args.dry_run:
        print("proximo passo: python scripts/index_runs.py --backend mlflow")
    return 0


# ---------------------------------------------------------------------- acompanhar

def _resolve_name(host, target: str, tag) -> str:
    if target.endswith((".yaml", ".yml")):
        return remote.container_name(_relative_config(target), tag or host.name)
    return target


def cmd_status(args, runner: Runner, config: dict) -> int:
    names = args.hosts or list(config["hosts"])
    for name in names:
        host = get_host(name, config)
        runner.script(host, remote.status_script(host), check=False)
        print()
    return 0


def cmd_logs(args, runner: Runner, config: dict) -> int:
    host = get_host(args.host, config)
    name = _resolve_name(host, args.target, args.tag)
    follow = "" if args.no_follow else "-f "
    return runner.interactive(host, f"docker logs {follow}--tail {int(args.tail)} {shlex.quote(name)}")


def cmd_stop(args, runner: Runner, config: dict) -> int:
    host = get_host(args.host, config)
    name = _resolve_name(host, args.target, args.tag)
    runner.script(host, f"docker stop {shlex.quote(name)}\n", check=False)
    print("parado. Relancar o mesmo submit retoma de last.ckpt.")
    return 0


# ---------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="imprime os comandos sem executar")
    parser.add_argument("--hosts-file", default=str(remote.HOSTS_FILE))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lock", help="congela o env conda da delaunay em docker/*.txt")
    p.add_argument("--env", default="vfss")

    sub.add_parser("build", help="constroi a imagem Docker na delaunay")

    p = sub.add_parser("setup", help="prepara um host: pastas, imagem, dados e teste de GPU")
    p.add_argument("host")
    p.add_argument("--datasets", nargs="*", default=["inca"],
                   help=f"datasets a copiar (apelidos: {', '.join(DATASET_ALIASES)}); padrao: inca")
    p.add_argument("--force", action="store_true", help="ignora o limite de disco")

    p = sub.add_parser("submit", help="lanca um sweep (inteiro ou um shard) num container")
    p.add_argument("host")
    p.add_argument("config", help="configs/sweep/*.yaml")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="parte K de N, 1-indexado (1/2 e 2/2). Remoto sem --shard = 1/1 (grade inteira)")
    p.add_argument("--tag", default=None, help="nome do shard e do container (padrao: o nome do host)")
    p.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR", help="sobrescritas do sweep")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--force-resume", action="store_true")
    p.add_argument("--sweep-dry-run", action="store_true", help="o container so mostra a grade e sai")
    p.add_argument("--force", action="store_true", help="ignora as checagens de disco e de dados")

    p = sub.add_parser("sync", help="puxa os logs dos hosts para logs/ da delaunay")
    p.add_argument("hosts", nargs="*", help="padrao: todos os remotos")
    p.add_argument("--family", nargs="*", default=[], help="so estas familias (ex.: e2-cv)")
    p.add_argument("--every", type=int, default=None, metavar="SEG", help="repete a cada SEG segundos")
    p.add_argument("--no-checkpoints", action="store_true", help="nao traz os checkpoints dos runs concluidos")

    p = sub.add_parser("merge", help="consolida as CSVs dos shards na CSV canonica da familia")
    p.add_argument("target", help="configs/sweep/*.yaml ou o nome da familia")
    p.add_argument("--dry-run", action="store_true", dest="merge_dry_run")

    p = sub.add_parser("status", help="containers, GPU e disco de cada host")
    p.add_argument("hosts", nargs="*")

    for command, text in (("logs", "console do container ao vivo"), ("stop", "para o container")):
        p = sub.add_parser(command, help=text)
        p.add_argument("host")
        p.add_argument("target", help="configs/sweep/*.yaml ou o nome do container")
        p.add_argument("--tag", default=None)
        if command == "logs":
            p.add_argument("--tail", default=100, type=int)
            p.add_argument("--no-follow", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        args.dry_run = args.merge_dry_run or args.dry_run
    runner = Runner(dry_run=args.dry_run)
    config = load_hosts(Path(args.hosts_file))
    handler = {"lock": cmd_lock, "build": cmd_build, "setup": cmd_setup, "submit": cmd_submit, "sync": cmd_sync,
               "merge": cmd_merge, "status": cmd_status, "logs": cmd_logs, "stop": cmd_stop}[args.command]
    try:
        return handler(args, runner, config)
    except (FileNotFoundError, KeyError, ValueError) as error:
        return _fail(str(error))
    except subprocess.CalledProcessError as error:
        return _fail(f"comando falhou (codigo {error.returncode}): {error.cmd}")


if __name__ == "__main__":
    sys.exit(main())
