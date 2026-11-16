'''
Lanca sweeps em containers -- nos servidores DeepLearning por SSH, ou na propria delaunay -- e traz os
logs de volta para a delaunay, que continua sendo a copia central (notebooks, TensorBoard, MLflow).

O desenho, em uma frase por peca:

  - a delaunay ORQUESTRA: copia codigo e dados, da a partida (`docker run -d`) e PUXA os logs. Os hosts
    remotos nunca precisam enxergar a delaunay;
  - o container reproduz os caminhos da delaunay (dados em `paths.data`, repositorio em PROJECT_ROOT),
    entao nenhum YAML muda e os caminhos absolutos gravados (output_dir, best_checkpoint_path) continuam
    validos depois do sync;
  - o container roda com o uid/gid de quem o lanca, e todo diretorio montado e criado ANTES do
    `docker run` -- senao o Docker o cria como root e voce nao consegue mover nem apagar o que ele grava;
  - um host remoto sempre roda como shard (no minimo `1/1`), isolado em `logs/<familia>/_sweep/<tag>/`.
    O sync nunca traz arquivos canonicos da familia, e `merge` os consolida explicitamente;
  - cada submit leva um retrato do codigo para `code/<container>/`: mexer no repositorio da delaunay (ou
    lancar outro sweep) nao altera um sweep que ja esta rodando.

A logica que monta comandos e pura (testada em tests/test_remote.py); `scripts/remote.py` e o CLI.
'''
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOSTS_FILE = PROJECT_ROOT / "configs" / "remote_hosts.yaml"
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"
ENVIRONMENT_FILE = PROJECT_ROOT / "environment.yml"
# O lock do env `vfss` da delaunay (scripts/remote.py lock): e ele, e nao o environment.yml, que a imagem
# instala -- o environment.yml quase nao pina versoes, e resolve-lo de novo trouxe outro Lightning, numpy,
# pandas, omegaconf e torchmetrics.
CONDA_LOCK = PROJECT_ROOT / "docker" / "conda-explicit.txt"
PIP_LOCK = PROJECT_ROOT / "docker" / "pip-requirements.txt"
# opencv-python precisa de libGL, que nem o container nem um servidor sem tela tem; a variante headless
# da MESMA versao e o mesmo codigo sem o modulo de GUI.
PIP_LOCK_SUBSTITUTIONS = {"opencv-python": "opencv-python-headless"}
CONTAINER_CACHE = "/cache"

# O que NAO vai no retrato de codigo. `logs/` e montado a parte; o resto e lixo ou peso morto.
CODE_EXCLUDES = ["/logs/", "__pycache__/", "*.pyc", ".ipynb_checkpoints/", "*.ckpt", "/lightning_logs/",
                 "/mlruns/", "/nohup.out", "/output.png", "*.tmp"]

# O que o sync periodico NUNCA traz: checkpoints (pesados e reescritos durante o treino -- os dos runs
# concluidos vem numa passada propria) e, por defesa, qualquer arquivo canonico de familia.
SYNC_EXCLUDES = ["*.ckpt", "*.tmp", "/_tracking/", "/*/results_*.csv", "/*/_sweep/state.json",
                 "/*/_sweep/console.log", "/*/_sweep/configs/"]


# ---------------------------------------------------------------------- hosts

@dataclass
class Host:
    name: str
    ssh: Optional[str]
    user: str
    root: str
    gpu: int
    shm_size: str
    quota_gb: float
    local: bool

    @property
    def target(self) -> str:
        '''Destino de ssh/rsync (`usuario@host`).'''
        return f"{self.user}@{self.ssh}"

    def remote_path(self, path: str) -> str:
        '''Caminho no formato do rsync: local na delaunay, `usuario@host:caminho` num remoto.'''
        return path if self.local else f"{self.target}:{path}"


def load_hosts(path: Path = HOSTS_FILE) -> Dict[str, object]:
    return yaml.safe_load(Path(path).read_text())


def get_host(name: str, config: Optional[dict] = None) -> Host:
    config = config or load_hosts()
    hosts = config.get("hosts", {})
    if name not in hosts:
        raise KeyError(f"host desconhecido: {name!r}. Definidos em {HOSTS_FILE.name}: {sorted(hosts)}")
    entry = {**(config.get("defaults") or {}), **(hosts[name] or {})}
    user = entry.get("user") or os.environ.get("USER") or "user"
    local = bool(entry.get("local", False))
    if not local and not entry.get("ssh"):
        raise ValueError(f"host {name!r} precisa de `ssh:` (ou `local: true`).")
    return Host(name=name, ssh=entry.get("ssh"), user=user,
                root=str(entry["root"]).format(user=user), gpu=int(entry.get("gpu", 0)),
                shm_size=str(entry.get("shm_size", "16g")), quota_gb=float(entry.get("quota_gb", 200)),
                local=local)


def data_path(config: Optional[dict] = None) -> str:
    return str((config or load_hosts())["paths"]["data"]).rstrip("/")


def python_paths(config: Optional[dict] = None) -> List[str]:
    '''Pacotes locais em modo editavel (ex.: vfss_data_split): montados no mesmo caminho + PYTHONPATH.'''
    return [str(path).rstrip("/") for path in ((config or load_hosts()).get("python_paths") or [])]


def deps_dir(host: "Host", name: str, index: int) -> str:
    '''Copia de um `python_paths` no host, ao lado do retrato do codigo (fora dele: o rsync usa --delete).'''
    return f"{host.root}/code/{name}.deps/{index}"


# ---------------------------------------------------------------------- imagem

def image_tag(config: Optional[dict] = None) -> str:
    '''`vfss-seg:env-<hash>` -- muda sozinha quando o lock do ambiente ou o Dockerfile mudam.'''
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in (DOCKERFILE, CONDA_LOCK, PIP_LOCK))
                            ).hexdigest()[:10]
    return f"{(config or load_hosts()).get('image', 'vfss-seg')}:env-{digest}"


def build_context() -> bytes:
    '''Contexto minimo (so os dois arquivos): o repositorio tem >150 GB de logs que o Docker enviaria.'''
    buffer = io.BytesIO()
    # USTAR: o `docker build -` nao reconhece o formato PAX (padrao do python) como contexto e o le como
    # se fosse o proprio Dockerfile.
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in (DOCKERFILE, CONDA_LOCK, PIP_LOCK):
            tar.add(str(path), arcname=path.name)
    return buffer.getvalue()


def pip_lock_lines(exported_pip: Sequence[str], editable: Sequence[str]) -> List[str]:
    '''
    A secao `pip:` do `conda env export`, sem os pacotes editaveis (vem por `python_paths`) e com as
    substituicoes de PIP_LOCK_SUBSTITUTIONS.
    '''
    skip = {name.split("==")[0].lower().replace("_", "-") for name in editable}
    lines = []
    for entry in exported_pip:
        name, _, version = str(entry).partition("==")
        key = name.lower().replace("_", "-")
        if key in skip:
            continue
        name = PIP_LOCK_SUBSTITUTIONS.get(key, name)
        lines.append(f"{name}=={version}" if version else name)
    return lines


# ---------------------------------------------------------------------- nomes e caminhos

def container_name(config_path: str, tag: str) -> str:
    stem = Path(config_path).stem
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"vfss-{stem}-{tag}")


def mounts(host: Host, name: str, config: Optional[dict] = None) -> List[tuple]:
    '''
    `(origem no host, destino no container, modo)`. Num remoto o repositorio vem do retrato
    `code/<container>` e os logs de `logs/`; na delaunay, os reais.
    '''
    data = data_path(config)
    project = str(PROJECT_ROOT)
    extra = python_paths(config)
    if host.local:
        return [(project, project, "rw"), (data, data, "ro"), (f"{host.root}/cache", CONTAINER_CACHE, "rw")] + \
            [(path, path, "ro") for path in extra]
    return [
        (f"{host.root}/code/{name}", project, "rw"),
        (f"{host.root}/logs", f"{project}/logs", "rw"),
        (f"{host.root}/data", data, "ro"),
        (f"{host.root}/cache", CONTAINER_CACHE, "rw"),
    ] + [(deps_dir(host, name, index), path, "ro") for index, path in enumerate(extra)]


def dirs_to_create(host: Host, name: str, config: Optional[dict] = None) -> List[str]:
    '''
    Todo diretorio montado, e os de cache, criados com o uid do usuario ANTES do docker run. O ponto de
    montagem `logs/` dentro do retrato tambem: senao o Docker o criaria como root dentro dele.
    '''
    real = {str(PROJECT_ROOT), data_path(config), *python_paths(config)}
    created = [source for source, _, _ in mounts(host, name, config)
               if not source.startswith(str(PROJECT_ROOT)) and source not in real]
    created += [f"{host.root}/cache/{sub}" for sub in ("home", "torch", "huggingface", "xdg", "matplotlib")]
    if not host.local:
        created.append(f"{host.root}/code/{name}/logs")
    return created


# ---------------------------------------------------------------------- comandos

def sweep_command(config_path: str, shard: Optional[str], tag: Optional[str],
                  overrides: Sequence[str] = (), extra: Sequence[str] = ()) -> List[str]:
    command = ["python", "scripts/run_sweep.py", "--config", config_path]
    if shard:
        command += ["--shard", shard, "--shard-tag", tag]
    command += list(extra)
    if overrides:
        command += ["--set", *overrides]
    return command


def docker_run_script(host: Host, name: str, image: str, command: Sequence[str], tag: str,
                      config: Optional[dict] = None, labels: Optional[Dict[str, str]] = None) -> str:
    '''
    Script bash que roda NO HOST (via `ssh ... bash -s`). `$(id -u)`/`$(id -g)` sao avaliados la:
    o container grava com o seu uid naquele host, e voce pode ler, mover e apagar o que ele salvar.
    '''
    q = shlex.quote
    env = {
        "HOME": f"{CONTAINER_CACHE}/home",
        "TORCH_HOME": f"{CONTAINER_CACHE}/torch",
        "HF_HOME": f"{CONTAINER_CACHE}/huggingface",
        "XDG_CACHE_HOME": f"{CONTAINER_CACHE}/xdg",
        "MPLCONFIGDIR": f"{CONTAINER_CACHE}/matplotlib",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VFSS_SHARD_TAG": tag,
    }
    if python_paths(config):
        env["PYTHONPATH"] = ":".join(python_paths(config))
    if not host.local:
        # O mlflow.db (sqlite) e da delaunay; os runs remotos entram nele por `index_runs.py` apos o sync.
        env["VFSS_TRACKING"] = "off"
    lines = [
        "set -euo pipefail",
        "umask 0002",
        f"if ! docker image inspect {q(image)} >/dev/null 2>&1; then",
        f"  echo 'imagem {image} ausente em '$(hostname)': rode  python scripts/remote.py setup {host.name}' >&2",
        "  exit 3",
        "fi",
        "mkdir -p " + " ".join(q(path) for path in dirs_to_create(host, name, config)),
        f"docker rm {q(name)} >/dev/null 2>&1 || true",
    ]
    run = ["docker run -d", f"--name {q(name)}", "--restart no",
           f"--gpus device={host.gpu}", '--user "$(id -u):$(id -g)"', f"--shm-size {q(host.shm_size)}",
           f"--hostname {q(tag)}", f"-w {q(str(PROJECT_ROOT))}", '-e USER="$(id -un)"']
    run += [f"-e {key}={q(value)}" for key, value in env.items()]
    run += [f"--label {q(f'{key}={value}')}" for key, value in (labels or {}).items()]
    run += [f"-v {q(f'{source}:{target}:{mode}')}" for source, target, mode in mounts(host, name, config)]
    run += [q(image), " ".join(q(part) for part in command)]
    lines.append(" \\\n  ".join(run))
    return "\n".join(lines) + "\n"


def container_state_script(name: str) -> str:
    '''Imprime `running`, `exited`, ... ou `absent`.'''
    q = shlex.quote
    return (f"docker inspect -f '{{{{.State.Status}}}}' {q(name)} 2>/dev/null || echo absent\n")


def rsync_code_command(host: Host, name: str) -> List[str]:
    command = ["rsync", "-a", "--delete"]
    command += [f"--exclude={pattern}" for pattern in CODE_EXCLUDES]
    command += [f"{PROJECT_ROOT}/", host.remote_path(f"{host.root}/code/{name}/")]
    return command


def rsync_deps_commands(host: Host, name: str, config: Optional[dict] = None) -> List[List[str]]:
    return [["rsync", "-a", "--delete", "--exclude=__pycache__/", f"{path}/",
             host.remote_path(f"{deps_dir(host, name, index)}/")]
            for index, path in enumerate(python_paths(config))]


def rsync_data_command(host: Host, dataset: str, config: Optional[dict] = None) -> List[str]:
    return ["rsync", "-a", "--info=progress2", f"{data_path(config)}/{dataset}/",
            host.remote_path(f"{host.root}/data/{dataset}/")]


def rsync_logs_command(host: Host, families: Sequence[str] = ()) -> List[str]:
    '''
    Puxa `<root>/logs/` do host para `logs/` da delaunay, sem `--delete` (nada local e apagado) e sem
    checkpoints. Com `families`, so essas familias.
    '''
    command = ["rsync", "-a"] + [f"--exclude={pattern}" for pattern in SYNC_EXCLUDES]
    if families:
        command += [f"--include=/{family}/***" for family in families] + ["--exclude=*"]
    command += [host.remote_path(f"{host.root}/logs/"), f"{PROJECT_ROOT}/logs/"]
    return command


def finished_checkpoint_dirs(logs_root: Path, families: Sequence[str] = (), tag: Optional[str] = None) -> List[str]:
    '''
    `checkpoints/` dos runs CONCLUIDOS segundo os `state.json` de shard ja sincronizados, relativos a
    `logs/`. So eles: o best de um run em andamento ainda muda de nome, e traze-lo deixaria copias velhas.
    '''
    logs_root = Path(logs_root)
    found = []
    family_dirs = [logs_root / family for family in families] if families else \
        [path for path in logs_root.iterdir() if path.is_dir()]
    for family_dir in family_dirs:
        for state_path in sorted((family_dir / "_sweep").glob(f"{tag or '*'}/state.json")):
            try:
                cells = json.loads(state_path.read_text()).get("cells", {})
            except Exception:
                continue
            for entry in cells.values():
                run_dir = entry.get("run_dir")
                if entry.get("status") != "done" or not run_dir:
                    continue
                try:
                    relative = Path(run_dir).relative_to(PROJECT_ROOT / "logs")
                except ValueError:
                    continue
                found.append(f"{relative}/checkpoints/")
    return sorted(set(found))


def rsync_checkpoints_command(host: Host, files_from: Path) -> List[str]:
    return ["rsync", "-a", "-r", f"--files-from={files_from}", "--exclude=last.ckpt",
            host.remote_path(f"{host.root}/logs/"), f"{PROJECT_ROOT}/logs/"]


def status_script(host: Host) -> str:
    q = shlex.quote
    return "\n".join([
        "echo \"== $(hostname)  GPU " + str(host.gpu) + " ==\"",
        "docker ps -a --filter name=vfss- --format 'table {{.Names}}\\t{{.Status}}\\t{{.RunningFor}}' || true",
        f"nvidia-smi -i {host.gpu} --query-gpu=name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader || true",
        f"if [ -d {q(host.root)} ]; then echo \"disco em {host.root}: $(du -sh {q(host.root)} 2>/dev/null | cut -f1)"
        f" (limite {host.quota_gb:g}G)\"; df -h {q(host.root)} | tail -1; fi",
    ]) + "\n"


def disk_usage_gb_script(host: Host) -> str:
    return f"du -sk {shlex.quote(host.root)} 2>/dev/null | cut -f1 || echo 0\n"


# ---------------------------------------------------------------------- execucao

class Runner:
    '''Executa (ou so imprime, com dry_run) comandos locais e scripts num host.'''

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def _show(self, text: str) -> None:
        print(f"$ {text}", flush=True)

    def local(self, command: Sequence[str], check: bool = True, input: Optional[bytes] = None,
              capture: bool = False) -> subprocess.CompletedProcess:
        self._show(" ".join(shlex.quote(part) for part in command))
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, stdout="" if capture else None)
        return subprocess.run(list(command), check=check, input=input, text=capture and input is None,
                              capture_output=capture)

    def script(self, host: Host, script: str, check: bool = True, capture: bool = False,
               tty: bool = False) -> subprocess.CompletedProcess:
        '''Roda um script bash no host: localmente na delaunay, por `ssh ... bash -s` num remoto.'''
        command = ["bash", "-s"] if host.local else \
            ["ssh", "-o", "BatchMode=yes", *(["-t"] if tty else []), host.target, "bash", "-s"]
        where = "local" if host.local else host.target
        self._show(f"[{where}] " + script.strip().replace("\n", "\n    "))
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, stdout="" if capture else None)
        return subprocess.run(command, input=script, text=True, check=check, capture_output=capture)

    def interactive(self, host: Host, remote_command: str) -> int:
        '''Comando com saida ao vivo (docker logs -f). Ctrl-C para de OLHAR, nao para o treino.'''
        command = ["bash", "-c", remote_command] if host.local else \
            ["ssh", "-t", host.target, remote_command]
        self._show(" ".join(shlex.quote(part) for part in command))
        if self.dry_run:
            return 0
        try:
            return subprocess.call(command)
        except KeyboardInterrupt:
            print("\nParei de ACOMPANHAR. O container segue treinando.")
            return 0


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                      cwd=str(PROJECT_ROOT), text=True)
        return bool(out.strip())
    except Exception:
        return False


def referenced_datasets(config_path: Path, config: Optional[dict] = None) -> List[str]:
    '''
    Datasets (subdiretorios de `paths.data`) citados pelo YAML do sweep e pelos YAMLs que ele referencia.
    Serve para recusar um submit antes de o container falhar no meio por falta de dados (ex.: o e2.yaml
    final avalia no UNM, o e2-cv nao).
    '''
    data = data_path(config)
    pattern = re.compile(re.escape(data) + r"/([^/\s'\"]+)")
    sweep = yaml.safe_load(Path(config_path).read_text()) or {}
    texts = [Path(config_path).read_text()]
    for key in ("base_configs", "arch_configs"):
        for referenced in (sweep.get(key) or {}).values():
            path = PROJECT_ROOT / str(referenced)
            if path.is_file():
                texts.append(path.read_text())
    return sorted({match.group(1) for text in texts for match in pattern.finditer(text)})


def remote_usage_gb(runner: Runner, host: Host) -> float:
    if runner.dry_run:
        return 0.0
    out = runner.script(host, disk_usage_gb_script(host), check=False, capture=True).stdout or "0"
    try:
        return int(out.strip().splitlines()[-1]) / 1024 ** 2
    except (ValueError, IndexError):
        return 0.0


def local_size_gb(path: Path) -> float:
    out = subprocess.check_output(["du", "-sk", str(path)], text=True)
    return int(out.split()[0]) / 1024 ** 2
