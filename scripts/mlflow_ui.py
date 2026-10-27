#!/usr/bin/env python3
'''
Sobe a interface do MLflow pronta para ser acessada por um tunel.

    python scripts/mlflow_ui.py                       # porta 5000, sobre logs/_tracking/mlflow.db
    python scripts/mlflow_ui.py --port 5050
    python scripts/mlflow_ui.py --upstream http://127.0.0.1:5000   # so o proxy, sobre um mlflow ja no ar

Sobe o `mlflow ui` numa porta interna e coloca na frente um proxy que COMPRIME as respostas e
reescreve `Origin`/`Referer` -- ver `src/mlflow_proxy.py` para o porque. So escuta em 127.0.0.1: o
acesso remoto e pelo tunel.

    # no SEU computador, depois de subir isto no servidor:
    ssh -L 5000:127.0.0.1:5000 usuario@servidor
    # e abra  http://localhost:5000
'''
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_proxy import build_app
from src.tracking import resolve_tracking_config


def wait_healthy(url: str, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def print_links(store_uri: str, public_url: str) -> None:
    '''
    Um link por experimento, ja na visao "Model training".

    O MLflow 3.x abre em "GenAI" (traces/avaliacao), que esta VAZIA para estes experimentos -- os
    runs de treino ficam em "Model training". A escolha vale por navegador, mas o link direto evita
    cair na tela vazia na primeira visita.
    '''
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=store_uri)
        experiments = [e for e in client.search_experiments() if e.name not in {"Default", "MLflow Demo"}]
    except Exception as error:
        print(f"(nao consegui listar os experimentos: {type(error).__name__})")
        return
    print("\nExperimentos (visao 'Model training', onde estao os runs):")
    for experiment in sorted(experiments, key=lambda e: e.name):
        print(f"  {experiment.name:38} {public_url}/#/experiments/{experiment.experiment_id}"
              f"/runs?workflowType=machine_learning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5000, help="porta publica (a do tunel)")
    parser.add_argument("--internal-port", type=int, default=None,
                        help="porta do mlflow ui por tras do proxy (padrao: --port + 1)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--logs", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--upstream", default=None,
                        help="URL de um mlflow ja rodando; nao sobe um novo, so o proxy")
    args = parser.parse_args()

    store_uri = resolve_tracking_config(Path(args.logs))["mlflow"]["tracking_uri"]
    child = None
    if args.upstream:
        upstream = args.upstream.rstrip("/")
    else:
        internal_port = args.internal_port or args.port + 1
        upstream = f"http://127.0.0.1:{internal_port}"
        mlflow_bin = shutil.which("mlflow") or str(Path(sys.executable).with_name("mlflow"))
        print(f"mlflow ui em {upstream}  (store: {store_uri})")
        # Absoluto de proposito: `sqlite:///logs/...` relativo resolve contra o CWD de quem chama, e
        # apontaria para um banco novo e vazio se o script fosse lancado de outro diretorio.
        child = subprocess.Popen([mlflow_bin, "ui", "--host", "127.0.0.1", "--port", str(internal_port),
                                  "--backend-store-uri", store_uri],
                                 env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})

    try:
        if not wait_healthy(upstream):
            print(f"O mlflow nao respondeu em {upstream}/health em 3 min.", file=sys.stderr)
            return 1
        public_url = f"http://localhost:{args.port}"
        print(f"\nProxy (comprime + reescreve Origin) em http://{args.host}:{args.port}")
        print(f"No seu computador:  ssh -L {args.port}:127.0.0.1:{args.port} usuario@servidor")
        print(f"e abra:             {public_url}")
        print_links(store_uri, public_url)
        sys.stdout.flush()

        import uvicorn
        uvicorn.run(build_app(upstream), host=args.host, port=args.port, log_level="warning")
    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
