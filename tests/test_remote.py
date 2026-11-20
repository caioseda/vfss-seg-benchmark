'''
Os comandos que `scripts/remote.py` monta para lancar e sincronizar sweeps em container.

Nada aqui fala com SSH nem com Docker: o que se testa e o TEXTO dos comandos, porque e nele que moram
as garantias que importam --
  - o container grava com o uid de quem lanca, avaliado NO HOST (senao arquivos de root que voce nao move);
  - so a GPU configurada fica visivel;
  - os caminhos da delaunay sao reproduzidos (YAMLs e output_dir validos sem mudanca);
  - todo diretorio montado e criado antes do docker run;
  - o sync nunca traz arquivo canonico de familia nem checkpoint em andamento;
  - um host remoto sempre roda como shard.

Roda com:  python -m unittest tests.test_remote -v
'''
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src import remote

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOSTS = {
    "defaults": {"user": "fulano", "root": "/local/{user}/vfss", "gpu": 1, "shm_size": "16g", "quota_gb": 200},
    "hosts": {"dl05": {"ssh": "Deeplearning05"},
              "delaunay": {"local": True, "gpu": 0, "root": "/scratch/{user}/vfss-docker"}},
    "paths": {"data": "/data_ssd/caioseda/data"},
    "python_paths": ["/home/fulano/projetos/lib/src"],
    "image": "vfss-seg",
}


class Hosts(unittest.TestCase):
    def test_defaults_and_user_substitution(self):
        host = remote.get_host("dl05", HOSTS)
        self.assertEqual(host.root, "/local/fulano/vfss")
        self.assertEqual(host.gpu, 1)
        self.assertEqual(host.target, "fulano@Deeplearning05")
        self.assertFalse(host.local)
        self.assertEqual(host.remote_path("/x"), "fulano@Deeplearning05:/x")

    def test_local_host(self):
        host = remote.get_host("delaunay", HOSTS)
        self.assertTrue(host.local)
        self.assertEqual(host.gpu, 0)
        self.assertEqual(host.remote_path("/x"), "/x")

    def test_unknown_host(self):
        with self.assertRaises(KeyError):
            remote.get_host("dl99", HOSTS)

    def test_repository_hosts_file_parses(self):
        config = remote.load_hosts()
        for name in config["hosts"]:
            remote.get_host(name, config)
        self.assertTrue(remote.image_tag(config).startswith(f"{config['image']}:env-"))


class DockerRun(unittest.TestCase):
    def script(self, host_name="dl05", command=("python", "scripts/run_sweep.py")):
        host = remote.get_host(host_name, HOSTS)
        name = remote.container_name("configs/sweep/e2-cv.yaml", host.name)
        return host, name, remote.docker_run_script(host, name, "vfss-seg:env-x", list(command), host.name, HOSTS)

    def test_runs_as_invoking_user_evaluated_on_host(self):
        _, _, script = self.script()
        # Literal: e o bash DO HOST que expande, nao o python da delaunay.
        self.assertIn('--user "$(id -u):$(id -g)"', script)
        self.assertIn("umask 0002", script)
        self.assertIn("HOME=/cache/home", script)

    def test_only_configured_gpu(self):
        self.assertIn("--gpus device=1", self.script()[2])
        self.assertIn("--gpus device=0", self.script("delaunay")[2])

    def test_reproduces_delaunay_paths(self):
        host, name, script = self.script()
        project = str(PROJECT_ROOT)
        self.assertIn(f"/local/fulano/vfss/code/{name}:{project}:rw", script)
        self.assertIn(f"/local/fulano/vfss/logs:{project}/logs:rw", script)
        self.assertIn("/local/fulano/vfss/data:/data_ssd/caioseda/data:ro", script)
        self.assertIn(f"-w {project}", script)

    def test_every_mount_source_created_before_run(self):
        for host_name in ("dl05", "delaunay"):
            host, name, script = self.script(host_name)
            mkdir_line = next(line for line in script.splitlines() if line.startswith("mkdir -p"))
            self.assertLess(script.index("mkdir -p"), script.index("docker run"))
            for source, _, _ in remote.mounts(host, name, HOSTS):
                if source.startswith(str(PROJECT_ROOT)) or source in {"/data_ssd/caioseda/data",
                                                                       "/home/fulano/projetos/lib/src"}:
                    continue   # na delaunay: os reais, que ja existem
                self.assertIn(source, mkdir_line, msg=f"{host_name}: {source}")
            if not host.local:
                self.assertIn(f"/code/{name}/logs", mkdir_line)   # ponto de montagem aninhado

    def test_tracking_off_only_on_remote(self):
        self.assertIn("VFSS_TRACKING=off", self.script("dl05")[2])
        self.assertNotIn("VFSS_TRACKING", self.script("delaunay")[2])

    def test_local_mounts_real_repo(self):
        host, name, _ = self.script("delaunay")
        sources = {source for source, _, _ in remote.mounts(host, name, HOSTS)}
        self.assertIn(str(PROJECT_ROOT), sources)
        self.assertFalse(any("/code/" in source for source in sources))

    def test_editable_packages_mounted_at_same_path(self):
        for host_name in ("dl05", "delaunay"):
            host, name, script = self.script(host_name)
            self.assertIn("PYTHONPATH=/home/fulano/projetos/lib/src", script)
            targets = {target: (source, mode) for source, target, mode in remote.mounts(host, name, HOSTS)}
            source, mode = targets["/home/fulano/projetos/lib/src"]
            self.assertEqual(mode, "ro")
            if host.local:
                self.assertEqual(source, "/home/fulano/projetos/lib/src")
            else:
                self.assertEqual(source, f"/local/fulano/vfss/code/{name}.deps/0")
                (command,) = remote.rsync_deps_commands(host, name, HOSTS)
                self.assertEqual(command[-1], f"fulano@Deeplearning05:/local/fulano/vfss/code/{name}.deps/0/")
                # o rsync so cria o ultimo nivel: `<name>.deps/` tem de ser criado antes
                self.assertIn(f"/local/fulano/vfss/code/{name}.deps/0", remote.snapshot_dirs(host, name, HOSTS))

    def test_container_name_is_docker_safe(self):
        self.assertEqual(remote.container_name("configs/sweep/e2-cv.yaml", "dl05"), "vfss-e2-cv-dl05")
        self.assertRegex(remote.container_name("a b/c@d.yaml", "x/y"), r"^[a-zA-Z0-9][a-zA-Z0-9_.-]+$")


class SweepCommand(unittest.TestCase):
    def test_shard_and_overrides(self):
        command = remote.sweep_command("configs/sweep/e2-cv.yaml", "2/3", "dl05", ["a.b=1"], ["--fast-dev-run"])
        self.assertEqual(command[:4], ["python", "scripts/run_sweep.py", "--config", "configs/sweep/e2-cv.yaml"])
        self.assertIn("--shard", command)
        self.assertEqual(command[command.index("--shard") + 1], "2/3")
        self.assertEqual(command[command.index("--shard-tag") + 1], "dl05")
        self.assertEqual(command[-2:], ["--set", "a.b=1"])

    def test_remote_submit_defaults_to_whole_grid_as_shard(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as hosts:
            json.dump(HOSTS, hosts)
            hosts.flush()
            out = io.StringIO()
            import scripts.remote as cli
            with contextlib.redirect_stdout(out):
                code = cli.main(["--dry-run", "--hosts-file", hosts.name, "submit", "dl05",
                                 "configs/sweep/e2-cv.yaml"])
            self.assertEqual(code, 0)
            self.assertIn("--shard 1/1 --shard-tag dl05", out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["--dry-run", "--hosts-file", hosts.name, "submit", "delaunay", "configs/sweep/e2-cv.yaml"])
            self.assertNotIn("--shard", out.getvalue().split("docker run")[1])


class Sync(unittest.TestCase):
    def test_logs_never_bring_canonical_files_or_checkpoints(self):
        command = remote.rsync_logs_command(remote.get_host("dl05", HOSTS))
        self.assertNotIn("--delete", command)
        for pattern in ("*.ckpt", "/*/results_*.csv", "/*/_sweep/state.json", "/_tracking/"):
            self.assertIn(f"--exclude={pattern}", command)
        self.assertEqual(command[-1], f"{PROJECT_ROOT}/logs/")

    def test_family_filter_after_guards(self):
        command = remote.rsync_logs_command(remote.get_host("dl05", HOSTS), ["e2-cv"])
        include = command.index("--include=/e2-cv/***")
        self.assertLess(command.index("--exclude=*.ckpt"), include)
        self.assertEqual(command[include + 1], "--exclude=*")

    def test_finished_checkpoints_only_done_runs_of_that_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            run = PROJECT_ROOT / "logs" / "e2-cv" / "runs" / "r1" / "uid"
            for tag, cells in {
                "dl05": {"a": {"status": "done", "run_dir": str(run)},
                         "b": {"status": "running", "run_dir": str(run.parent / "uid2")}},
                "dl01": {"c": {"status": "done", "run_dir": str(run.parent / "uid3")}},
            }.items():
                shard = logs / "e2-cv" / "_sweep" / tag
                shard.mkdir(parents=True)
                (shard / "state.json").write_text(json.dumps({"cells": cells}))
            found = remote.finished_checkpoint_dirs(logs, ["e2-cv"], tag="dl05")
            self.assertEqual(found, ["e2-cv/runs/r1/uid/checkpoints/"])


class Datasets(unittest.TestCase):
    def test_final_e2_needs_unm_cv_does_not(self):
        config = remote.load_hosts()
        final = remote.referenced_datasets(PROJECT_ROOT / "configs/sweep/e2.yaml", config)
        cv = remote.referenced_datasets(PROJECT_ROOT / "configs/sweep/e2-cv.yaml", config)
        self.assertIn("dataset_unm", final)
        self.assertIn("dataset_inca_all_frames", final)
        self.assertEqual(cv, ["dataset_inca_all_frames"])


if __name__ == "__main__":
    unittest.main()
