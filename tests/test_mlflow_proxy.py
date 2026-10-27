'''
O proxy do `mlflow ui` comprime e reescreve Origin -- e nao estraga o resto.

Dois sintomas reais motivam cada teste (medidos contra o MLflow 3.16.1):

  - o servidor nao comprime: a pagina de runs baixa ~8.5 MB, e por um tunel lento isso e o
    "carregando para sempre";
  - um POST com `Origin` que nao seja localhost leva 403 "Cross-origin request blocked", e
    `runs/search` e `experiments/search` sao POSTs: num tunel publico (https://...) a lista de
    experimentos carrega (GET) e a tabela de runs fica vazia.

O servidor falso abaixo REPRODUZ esse bloqueio, entao um teste pode provar que e o proxy -- e nao o
acaso -- que o contorna: o mesmo POST direto no upstream leva 403.

Roda com:  python -m unittest tests.test_mlflow_proxy -v
'''

import gzip
import json
import socket
import threading
import time
import unittest

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from src.mlflow_proxy import build_app

BIG_JS = ("function f(){return 'conteudo repetitivo de um bundle de interface';}\n" * 40_000).encode()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Served:
    '''Um app ASGI em um thread, com a porta que ele realmente ocupa.'''

    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level="error", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("servidor de teste nao subiu")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def make_fake_mlflow(port_holder: dict) -> Starlette:
    async def big(request: Request):
        return Response(BIG_JS, media_type="application/javascript",
                        headers={"etag": '"abc"', "cache-control": "public, max-age=2419200"})

    async def echo(request: Request):
        origin = request.headers.get("origin")
        # O mesmo criterio do MLflow: POST so se o Origin for o do proprio servidor.
        if origin and origin != f"http://127.0.0.1:{port_holder['port']}":
            return PlainTextResponse("Cross-origin request blocked", status_code=403)
        return JSONResponse({"origin": origin, "referer": request.headers.get("referer"),
                             "host": request.headers.get("host"),
                             "encoding": request.headers.get("accept-encoding"),
                             "body": (await request.body()).decode()})

    async def events(request: Request):
        async def gen():
            for index in range(3):
                yield f"data: evento {index}\n\n".encode()
                time.sleep(0.4)       # bloqueante de proposito: simula um produtor lento
        return StreamingResponse(gen(), media_type="text/event-stream")

    async def missing(request: Request):
        return PlainTextResponse("nao existe", status_code=404, headers={"x-marca": "preservada"})

    return Starlette(routes=[
        Route("/big.js", big), Route("/echo", echo, methods=["GET", "POST"]),
        Route("/events", events), Route("/missing", missing),
    ])


class ProxyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        holder = {}
        cls.upstream = Served(make_fake_mlflow(holder))
        holder["port"] = cls.upstream.port          # conhecida so apos escolher a porta
        cls.upstream.__enter__()
        cls.proxy = Served(build_app(cls.upstream.url))
        cls.proxy.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.proxy.__exit__(None, None, None)
        cls.upstream.__exit__(None, None, None)


class Compression(ProxyCase):

    def test_responses_are_compressed_when_the_client_accepts_gzip(self):
        with httpx.Client() as client, client.stream(
                "GET", self.proxy.url + "/big.js", headers={"accept-encoding": "gzip"}) as response:
            body = b"".join(response.iter_raw())
            self.assertEqual(response.headers.get("content-encoding"), "gzip")
        self.assertLess(len(body), len(BIG_JS) / 20, "a compressao nao reduziu o peso")
        self.assertEqual(gzip.decompress(body), BIG_JS, "o conteudo mudou ao comprimir")

    def test_upstream_alone_does_not_compress(self):
        '''Controle: sem o proxy o peso e inteiro. E o que o MLflow faz hoje.'''
        with httpx.Client() as client, client.stream(
                "GET", self.upstream.url + "/big.js", headers={"accept-encoding": "gzip"}) as response:
            body = b"".join(response.iter_raw())
            self.assertIsNone(response.headers.get("content-encoding"))
        self.assertEqual(len(body), len(BIG_JS))

    def test_no_gzip_is_sent_to_a_client_that_did_not_ask(self):
        with httpx.Client() as client, client.stream(
                "GET", self.proxy.url + "/big.js", headers={"accept-encoding": "identity"}) as response:
            body = b"".join(response.iter_raw())
            self.assertIsNone(response.headers.get("content-encoding"))
        self.assertEqual(body, BIG_JS)

    def test_cache_headers_survive(self):
        response = httpx.get(self.proxy.url + "/big.js")
        self.assertEqual(response.headers.get("etag"), '"abc"')
        self.assertIn("max-age=2419200", response.headers.get("cache-control", ""))


class OriginRewriting(ProxyCase):
    PUBLIC = "https://meu-tunel.exemplo.com"

    def test_a_public_tunnel_origin_is_blocked_without_the_proxy(self):
        '''Controle: reproduz o 403 do MLflow. Sem isto o teste abaixo nao provaria nada.'''
        response = httpx.post(self.upstream.url + "/echo", content="{}",
                              headers={"origin": self.PUBLIC})
        self.assertEqual(response.status_code, 403)

    def test_the_proxy_makes_the_same_post_succeed(self):
        response = httpx.post(self.proxy.url + "/echo", content='{"experiment_ids":["3"]}',
                              headers={"origin": self.PUBLIC, "referer": self.PUBLIC + "/#/x"})
        self.assertEqual(response.status_code, 200)
        seen = response.json()
        self.assertEqual(seen["origin"], self.upstream.url)
        self.assertEqual(seen["referer"], self.upstream.url + "/")
        self.assertEqual(seen["body"], '{"experiment_ids":["3"]}', "o corpo do POST foi alterado")

    def test_host_is_the_upstreams_not_the_tunnels(self):
        seen = httpx.post(self.proxy.url + "/echo", content="x",
                          headers={"host": "meu-tunel.exemplo.com"}).json()
        self.assertEqual(seen["host"], f"127.0.0.1:{self.upstream.port}")

    def test_requests_without_origin_are_left_without_one(self):
        self.assertIsNone(httpx.get(self.proxy.url + "/echo").json()["origin"])

    def test_upstream_receives_identity_so_we_compress_exactly_once(self):
        seen = httpx.get(self.proxy.url + "/echo", headers={"accept-encoding": "gzip"}).json()
        self.assertEqual(seen["encoding"], "identity")


class Passthrough(ProxyCase):

    def test_status_and_custom_headers_pass_through(self):
        response = httpx.get(self.proxy.url + "/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers.get("x-marca"), "preservada")

    def test_event_streams_arrive_progressively_and_uncompressed(self):
        '''Um stream de eventos (o assistente do MLflow usa) nao pode ficar retido ate o fim.'''
        arrivals = []
        started = time.time()
        with httpx.Client() as client, client.stream(
                "GET", self.proxy.url + "/events", headers={"accept-encoding": "gzip"}) as response:
            self.assertIsNone(response.headers.get("content-encoding"),
                              "um stream SSE nao deve ser comprimido")
            for chunk in response.iter_raw():
                if chunk:
                    arrivals.append(time.time() - started)
        self.assertGreaterEqual(len(arrivals), 2)
        self.assertLess(arrivals[0], arrivals[-1] - 0.3,
                        f"o stream chegou de uma vez so, retido pelo proxy: {arrivals}")


class UpstreamDown(unittest.TestCase):

    def test_a_dead_upstream_is_a_clear_502_not_a_hang(self):
        with Served(build_app(f"http://127.0.0.1:{free_port()}")) as proxy:
            response = httpx.get(proxy.url + "/anything", timeout=30)
        self.assertEqual(response.status_code, 502)
        self.assertIn("nao respondeu", response.text)


if __name__ == "__main__":
    unittest.main()
