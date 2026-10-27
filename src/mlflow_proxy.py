'''
Proxy HTTP na frente do `mlflow ui`, para acessa-lo por um tunel.

Resolve dois problemas, medidos contra o MLflow 3.16.1 desta maquina, que fazem a interface
"carregar para sempre" ou abrir sem dados quando o acesso e por tunel:

  1. PESO. O servidor (`uvicorn`) nao comprime nada. A pagina de runs baixa ~8.5 MB: o `main.js` tem
     4.4 MB, mais um chunk de 1.15 MB e 0.83 MB de CSS. Com gzip o `main.js` cai para 1.2 MB (3.6x) e
     o CSS para 95 KB (8.8x). Os estaticos tem `cache-control: max-age=28d`, entao o peso e so da
     primeira visita -- e se ela nunca termina, o cache nunca enche.

  2. ORIGIN. O middleware de seguranca do MLflow responde 403 "Cross-origin request blocked" a um
     POST cujo `Origin` nao seja `localhost`. `runs/search` e `experiments/search` sao POSTs. Num
     tunel `ssh -L` o Origin e `http://localhost:PORTA` e passa; num tunel publico
     (`https://algo.devtunnels.ms`, ngrok, cloudflared) o Origin e a URL publica, a lista de
     experimentos carrega (e GET) e a tabela de runs fica vazia. O proxy reescreve `Origin` e
     `Referer` para os do servidor, entao funciona seja qual for o tunel.

O proxy so repassa HTTP -- nao conhece o MLflow --, entao nao depende de nenhum detalhe interno dele.
'''
from typing import Dict
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route

# RFC 7230 6.1, mais `host` (reescrito pelo cliente httpx) e `accept-encoding` (pedimos `identity`
# ao upstream para comprimir UMA vez, aqui, em vez de comprimir duas).
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "accept-encoding"}


def _origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def build_app(upstream: str, minimum_size: int = 500) -> Starlette:
    upstream_origin = _origin_of(upstream)

    async def lifespan(app):
        app.state.client = httpx.AsyncClient(
            base_url=upstream, follow_redirects=False,
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
        yield
        await app.state.client.aclose()

    async def proxy(request: Request):
        headers: Dict[str, str] = {k: v for k, v in request.headers.items()
                                   if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        # O upstream enxerga uma requisicao do mesmo site. Sem isto o middleware do MLflow barra o
        # POST de um navegador cuja origem e a URL publica do tunel.
        if "origin" in headers:
            headers["origin"] = upstream_origin
        if "referer" in headers:
            headers["referer"] = upstream_origin + "/"

        try:
            upstream_request = request.app.state.client.build_request(
                request.method, httpx.URL(path=request.url.path, query=request.url.query.encode()),
                headers=headers, content=await request.body())
            upstream_response = await request.app.state.client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            return PlainTextResponse(
                f"O servidor MLflow em {upstream} nao respondeu ({type(error).__name__}).\\n"
                "Ele esta rodando? Confira com `curl " + upstream + "/health`.", status_code=502)

        response_headers = {k: v for k, v in upstream_response.headers.items()
                            if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(upstream_response.aiter_raw(), status_code=upstream_response.status_code,
                                 headers=response_headers,
                                 background=BackgroundTask(upstream_response.aclose))

    methods = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    return Starlette(
        routes=[Route("/{path:path}", proxy, methods=methods)],
        middleware=[Middleware(GZipMiddleware, minimum_size=minimum_size)],
        lifespan=lifespan,
    )
