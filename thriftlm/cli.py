"""
thriftlm CLI entry point.

Usage:
    thriftlm serve --api-key sc_xxx
    thriftlm serve --api-key sc_xxx --port 8000 --host 127.0.0.1
"""

import os
import threading
import webbrowser

import click


@click.group()
def cli():
    """ThriftLM — semantic cache for LLM applications."""


@cli.command()
@click.option("--api-key", "-k", required=True, envvar="THRIFTLM_API_KEY",
              help="Your ThriftLM API key (or set THRIFTLM_API_KEY env var).")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
@click.option("--no-browser", is_flag=True, help="Don't open the browser automatically.")
def serve(api_key: str, port: int, host: str, no_browser: bool):
    """Start the ThriftLM local dashboard server.

    Reads SUPABASE_URL and SUPABASE_KEY from the environment (or .env file).
    Requires: pip install thriftlm[api]
    """
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "Missing dependencies. Run: pip install thriftlm[api]"
        )

    # Pass config to the server via env vars (uvicorn runs in the same process).
    os.environ["THRIFTLM_SERVE_API_KEY"] = api_key
    os.environ["THRIFTLM_SERVE_PORT"] = str(port)
    os.environ["THRIFTLM_SERVE_HOST"] = host

    # Load .env so SUPABASE_URL / SUPABASE_KEY are available if not already set.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = f"http://localhost:{port}"
    click.echo(f"ThriftLM dashboard → {url}")
    click.echo("Press Ctrl+C to stop.\n")

    if not no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    import uvicorn
    uvicorn.run("thriftlm._server:app", host=host, port=port, reload=False)
