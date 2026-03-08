"""Smoke tests for the thriftlm CLI."""

from click.testing import CliRunner

from thriftlm.cli import cli


def test_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_serve_help():
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--api-key" in result.output
    assert "--port" in result.output
    assert "--host" in result.output


def test_serve_requires_api_key():
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code != 0
    assert "api-key" in result.output.lower() or "api_key" in result.output.lower()
