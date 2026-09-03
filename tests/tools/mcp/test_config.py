import pytest

from tools.mcp.config import MCPServerConfig


def test_creates_stdio_server_config():
    config = MCPServerConfig(
        name="enterprise-test-server",
        transport="stdio",
        command="/usr/bin/python",
        args=(
            "server.py",
            "--test",
        ),
        env={
            "ENVIRONMENT": "test",
        },
        cwd="/tmp/mcp",
    )

    assert config.name == "enterprise-test-server"
    assert config.transport == "stdio"
    assert config.command == "/usr/bin/python"
    assert config.args == (
        "server.py",
        "--test",
    )
    assert config.env == {
        "ENVIRONMENT": "test",
    }
    assert config.cwd == "/tmp/mcp"


def test_defaults_optional_stdio_configuration():
    config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command="python",
    )

    assert config.args == ()
    assert config.env == {}
    assert config.cwd is None


def test_rejects_empty_server_name():
    with pytest.raises(
        ValueError,
        match="MCP server name must not be empty",
    ):
        MCPServerConfig(
            name=" ",
            transport="stdio",
            command="python",
        )


def test_rejects_empty_transport():
    with pytest.raises(
        ValueError,
        match="MCP server transport must not be empty",
    ):
        MCPServerConfig(
            name="test-server",
            transport=" ",
            command="python",
        )


def test_stdio_requires_command():
    with pytest.raises(
        ValueError,
        match="MCP stdio server requires a command",
    ):
        MCPServerConfig(
            name="test-server",
            transport="stdio",
        )


def test_rejects_empty_command():
    with pytest.raises(
        ValueError,
        match="MCP server command must not be empty",
    ):
        MCPServerConfig(
            name="test-server",
            transport="stdio",
            command=" ",
        )


def test_rejects_empty_argument():
    with pytest.raises(
        ValueError,
        match="MCP server arguments must not contain empty values",
    ):
        MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="python",
            args=(
                "server.py",
                "",
            ),
        )


def test_rejects_empty_environment_variable_name():
    with pytest.raises(
        ValueError,
        match="MCP server environment variable names must not be empty",
    ):
        MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="python",
            env={
                "": "value",
            },
        )


def test_rejects_empty_cwd():
    with pytest.raises(
        ValueError,
        match="MCP server cwd must not be empty",
    ):
        MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="python",
            cwd=" ",
        )


def test_configuration_is_immutable():
    config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command="python",
    )

    with pytest.raises(AttributeError):
        config.name = "changed"


def test_streamable_http_config_requires_url():
    with pytest.raises(
        ValueError,
        match="requires a URL",
    ):
        MCPServerConfig(
            name="remote-server",
            transport="streamable-http",
        )


def test_streamable_http_config_accepts_url():
    config = MCPServerConfig(
        name="remote-server",
        transport="streamable-http",
        url="http://127.0.0.1:8000/mcp",
    )

    assert config.name == "remote-server"
    assert config.transport == "streamable-http"
    assert config.url == "http://127.0.0.1:8000/mcp"


def test_streamable_http_config_accepts_headers():
    config = MCPServerConfig(
        name="remote-server",
        transport="streamable-http",
        url="http://127.0.0.1:8000/mcp",
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert config.headers == {
        "Authorization": "Bearer test-token",
    }


def test_streamable_http_config_rejects_empty_header_name():
    with pytest.raises(
        ValueError,
        match="header names",
    ):
        MCPServerConfig(
            name="remote-server",
            transport="streamable-http",
            url="http://127.0.0.1:8000/mcp",
            headers={
                "": "value",
            },
        )
