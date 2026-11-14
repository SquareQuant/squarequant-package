import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must run without network access")

    monkeypatch.setattr(socket.socket, "connect", blocked)
