"""The LiveKit signal forwarder, against a fake LiveKit server (a real WebSocket server on a free port; no LiveKit, no network)."""
import asyncio
import http
import socket
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import serve

from orb_agent.server import create_app
from orb_agent.voice import VoiceConfig
from orb_core.agent import SessionStore
from orb_discussion.agent import DiscussionAgent
from test_server import FakeLLM, ScriptedReader

KEY = "0123456789abcdef-0123456789"
TUNNEL_HOST = "quiet-river-1234.ngrok-free.dev"
THIS_COMPUTER = {"host": "127.0.0.1:8100"}  # the test client always connects a websocket to ws://testserver: each call names its Host


class FakeLiveKit:
    """Sends a binary greeting, echoes text as "echo:<text>" and bytes unchanged, and closes with 4001 on "bye"."""

    def __init__(self):
        self.paths, self.closed_by_client = [], threading.Event()
        ready = threading.Event()

        def run():
            async def main():
                self.stop = asyncio.Event()
                async with serve(self.handler, "127.0.0.1", 0, process_request=self.process) as server:
                    self.port = server.sockets[0].getsockname()[1]
                    self.loop = asyncio.get_running_loop()
                    ready.set()
                    await self.stop.wait()
            asyncio.run(main())

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert ready.wait(10)

    def process(self, connection, request):
        self.paths.append(request.path)
        if "/validate" in request.path:  # LiveKit tells why a connection would fail
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "invalid token\n")

    async def handler(self, connection):
        await connection.send(b"\x01hello")
        try:
            async for message in connection:
                if message == "bye":
                    await connection.close(code=4001, reason="done")
                    return
                await connection.send(message if isinstance(message, bytes) else "echo:" + message)
        finally:
            self.closed_by_client.set()

    def stop_now(self):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(10)


@pytest.fixture()
def livekit():
    server = FakeLiveKit()
    yield server
    server.stop_now()


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def app_for(server_url: str, share_key=None) -> TestClient:
    voice = VoiceConfig("ws://127.0.0.1:7880", "devkey", "secret", dev=True, server_url=server_url)
    store = SessionStore({"discussion": DiscussionAgent(ScriptedReader([]), FakeLLM())})
    return TestClient(create_app(store, voice, share_key=share_key), base_url="http://127.0.0.1:8100")


def test_text_and_binary_frames_pass_both_ways_with_the_query_unchanged(livekit):
    c = app_for(f"ws://127.0.0.1:{livekit.port}")
    query = "access_token=eyJhbGciOi.J9%2Bpayload%3D.sig-_&auto_subscribe=1&join_request=AbC%2FdEf%3D"
    with c.websocket_connect(f"/rtc/v1?{query}", headers=THIS_COMPUTER) as ws:
        assert ws.receive_bytes() == b"\x01hello"      # a frame that the server sends first
        ws.send_text("ping")
        assert ws.receive_text() == "echo:ping"
        ws.send_bytes(b"\x00\x01\x02\xff")
        assert ws.receive_bytes() == b"\x00\x01\x02\xff"
        ws.send_text("é" * 5000)                        # a large frame (a session description)
        assert ws.receive_text() == "echo:" + "é" * 5000
    assert livekit.paths == [f"/rtc/v1?{query}"]        # the same path and the same query, byte for byte
    assert livekit.closed_by_client.wait(5)             # the person's side closed, so the server's side closed too


def test_the_path_without_v1_and_the_close_code_of_the_server(livekit):
    c = app_for(f"ws://127.0.0.1:{livekit.port}")
    with c.websocket_connect("/rtc?access_token=t", headers=THIS_COMPUTER) as ws:
        ws.receive_bytes()
        ws.send_text("bye")                             # the server closes with 4001: the client must see that code
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        assert closed.value.code == 4001
    assert livekit.paths == ["/rtc?access_token=t"]


def test_a_server_that_does_not_answer_refuses_the_handshake_and_validate_says_why():
    c = app_for(f"ws://127.0.0.1:{closed_port()}")
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/rtc/v1?access_token=t", headers=THIS_COMPUTER):
            pass
    r = c.get("/rtc/v1/validate?access_token=t")
    assert r.status_code == 502 and "does not answer" in r.text


def test_validate_gives_the_text_and_status_of_livekit_and_nothing_else_is_forwarded(livekit):
    c = app_for(f"ws://127.0.0.1:{livekit.port}")
    for path in ("/rtc/validate", "/rtc/v1/validate"):
        r = c.get(path + "?access_token=bad")
        assert r.status_code == 401 and r.text.strip() == "invalid token"
    assert [p.split("?")[0] for p in livekit.paths] == ["/rtc/validate", "/rtc/v1/validate"]
    for path in ("/rtc/admin", "/rtc/v1/", "/rtc/../twirp/livekit.RoomService/ListRooms", "/twirp/livekit.RoomService/ListRooms"):
        assert c.get(path).status_code == 404, path      # only the two paths of the browser client; no admin API
    assert len(livekit.paths) == 2


def test_the_forwarder_is_behind_the_access_gate(livekit):
    c = TestClient(app_for(f"ws://127.0.0.1:{livekit.port}", share_key=KEY).app)   # no base_url: Host "testserver" is remote
    tunnel = {"host": TUNNEL_HOST}
    with pytest.raises(WebSocketDisconnect) as refused:
        with c.websocket_connect("/rtc/v1?access_token=t", headers=tunnel):
            pass
    assert refused.value.code == 1008
    assert c.get("/rtc/v1/validate", headers=tunnel).status_code == 403
    assert livekit.paths == []                           # nothing reached LiveKit
    with c.websocket_connect("/rtc/v1?access_token=t", headers={**tunnel, "x-orb-key": KEY, "x-forwarded-proto": "https"}) as ws:
        assert ws.receive_bytes() == b"\x01hello"
    assert c.get("/rtc/v1/validate", headers={**tunnel, "x-orb-key": KEY}).status_code == 401   # LiveKit's own answer


def test_it_is_on_only_for_a_livekit_server_on_this_computer(livekit):
    local = app_for(f"ws://127.0.0.1:{livekit.port}")
    assert local.get("/v1/voice/status").json()["proxy"] is True
    elsewhere = app_for("wss://orb-1a2b.livekit.cloud")
    assert elsewhere.get("/v1/voice/status").json()["proxy"] is False
    assert elsewhere.get("/rtc/v1/validate").status_code == 404 and livekit.paths == []   # no route: this is not a proxy to the internet
    with pytest.raises(WebSocketDisconnect):
        with elsewhere.websocket_connect("/rtc/v1?access_token=t", headers=THIS_COMPUTER):
            pass
