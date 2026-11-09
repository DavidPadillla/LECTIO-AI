import asyncio
import unittest

from main import app


ALLOWED_ORIGINS = (
    "https://lectio-biblioteca.onrender.com",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
)


async def send_preflight(origin):
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "OPTIONS",
        "scheme": "http",
        "path": "/predictions",
        "raw_path": b"/predictions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"origin", origin.encode("ascii")),
            (b"access-control-request-method", b"GET"),
            (
                b"access-control-request-headers",
                b"authorization,content-type",
            ),
        ],
        "client": ("127.0.0.1", 5000),
        "server": ("127.0.0.1", 8000),
    }

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    return start["status"], {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start["headers"]
    }


class CorsPreflightTests(unittest.TestCase):
    def test_allowed_origins_receive_preflight_headers(self):
        for origin in ALLOWED_ORIGINS:
            with self.subTest(origin=origin):
                status, headers = asyncio.run(send_preflight(origin))
                self.assertEqual(status, 200)
                self.assertEqual(headers["access-control-allow-origin"], origin)
                self.assertIn("GET", headers["access-control-allow-methods"])
                self.assertIn("OPTIONS", headers["access-control-allow-methods"])
                self.assertIn("authorization", headers["access-control-allow-headers"].lower())
                self.assertIn("content-type", headers["access-control-allow-headers"].lower())
                self.assertNotIn("access-control-allow-credentials", headers)


if __name__ == "__main__":
    unittest.main()
