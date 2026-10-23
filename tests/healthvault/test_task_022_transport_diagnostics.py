from __future__ import annotations

import asyncio
import io
import json
import unittest
import urllib.error
from email.message import Message
from unittest.mock import patch

from healthvault.evaluation.skill.benchmark.task_022 import (
    TransportError,
    UrllibJsonTransport,
)


class UrllibJsonTransportDiagnosticsTests(unittest.TestCase):
    def test_http_401_retains_only_redacted_provider_diagnostics(self) -> None:
        secret = "unit-test-secret-key"
        headers = Message()
        headers["X-Request-ID"] = "req-safe-401"
        body = json.dumps({
            "error": {
                "code": "AUTH-401",
                "message": f"Rejected Bearer {secret}; supplied key was {secret}",
            }
        }).encode("utf-8")
        error = urllib.error.HTTPError(
            "https://provider.invalid/chat/completions",
            401,
            "Unauthorized",
            headers,
            io.BytesIO(body),
        )
        transport = UrllibJsonTransport()
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(TransportError) as raised:
                asyncio.run(transport.post_json(
                    "https://provider.invalid/chat/completions",
                    {
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                    },
                    {"model": "fake", "messages": [{"role": "user", "content": "OK"}]},
                    1.0,
                ))

        caught = raised.exception
        self.assertFalse(caught.retryable)
        self.assertEqual({
            "http_status": 401,
            "provider_error_code": "AUTH-401",
            "provider_error_message": "Rejected [REDACTED]; supplied key was [REDACTED]",
            "request_id": "req-safe-401",
        }, caught.safe_diagnostics())
        serialized = json.dumps(caught.safe_diagnostics(), sort_keys=True)
        self.assertNotIn(secret, serialized)
        self.assertNotIn("Authorization", serialized)


if __name__ == "__main__":
    unittest.main()
