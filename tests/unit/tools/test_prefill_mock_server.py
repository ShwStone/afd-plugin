# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

from tools.benchmarks.prefill_experiment import SystemConfig, wait_for_server
from tools.benchmarks.prefill_mock_server import PrefillMockRequestHandler


def test_mock_server_accepts_exact_token_prompt() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), PrefillMockRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
        )
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(
                {
                    "model": "prefill-mock",
                    "prompt": [10, 11, 12],
                    "max_tokens": 1,
                    "stream": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response_body = response.read().decode()
        connection.close()

        assert response.status == 200
        assert '"prompt_tokens": 3' in response_body
        assert "你" in response_body
        assert response_body.endswith("data: [DONE]\n\n")
        wait_for_server(
            SystemConfig(
                name="mock",
                base_url=f"http://127.0.0.1:{server.server_address[1]}",
                endpoint="/v1/completions",
                server_command_template="mock",
            ),
            "prefill-mock",
            timeout_seconds=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
