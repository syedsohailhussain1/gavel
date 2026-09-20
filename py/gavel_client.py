"""gavel_client: stdlib-only Python client for gavel-server.

Usage:
    from gavel_client import GavelClient

    client = GavelClient("http://localhost:7575")
    client.define("route_ticket", {"type": "object"}, {"type": "object"})
    client.train("route_ticket", [("charged twice", "billing")])
    decision = client.ask("route_ticket", {"subject": "refund my invoice"})
"""

import json
import urllib.error
import urllib.parse
import urllib.request


class GavelError(Exception):
    """Raised for HTTP errors and gavel-server error responses."""


class GavelClient:
    def __init__(self, base_url="http://localhost:7575"):
        self.base_url = base_url.rstrip("/")

    # -- internals ---------------------------------------------------------
    def _request(self, method, path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise GavelError(f"HTTP {e.code} on {method} {path}: {detail}") from None
        except urllib.error.URLError as e:
            raise GavelError(f"could not reach {self.base_url}: {e.reason}") from None

    @staticmethod
    def _normalize_examples(examples):
        """Accept [(input, label)] tuples or [{"input":..,"label":..}] dicts."""
        out = []
        for ex in examples:
            if isinstance(ex, dict):
                out.append({"input": ex["input"], "label": ex["label"]})
            else:
                text, label = ex
                out.append({"input": text, "label": label})
        return out

    # -- API ---------------------------------------------------------------
    def define(self, name, input_schema, output_schema, policy=None):
        payload = {
            "name": name,
            "inputSchema": input_schema,
            "outputSchema": output_schema,
        }
        if policy is not None:
            payload["policy"] = policy
        return self._request("POST", "/define", payload)

    def train(self, question, examples):
        return self._request(
            "POST", "/train",
            {"question": question, "examples": self._normalize_examples(examples)},
        )

    def calibrate(self, question, validation):
        return self._request(
            "POST", "/calibrate",
            {"question": question, "validation": self._normalize_examples(validation)},
        )

    def ask(self, question, input):
        return self._request("POST", "/ask", {"question": question, "input": input})

    def metrics(self, question):
        q = urllib.parse.quote(question, safe="")
        return self._request("GET", f"/metrics?question={q}")

    def questions(self):
        return self._request("GET", "/questions")
