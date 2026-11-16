"""n8n integration: the platform publishes events to an n8n webhook (N8N_WEBHOOK_URL); n8n workflows
(n8n/workflows/*.json) turn them into e-mails / Slack messages and call back the REST API (e.g. to approve
a purchase order or trigger the daily automation run). Without a configured URL events are only logged."""
from __future__ import annotations

from datetime import datetime

from scip.core.config import get_settings
from scip.core.logging import get_logger
from scip.integrations.http import ExternalAPIError, post_json

log = get_logger("n8n")


def publish(event_type: str, payload: dict) -> dict:
    url = get_settings().n8n_webhook_url
    body = {"event": event_type, "sent_at": datetime.now().isoformat(timespec="seconds"),
            "api_base_url": get_settings().api_base_url, **payload}
    if not url:
        return {"delivered": False, "reason": "N8N_WEBHOOK_URL not configured (event logged only)"}
    try:
        code = post_json(url, body)
        return {"delivered": 200 <= code < 300, "status_code": code}
    except ExternalAPIError as exc:
        log.warning("n8n webhook failed: %s", exc)
        return {"delivered": False, "reason": str(exc)}
