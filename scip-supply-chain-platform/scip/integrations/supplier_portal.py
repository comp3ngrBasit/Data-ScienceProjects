"""MOCK supplier portal integration (no real supplier credentials exist).

`send_purchase_order` simulates the EDI/API acknowledgement a supplier system would return: a confirmation
number and a confirmed delivery date (quoted lead time). Deterministic for a given PO id. Replace
`MockSupplierPortal` with a real connector implementing the same `send_purchase_order` signature.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta


class MockSupplierPortal:
    name = "mock-supplier-portal"

    def send_purchase_order(self, po_id: str, supplier_id: str, sku: str, quantity: float, order_date: date,
                            lead_time_days: float) -> dict:
        digest = hashlib.sha1(f"{po_id}{supplier_id}".encode()).hexdigest()[:8].upper()
        return {"accepted": True, "confirmation_number": f"ACK-{digest}",
                "confirmed_delivery_date": (order_date + timedelta(days=int(round(lead_time_days)))).isoformat(),
                "portal": self.name, "mock": True}


def get_portal() -> MockSupplierPortal:
    return MockSupplierPortal()
