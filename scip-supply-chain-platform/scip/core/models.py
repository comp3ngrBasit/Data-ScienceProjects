"""SQLAlchemy ORM models — the unified schema every module reads and writes.

`data_origin` marks every business record as real (`real:uci`, `real:olist`, `real:dataco`, `real:m5`)
or `simulated`, so dashboards and reports can always distinguish the two.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------------------- reference
class DataSource(Base):
    __tablename__ = "data_sources"
    source_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(String(400))
    origin: Mapped[str] = mapped_column(String(20))           # real | simulated
    row_count: Mapped[int | None] = mapped_column(Integer)
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(Text)


class Country(Base):
    __tablename__ = "countries"
    country_code: Mapped[str] = mapped_column(String(3), primary_key=True)    # ISO alpha-3
    name: Mapped[str] = mapped_column(String(120))
    region: Mapped[str | None] = mapped_column(String(60))
    subregion: Mapped[str | None] = mapped_column(String(80))
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(10))
    source: Mapped[str] = mapped_column(String(30))                         # restcountries | static_fallback


class Location(Base):
    __tablename__ = "locations"
    location_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    location_type: Mapped[str] = mapped_column(String(30))                   # warehouse | customer_region | supplier_site
    name: Mapped[str] = mapped_column(String(160))
    city: Mapped[str | None] = mapped_column(String(120))
    state: Mapped[str | None] = mapped_column(String(80))
    country_code: Mapped[str | None] = mapped_column(String(3))
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    capacity_units: Mapped[float | None] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20))


class Product(Base):
    __tablename__ = "products"
    sku: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    category: Mapped[str | None] = mapped_column(String(120))
    unit_cost: Mapped[float | None] = mapped_column(Float)
    unit_price: Mapped[float | None] = mapped_column(Float)
    is_planning_sku: Mapped[bool] = mapped_column(Boolean, default=False)
    data_origin: Mapped[str] = mapped_column(String(20))
    source_id: Mapped[str | None] = mapped_column(ForeignKey("data_sources.source_id"))


class Supplier(Base):
    __tablename__ = "suppliers"
    supplier_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    country_code: Mapped[str | None] = mapped_column(String(3))
    city: Mapped[str | None] = mapped_column(String(120))
    state: Mapped[str | None] = mapped_column(String(80))
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    supplier_type: Mapped[str] = mapped_column(String(30), default="vendor")   # vendor | marketplace_seller
    data_origin: Mapped[str] = mapped_column(String(20))
    source_id: Mapped[str | None] = mapped_column(ForeignKey("data_sources.source_id"))


class SupplierProduct(Base):
    __tablename__ = "supplier_products"
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id", ondelete="CASCADE"), primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku", ondelete="CASCADE"), primary_key=True)
    unit_cost: Mapped[float] = mapped_column(Float)
    moq: Mapped[int] = mapped_column(Integer, default=1)
    lead_time_days: Mapped[float] = mapped_column(Float)
    lead_time_std: Mapped[float] = mapped_column(Float)
    capacity_per_month: Mapped[float | None] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20), default="simulated")


# ----------------------------------------------------------------------------- facts
class SalesDaily(Base):
    __tablename__ = "sales_daily"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku", ondelete="CASCADE"), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    quantity: Mapped[float] = mapped_column(Float)
    revenue: Mapped[float] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20))
    __table_args__ = (Index("ix_sales_sku_date", "sku", "date"),)


class Shipment(Base):
    __tablename__ = "shipments"
    shipment_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("data_sources.source_id"))
    order_date: Mapped[date] = mapped_column(Date)
    ship_date: Mapped[date | None] = mapped_column(Date)
    promised_date: Mapped[date | None] = mapped_column(Date)
    delivered_date: Mapped[date | None] = mapped_column(Date)
    ship_mode: Mapped[str | None] = mapped_column(String(40))
    scheduled_days: Mapped[float | None] = mapped_column(Float)
    actual_days: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30))            # delivered | in_transit | late_in_transit | cancelled
    is_late: Mapped[bool | None] = mapped_column(Boolean)
    supplier_id: Mapped[str | None] = mapped_column(String(60))
    origin_state: Mapped[str | None] = mapped_column(String(80))
    dest_state: Mapped[str | None] = mapped_column(String(80))
    dest_country: Mapped[str | None] = mapped_column(String(80))
    market: Mapped[str | None] = mapped_column(String(60))
    region: Mapped[str | None] = mapped_column(String(80))
    category: Mapped[str | None] = mapped_column(String(120))
    quantity: Mapped[float | None] = mapped_column(Float)
    value: Mapped[float | None] = mapped_column(Float)
    freight_value: Mapped[float | None] = mapped_column(Float)
    distance_km: Mapped[float | None] = mapped_column(Float)
    dest_lat: Mapped[float | None] = mapped_column(Float)
    dest_lng: Mapped[float | None] = mapped_column(Float)
    review_score: Mapped[float | None] = mapped_column(Float)
    customer_segment: Mapped[str | None] = mapped_column(String(40))
    department: Mapped[str | None] = mapped_column(String(60))
    order_type: Mapped[str | None] = mapped_column(String(30))
    items: Mapped[int | None] = mapped_column(Integer)
    discount_rate: Mapped[float | None] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20))
    __table_args__ = (Index("ix_ship_source_date", "source_id", "order_date"), Index("ix_ship_supplier", "supplier_id"))


class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku", ondelete="CASCADE"), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    on_hand: Mapped[float] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20), default="simulated")


class InventoryCurrent(Base):
    __tablename__ = "inventory_current"
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku", ondelete="CASCADE"), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    on_hand: Mapped[float] = mapped_column(Float)
    as_of: Mapped[date] = mapped_column(Date)
    data_origin: Mapped[str] = mapped_column(String(20), default="simulated")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    po_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"))
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku", ondelete="CASCADE"))
    quantity: Mapped[float] = mapped_column(Float)
    unit_cost: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)
    order_date: Mapped[date] = mapped_column(Date)
    promised_date: Mapped[date | None] = mapped_column(Date)
    received_date: Mapped[date | None] = mapped_column(Date)
    qty_received: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30))
    approval_level: Mapped[str | None] = mapped_column(String(20))
    created_by: Mapped[str] = mapped_column(String(40), default="history")
    approved_by: Mapped[str | None] = mapped_column(String(80))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    rationale: Mapped[str | None] = mapped_column(Text)
    predicted_late_risk: Mapped[float | None] = mapped_column(Float)
    data_origin: Mapped[str] = mapped_column(String(20), default="simulated")
    __table_args__ = (Index("ix_po_status", "status"), Index("ix_po_sku", "sku"))


class POApproval(Base):
    __tablename__ = "po_approvals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    po_id: Mapped[str] = mapped_column(ForeignKey("purchase_orders.po_id", ondelete="CASCADE"))
    action: Mapped[str] = mapped_column(String(30))
    actor: Mapped[str] = mapped_column(String(80))
    comment: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class WeatherDaily(Base):
    __tablename__ = "weather_daily"
    location_key: Mapped[str] = mapped_column(String(60), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    temp_max: Mapped[float | None] = mapped_column(Float)
    temp_min: Mapped[float | None] = mapped_column(Float)
    precipitation: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30))       # open-meteo-archive | open-meteo-forecast


# ----------------------------------------------------------------------------- analytics
class ModelRun(Base):
    __tablename__ = "model_runs"
    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(30))     # forecast | stockout | delivery_risk
    model_name: Mapped[str] = mapped_column(String(60))
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    data_window: Mapped[str | None] = mapped_column(String(80))
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)


class Forecast(Base):
    __tablename__ = "forecasts"
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    model: Mapped[str] = mapped_column(String(60), primary_key=True)
    yhat: Mapped[float] = mapped_column(Float)
    yhat_lower: Mapped[float | None] = mapped_column(Float)
    yhat_upper: Mapped[float | None] = mapped_column(Float)


class ForecastMetric(Base):
    __tablename__ = "forecast_metrics"
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id", ondelete="CASCADE"), primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), primary_key=True)
    model: Mapped[str] = mapped_column(String(60), primary_key=True)
    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    mape: Mapped[float | None] = mapped_column(Float)
    wape: Mapped[float | None] = mapped_column(Float)


class StockoutPrediction(Base):
    __tablename__ = "stockout_predictions"
    scored_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), primary_key=True)
    probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(20))
    run_id: Mapped[int | None] = mapped_column(Integer)


class SupplierScore(Base):
    __tablename__ = "supplier_scores"
    scored_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    supplier_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    shipments: Mapped[int] = mapped_column(Integer)
    on_time_rate: Mapped[float | None] = mapped_column(Float)
    avg_delay_days: Mapped[float | None] = mapped_column(Float)
    predicted_late_risk: Mapped[float | None] = mapped_column(Float)
    review_score: Mapped[float | None] = mapped_column(Float)
    lead_time_cv: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(20))
    data_origin: Mapped[str] = mapped_column(String(20))


class InventoryPolicy(Base):
    __tablename__ = "inventory_policy"
    computed_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), primary_key=True)
    abc_class: Mapped[str] = mapped_column(String(1))
    service_level: Mapped[float] = mapped_column(Float)
    avg_daily_demand: Mapped[float] = mapped_column(Float)
    demand_std: Mapped[float] = mapped_column(Float)
    lead_time_days: Mapped[float] = mapped_column(Float)
    lead_time_std: Mapped[float] = mapped_column(Float)
    safety_stock: Mapped[float] = mapped_column(Float)
    reorder_point: Mapped[float] = mapped_column(Float)
    eoq: Mapped[float] = mapped_column(Float)
    max_level: Mapped[float] = mapped_column(Float)
    on_hand: Mapped[float] = mapped_column(Float)
    on_order: Mapped[float] = mapped_column(Float)
    recommended_qty: Mapped[float] = mapped_column(Float)
    days_cover: Mapped[float | None] = mapped_column(Float)
    stockout_date: Mapped[date | None] = mapped_column(Date)
    inventory_value: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))
    demand_basis: Mapped[str] = mapped_column(String(60))


class ProcurementRecommendation(Base):
    __tablename__ = "procurement_recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    sku: Mapped[str] = mapped_column(String(40))
    supplier_id: Mapped[str] = mapped_column(String(60))
    quantity: Mapped[float] = mapped_column(Float)
    unit_cost: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)
    score_details: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[str] = mapped_column(Text)
    po_id: Mapped[str | None] = mapped_column(String(40))


class LogisticsPlan(Base):
    __tablename__ = "logistics_plans"
    plan_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    plan_type: Mapped[str] = mapped_column(String(30))
    objective_value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30))
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    severity: Mapped[str] = mapped_column(String(20))
    alert_type: Mapped[str] = mapped_column(String(40))
    entity: Mapped[str] = mapped_column(String(80))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open")
    __table_args__ = (Index("ix_alert_status", "status"),)


class AutomationLog(Base):
    __tablename__ = "automation_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    job: Mapped[str] = mapped_column(String(60))
    action: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    notified: Mapped[bool] = mapped_column(Boolean, default=False)


class Report(Base):
    __tablename__ = "reports"
    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    report_type: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    html: Mapped[str] = mapped_column(Text)


class RagDocument(Base):
    __tablename__ = "rag_documents"
    doc_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    doc_type: Mapped[str] = mapped_column(String(40))
    entity: Mapped[str | None] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class CopilotMessage(Base):
    __tablename__ = "copilot_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    provider: Mapped[str] = mapped_column(String(40))


class AppMetadata(Base):
    __tablename__ = "app_metadata"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)
