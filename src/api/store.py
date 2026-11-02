from sqlalchemy import (JSON, BigInteger, Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, create_engine, desc, func, insert, select, text)

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("model_version", String(16), nullable=False),
    Column("source", String(32), nullable=False, server_default="api", index=True),
    Column("zip", String(5)),
    Column("request", JSON, nullable=False),
    Column("features", JSON, nullable=False),
    Column("predicted_rent", Float, nullable=False),
    Column("range_low", Float, nullable=False),
    Column("range_high", Float, nullable=False),
    Column("rent_today", Float, nullable=False),
    Column("known_location", Boolean, nullable=False),
    Column("latency_ms", Float, nullable=False),
)

rent_actuals = Table(
    "rent_actuals", metadata,
    Column("id", Integer, primary_key=True),
    Column("prediction_id", Integer, ForeignKey("predictions.id"), nullable=False, unique=True),
    Column("listing_id", BigInteger),
    Column("rent", Float, nullable=False),
    Column("observed_at", DateTime, nullable=False),
    Column("observed_month", Date, nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

simulations = Table(
    "simulations", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("model_version", String(16), nullable=False),
    Column("zip", String(5)),
    Column("request", JSON, nullable=False),
    Column("result", JSON, nullable=False),
    Column("latency_ms", Float, nullable=False),
)


class Store:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True)
        metadata.create_all(self.engine)

    def log(self, table: Table, row: dict) -> int:
        with self.engine.begin() as conn:
            return conn.execute(insert(table).values(**row).returning(table.c.id)).scalar_one()

    def log_many(self, table: Table, rows: list[dict]) -> list[int]:
        with self.engine.begin() as conn:
            return list(conn.execute(insert(table).returning(table.c.id, sort_by_parameter_order=True), rows).scalars())

    def recent(self, table: Table, limit: int) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(table).order_by(desc(table.c.id)).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
