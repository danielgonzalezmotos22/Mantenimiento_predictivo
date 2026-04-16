"""GET /sensores — devuelve el histórico de lecturas con filtros básicos."""
from fastapi import APIRouter, Depends, Query

from app.routers.deps import get_sensor_store
from app.services.sensor_store import SensorStore

router = APIRouter(tags=["sensores"])


@router.get("/sensores")
def get_sensors(
    reactor_id: str | None = Query(default=None, description="Filtra por reactor"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    store: SensorStore = Depends(get_sensor_store),
) -> dict:
    rows = store.query(reactor_id=reactor_id, limit=limit, offset=offset)
    return {
        "total": store.total_rows,
        "returned": len(rows),
        "limit": limit,
        "offset": offset,
        "reactor_id": reactor_id,
        "data": rows,
    }


@router.get("/sensores/reactores")
def get_reactors(store: SensorStore = Depends(get_sensor_store)) -> dict:
    return {"reactors": store.reactors()}
