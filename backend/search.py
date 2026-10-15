from datetime import datetime

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    q: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    extension: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


def build_where(extension=None, date_from=None, date_to=None) -> dict | None:
    clauses = []
    if extension:
        clauses.append({"extension": {"$in": extension}})
    if date_from:
        clauses.append({"created_at_ts": {"$gte": date_from.timestamp()}})
    if date_to:
        clauses.append({"created_at_ts": {"$lte": date_to.timestamp()}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def normalize_distance(distance: float) -> float:
    return round(1 / (1 + distance), 4)


def shape_search_response(raw: dict, request: SearchRequest) -> dict:
    filters = {
        "extension": request.extension,
        "date_from": request.date_from.isoformat() if request.date_from else None,
        "date_to": request.date_to.isoformat() if request.date_to else None,
    }

    if not raw or not raw.get("ids"):
        return {
            "query": request.q,
            "total_results": 0,
            "filters": filters,
            "results": [],
            "files": [],
        }

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results = []
    file_summary = {}
    for rank, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True), start=1
    ):
        score = normalize_distance(dist)
        file_name = meta.get("file_name", "unknown")
        file_id = meta.get("file_id")
        # Group by file_id so distinct uploads sharing a name stay separate.
        # Chunks written before file_id existed fall back to the name.
        key = file_id if file_id is not None else ("name", file_name)

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "chunk_text": text,
                "distance": round(dist, 4),
                "score": score,
                "file": {
                    "file_id": file_id,
                    "file_name": file_name,
                    "extension": meta.get("extension"),
                    "created_at_ts": meta.get("created_at_ts"),
                },
            }
        )

        entry = file_summary.setdefault(
            key,
            {
                "file_id": file_id,
                "file_name": file_name,
                "hit_count": 0,
                "best_score": score,
                "best_distance": round(dist, 4),
            },
        )
        entry["hit_count"] += 1
        if score > entry["best_score"]:
            entry["best_score"] = score
            entry["best_distance"] = round(dist, 4)

    return {
        "query": request.q,
        "total_results": len(results),
        "filters": filters,
        "results": results,
        "files": sorted(
            file_summary.values(), key=lambda f: f["best_score"], reverse=True
        ),
    }
