# GeoServer security configuration – rag_geo read-only role

## Role and user

| Field | Value |
|---|---|
| Username | `rag_geo` |
| Role | `RAG_GEO` (custom role) |
| Permissions | Read-only WMS/WFS on workspace `east_india` |
| Admin REST API | Denied (HTTP 403 verified in evidence/geoserver_test_B_rag_geo.txt) |

## What was done (GeoServer web UI)

1. **Security → Users, Groups, Roles → Roles**: Added role `RAG_GEO`.
2. **Security → Users, Groups, Roles → Users**: Added user `rag_geo`, assigned role `RAG_GEO`. Password set from `GEOSERVER_RAG_PASSWORD` env variable (not recorded here).
3. **Security → Data**: Added a rule `east_india.*.r` → `RAG_GEO` (read access to all layers in the `east_india` workspace).
4. **Security → Services**: Added rule for WMS and WFS → `RAG_GEO`. REST admin (`/rest/**`) is left for ADMIN only.

## Verified behaviour

- Unauthenticated GET to `/geoserver/east_india/ows?service=WFS` → HTTP 401 (see `evidence/geoserver_test_A_no_login.txt`)
- `rag_geo` GET to the same URL → HTTP 200, GeoJSON with school features (see `evidence/geoserver_test_B_rag_geo.txt`)
- `rag_geo` GET to `/geoserver/rest/workspaces` → HTTP 403

## WFS example request (used in pipeline)

```
GET http://localhost:8080/geoserver/east_india/ows
    ?service=WFS&version=2.0.0&request=GetFeature
    &typeName=east_india:schools&outputFormat=application/json
    &count=5
Authorization: Basic <base64(rag_geo:GEOSERVER_RAG_PASSWORD)>
```

## Notes

- The `east_india` workspace publishes three layers: `schools`, `roads`, `landuse` (all EPSG:4326).
- The `schools` layer contains 30 curated school features with `board`, `seats`, `cut_off_marks`, `locality` columns (synthetic).
