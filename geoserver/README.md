# GeoServer Configuration and Security: `rag_geo` Read-Only Role

## 1. Role and User Overview

| Item | Value |
|---|---|
| Username | `rag_geo` |
| Role | `ROLE_RAG_GEO_READ` (or `RAG_GEO`) |
| Data Rule | `east_india.*.r` (read-only access to all layers in workspace `east_india`) |
| Service Rule | `wfs.*` (allowed WFS/WMS read operations) |
| Catalog Mode | `CHALLENGE` (prompts authentication rather than hiding secured resources) |
| Admin REST API | Denied (`/rest/**` restricted to `ROLE_ADMINISTRATOR` only) |

## 2. Security Configuration Steps

1. **Role Creation**:
   - In GeoServer Admin: *Security* → *Users, Groups, Roles* → *Roles*.
   - Created role `ROLE_RAG_GEO_READ`.
2. **User Creation**:
   - In *Users*: Created user `rag_geo`, assigned role `ROLE_RAG_GEO_READ`.
   - The password is provided at runtime strictly via the environment variable `GEOSERVER_RAG_PASSWORD`. No passwords are saved or committed.
3. **Data Security**:
   - In *Security* → *Data*: Added access rule `east_india.*.r` assigned to `ROLE_RAG_GEO_READ` (granting read-only privilege on all workspace layers: `schools`, `roads`, `landuse`).
4. **Service Security & Catalog Mode**:
   - In *Security* → *Services*: Added rule permitting WFS (`wfs.*`) and WMS requests for `ROLE_RAG_GEO_READ`.
   - Catalog Mode is set to `CHALLENGE` so that unauthenticated requests to protected endpoints fail with HTTP 401 rather than empty results or silent hides.

## 3. Verified HTTP Responses

- **Unauthenticated access**:
  - Request: `GET /geoserver/east_india/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=east_india:schools&outputFormat=application/json` without authentication.
  - Result: **HTTP 401 Unauthorized** (catalog challenge mode active).
  - Verified in `evidence/geoserver_test_A_no_login.txt`.
- **Authenticated `rag_geo` access**:
  - Request: Same WFS request authenticated as `rag_geo` using `GEOSERVER_RAG_PASSWORD`.
  - Result: **HTTP 200 OK**, returns valid GeoJSON feature collection.
  - Verified in `evidence/geoserver_test_B_rag_geo.txt`.
- **Administrative REST API access**:
  - Request: `GET /geoserver/rest/workspaces.json` authenticated as `rag_geo`.
  - Result: **HTTP 403 Forbidden** (`rag_geo` has no administrative privileges).

## 4. How Layers Are Published (`publish_layers.py`)

Layers are published from the GeoPackage `east_india.gpkg` using `geoserver/publish_layers.py`:
- Uses the GeoServer REST API with administrator credentials (`GEOSERVER_ADMIN_PASSWORD`).
- Configures the data store `east_india_store` pointing to the GeoPackage.
- Automatically publishes three vector feature types: `schools`, `roads`, and `landuse` under EPSG:4326.
- Computes native and lat/lon bounding boxes.
- Performs an automated post-publish verification test against WFS using the read-only `rag_geo` identity.

```powershell
# Usage:
$env:GEOSERVER_ADMIN_PASSWORD="<admin_password>"
$env:GEOSERVER_RAG_PASSWORD="<rag_geo_password>"
python geoserver/publish_layers.py
```
