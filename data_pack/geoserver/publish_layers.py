"""Publishes schools, roads, landuse from data/east_india.gpkg into GeoServer workspace east_india via REST,
then tests WFS as the read-only user rag_geo. Passwords come from environment variables (never hard-coded).
  GEOSERVER_ADMIN_PASSWORD  -> admin password
  GEOSERVER_RAG_PASSWORD    -> rag_geo password
"""
import os, sys, requests
GS = "http://localhost:8080/geoserver"
admin = ("admin", os.environ["GEOSERVER_ADMIN_PASSWORD"])
H = {"Content-Type": "application/json", "Accept": "application/json"}
WS, STORE = "east_india", "east_india_store"

def show(label, r): print(f"{label}: {r.status_code} {r.text[:150]}")

r = requests.get(f"{GS}/rest/workspaces/{WS}.json", auth=admin)
if r.status_code == 401: sys.exit("Admin login refused (401). Check GEOSERVER_ADMIN_PASSWORD.")
show("workspace", r)
# remove the old single-layer store so the layer name 'schools' is free
show("delete old layer", requests.delete(f"{GS}/rest/layers/{WS}:schools", auth=admin))
show("delete old store", requests.delete(f"{GS}/rest/workspaces/{WS}/datastores/schools_store?recurse=true", auth=admin))
show("delete store if rerun", requests.delete(f"{GS}/rest/workspaces/{WS}/datastores/{STORE}?recurse=true", auth=admin))
body = {"dataStore": {"name": STORE, "type": "GeoPackage", "enabled": True,
        "connectionParameters": {"entry": [{"@key": "database", "$": "file:///opt/gisdata/east_india.gpkg"},
                                            {"@key": "dbtype", "$": "geopkg"}]}}}
show("create store", requests.post(f"{GS}/rest/workspaces/{WS}/datastores", json=body, headers=H, auth=admin))
for name in ("schools", "roads", "landuse"):
    ft = {"featureType": {"name": name, "nativeName": name, "title": name, "srs": "EPSG:4326", "enabled": True}}
    show(f"publish {name}", requests.post(f"{GS}/rest/workspaces/{WS}/datastores/{STORE}/featuretypes", json=ft, headers=H, auth=admin))
    show(f"bbox {name}", requests.put(f"{GS}/rest/workspaces/{WS}/datastores/{STORE}/featuretypes/{name}?recalculate=nativebbox,latlonbbox",
         json={"featureType": {"name": name}}, headers=H, auth=admin))

print("\n--- WFS test as rag_geo (read-only) ---")
rag = ("rag_geo", os.environ["GEOSERVER_RAG_PASSWORD"])
for name in ("schools", "roads", "landuse"):
    p = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"{WS}:{name}", outputFormat="application/json", maxFeatures=2)
    r = requests.get(f"{GS}/{WS}/ows", params=p, auth=rag)
    print(name, r.status_code, "features:", len(r.json()["features"]) if r.ok else r.text[:100])
print("\n--- same request without login (must be refused) ---")
r = requests.get(f"{GS}/{WS}/ows", params=dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"{WS}:schools", outputFormat="application/json", maxFeatures=1))
print("no login:", r.status_code)
