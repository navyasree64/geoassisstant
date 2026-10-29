"""One-time security setup script for Elasticsearch: creates the limited rag_app role
and rag_app_user.

Usage:
  $env:ELASTIC_PASSWORD = "<superuser_password>"
  $env:RAG_APP_PASSWORD = "<app_password>"
  python es/setup_security.py

Security Guarantee:
  Passwords are read strictly from environment variables and never logged or persisted.
"""
import os, requests

ES = "http://localhost:9200"
admin = ("elastic", os.environ["ELASTIC_PASSWORD"])

role = {
    "indices": [{
        "names": ["east_india_chunks", "east_india_entities", "east_india_relations"],
        "privileges": ["read", "write", "create_index", "view_index_metadata"],
    }]
}
r = requests.put(f"{ES}/_security/role/rag_app", json=role, auth=admin)
print("role:", r.status_code, r.text)

user = {"password": os.environ["RAG_APP_PASSWORD"], "roles": ["rag_app"]}
r = requests.put(f"{ES}/_security/user/rag_app_user", json=user, auth=admin)
print("user:", r.status_code, r.text)