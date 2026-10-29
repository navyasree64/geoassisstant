# Elasticsearch Security Architecture and `rag_app` Role

## 1. Overview and Principles

In compliance with the principle of least privilege, the application runs exclusively under a restricted application identity (`rag_app_user`) rather than the superuser `elastic`.

- **User**: `rag_app_user`
- **Assigned Role**: `rag_app`
- **Privileges**:
  - `indices`: `["east_india_chunks", "east_india_entities", "east_india_relations"]`
  - `privileges`: `["read", "write", "create_index", "view_index_metadata"]`
  - `cluster`: `[]` (NO cluster privileges)

## 2. Environment Variables and Secrets Handling

Passwords are supplied at runtime solely through environment variables:
- `ELASTIC_PASSWORD`: Used once during provisioning by `es/setup_security.py` to create the role and user.
- `RAG_APP_PASSWORD`: Used by all ingestion, retrieval, evaluation, and application pipelines.

**No passwords, API keys, or tokens are hard-coded, committed, or written to logs.**

## 3. Provisioning Procedure (`es/setup_security.py`)

The role and user are created via Elasticsearch's native Security REST API (`/_security/role` and `/_security/user`):

```powershell
$env:ELASTIC_PASSWORD = "<superuser_password>"
$env:RAG_APP_PASSWORD = "<app_user_password>"
python es/setup_security.py
```

The script executes two REST requests against `http://localhost:9200`:
1. `PUT /_security/role/rag_app` with index privileges limited to `east_india_chunks`, `east_india_entities`, and `east_india_relations`.
2. `PUT /_security/user/rag_app_user` assigning `roles: ["rag_app"]`.

## 4. Verification and 403 Proof

The restricted boundary of `rag_app_user` is verified by automated checks in `evalsuite/selfcheck.py`:

| Request by `rag_app_user` | HTTP Status | Explanation |
|---|---|---|
| `GET /_cluster/health` | **HTTP 403 Forbidden** | Cluster-level inspection is disallowed; role has zero cluster privileges. |
| `PUT /some_other_index` | **HTTP 403 Forbidden** | Arbitrary index creation outside the 3 allowed names is blocked. |
| `GET /east_india_chunks/_count` | **HTTP 200 OK** | Read access on allowed application indices succeeds. |
| `PUT /east_india_chunks` | **HTTP 200 OK** | Allowed to create/manage its own application index. |

Verification evidence is logged in `evidence/es_security_403.txt`.

## 5. Why Index Names Are Lowercase

The assignment statement references names such as `East India_chunks`, `East India_entities`, and `East India_relations`.
In Elasticsearch:
1. Index names must be **strictly lowercase**; uppercase characters are prohibited and rejected with `invalid_index_name_exception`.
2. Spaces are disallowed in index names.

Therefore, standard snake_case naming is enforced throughout this project:
- `east_india_chunks`
- `east_india_entities`
- `east_india_relations`
