# Kishan Saathi Backend

Production-oriented FastAPI backend for an authenticated farmer-assistance mobile app.

## Architecture

The backend uses Python 3.12, FastAPI, SQLAlchemy, Alembic, and `uv`. Product modules own
their routes, schemas, services, repositories, and models. External providers sit behind typed
integration boundaries:

- Amazon Cognito validates access tokens and manages the account lifecycle.
- Amazon Bedrock Converse generates structured chat replies, titles, routing decisions,
  canonical-memory candidates, and non-diagnostic progression comparisons.
- Supabase PostgreSQL stores profiles, farms, plots, crop cycles, diagnoses, chat history,
  reminders, weather snapshots, memory records, activity timelines, reports, cleanup jobs, and
  provider-usage events.
- Cloudinary stores authenticated, owner-scoped diagnosis and activity images.
- OpenWeather supplies current conditions; Open-Meteo supplies forecasts and geocoding.
- Mem0 indexes canonical facts that remain anchored to PostgreSQL records.

Chat and weather have independent routers, services, repositories, and providers. Plot-scoped
chat receives weather only through the narrow `PlotWeatherPort`; the chat service does not import
the weather module.

## Local setup

```bash
cd backend
uv sync --group dev
cp .env.example .env
docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

The API exposes OpenAPI at `/docs`, liveness at `/health/live`, and database-backed readiness at
`/health/ready`. Product endpoints are under `/api/v1`.

## Required production configuration

| Area | Variables |
| --- | --- |
| App | `APP_ENV=production`, `DEBUG=false`, `AUTH_RATE_LIMIT_HMAC_KEY` |
| Database | `DATABASE_URL` with TLS, pool size and timeout settings |
| Cognito | `COGNITO_REGION`, `COGNITO_USER_POOL_ID`, `COGNITO_APP_CLIENT_ID` |
| Bedrock | `BEDROCK_API_KEY`, `BEDROCK_MODEL_ID`, `BEDROCK_REGION` |
| Cloudinary | `STORAGE_BACKEND=cloudinary`, `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` |
| Weather | `OPENWEATHER_API_KEY`; Open-Meteo needs no key for the configured APIs |
| Memory | `MEM0_API_KEY` |
| Web clients | `CORS_ALLOWED_ORIGINS` as a JSON array; keep `[]` for native-mobile-only access |

See [.env.example](.env.example) for every setting and safe defaults. Never commit a populated
`.env` file.

### Supabase PostgreSQL

Render does not provide IPv6 database egress, so use the Supabase shared **session pooler** URL on
port `5432`, append `sslmode=require`, and store it as Render's `DATABASE_URL` secret. Settings
normalize Supabase's `postgres://` or `postgresql://` scheme to SQLAlchemy's asyncpg dialect and
translate `sslmode` to the asyncpg `ssl` parameter.

Alembic remains the schema authority:

```bash
DATABASE_URL='postgresql://postgres.PROJECT_REF:ENCODED_PASSWORD@POOLER:5432/postgres?sslmode=require' \
  uv run alembic upgrade head
```

The mobile app must not connect to these tables with the database password. Keep Supabase Data
API access denied unless explicit RLS policies are designed and tested later.

### Amazon Cognito

The Cognito and Bedrock regions are intentionally separate. The configured user pool region is
used to build and verify the exact issuer, JWKS URL, token expiry, `token_use`, and app-client
binding. The app client must enable the password and refresh-token flows used by the auth routes.

### Amazon Bedrock

The adapter calls the official Converse endpoint directly with a Bedrock API key. It uses
validated JSON-schema outputs, bounded retries, a maximum four-step tool loop, deterministic
Pydantic safety validation, a second structured safety review, and usage recording. Provider
authentication, authorization, throttling, timeout, invalid-model, and malformed-response errors
map to neutral application errors without exposing provider details.

### Cloudinary

Production images use Cloudinary's `authenticated` delivery type. Public IDs have the form
`<folder>/<farmer-id>/<category>/<opaque-id>`. Every read and delete validates that owner prefix
before network access. Reads use one-minute signed download URLs; deletes remain idempotent so the
durable cleanup worker can retry safely.

## Chat lifecycle

Chats are owned by one authenticated farmer and may be general, farm, plot, or scan scoped.
`POST /api/v1/chats/{chat_id}/messages` queues an ordered durable turn and returns `202`. Clients
poll the turn endpoint. The worker stores both the user message and the structured assistant
reply, preserves history, prevents duplicate paid calls with an idempotency key, and enforces
ownership on list, read, update, retry, and delete operations.

## Verification

```bash
uv sync --group dev
uv run ruff check app tests scripts
uv run mypy app tests scripts
uv run pytest
uv run alembic upgrade head --sql
```

Run configured provider probes with `uv run python -m scripts.provider_smoke` and the live LLM
evaluation set with `uv run python -m scripts.llm_response_eval`. These scripts make real network
requests and must only be run with rotated, non-committed credentials. AWS can spend several
minutes compiling a newly seen structured-output schema, so the default Bedrock timeout allows a
bounded five-minute cold path.

## Render

The repository root contains `render.yaml`. It builds the backend Docker image, runs Alembic as a
pre-deploy command, starts Uvicorn on Render's assigned port, and checks `/health/ready`. Follow
[the deployment runbook](docs/DEPLOYMENT.md) before creating the service.

## Known limitations

- Leaf classification remains deliberately disabled until an evaluated checkpoint, immutable
  class manifest, and serving contract are supplied. Image uploads and diagnosis case history
  still work, but classification returns `LEAF_MODEL_NOT_CONFIGURED` rather than invented results.
- Exact chemical product, ingredient, dosage, mixing, frequency, and application instructions
  remain blocked until an approved local authoritative source is configured.
- Voice transcription and synthesis return `VOICE_PROVIDER_NOT_CONFIGURED`; the previous provider
  was removed and no approved replacement credentials or contract were supplied.
- A live deployment cannot be verified until Supabase, Cloudinary, OpenWeather, and Cognito app
  client values are available. See the endpoint report for the precise verified/unverified split.
