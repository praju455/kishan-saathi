# Supabase and Render Deployment Runbook

This runbook is intentionally ordered so secrets never enter Git and a failed dependency cannot
produce a deceptively healthy deployment.

## 1. Rotate disclosed credentials

Rotate the Bedrock and Mem0 API keys that were pasted into chat. Use only the replacement values
in local `.env` files and Render secret fields. Do not reuse the disclosed values in production.

## 2. Complete provider prerequisites

1. In Cognito, locate the app client for pool `ap-northeast-1_2tJtNxZbb`. Enable
   `USER_PASSWORD_AUTH` and refresh-token auth, and record the app client ID. Do not create a client
   secret for a mobile public client.
2. In Bedrock, ensure the replacement key can invoke
   `global.anthropic.claude-sonnet-4-6` from `ap-south-1`.
3. Create a Cloudinary product environment and record its cloud name, API key, and API secret.
4. Create or rotate an OpenWeather API key and wait for provider activation if required.
5. Record the replacement Mem0 key.

## 3. Create Supabase PostgreSQL

1. Create a Supabase project in the closest practical region.
2. In **Connect**, copy the shared **Session pooler** URL (port `5432`). Render is IPv4-only; do
   not use the default IPv6 direct endpoint unless the Supabase IPv4 add-on is enabled.
3. Percent-encode reserved characters in the database password.
4. Append `sslmode=require`.
5. From a trusted machine, validate the schema without applying it:

   ```bash
   cd backend
   DATABASE_URL='postgresql://postgres.PROJECT_REF:ENCODED_PASSWORD@POOLER:5432/postgres?sslmode=require' \
     uv run alembic upgrade head --sql > /tmp/kishan-saathi-schema.sql
   ```

6. Apply migrations and confirm the current revision:

   ```bash
   DATABASE_URL='postgresql://postgres.PROJECT_REF:ENCODED_PASSWORD@POOLER:5432/postgres?sslmode=require' \
     uv run alembic upgrade head
   DATABASE_URL='postgresql://postgres.PROJECT_REF:ENCODED_PASSWORD@POOLER:5432/postgres?sslmode=require' \
     uv run alembic current
   ```

7. Do not publish table access through Supabase's anon key. The FastAPI backend is the ownership
   and authorization boundary. Add RLS policies only as a separately reviewed change if direct
   Data API access becomes a product requirement.

## 4. Create the Render service

1. Merge the feature pull request after review.
2. In Render, create a Blueprint from the repository's root `render.yaml`.
3. Populate every `sync: false` variable:
   `DATABASE_URL`, `COGNITO_APP_CLIENT_ID`, `BEDROCK_API_KEY`, `CLOUDINARY_CLOUD_NAME`,
   `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`, `OPENWEATHER_API_KEY`, `MEM0_API_KEY`, and
   `CORS_ALLOWED_ORIGINS` as a JSON array containing the deployed frontend origin.
4. Confirm the generated `AUTH_RATE_LIMIT_HMAC_KEY` is present.
5. Keep `RUN_MIGRATIONS_ON_START` unset. The Blueprint's pre-deploy command runs Alembic once,
   before the new service starts.
6. Deploy and require `/health/ready` to return `200` before sending traffic.

## 5. Production acceptance test

Create a dedicated test user in Cognito and run actual HTTPS requests against the Render URL:

1. Sign in and retain the access and refresh tokens outside shell history.
2. `GET /api/v1/me`, then complete the profile.
3. Create a farm, plot, crop, activity, reminder, and activity photo; retrieve each and verify a
   second Cognito user receives `404` for owner-scoped IDs.
4. Call current weather and plot forecast with valid and invalid locations.
5. Create a general chat, enqueue a message with an `Idempotency-Key`, poll the turn to
   `completed`, retrieve persisted messages, send a contextual follow-up, and delete the chat.
6. In Supabase, verify the two user messages and two assistant messages belong to that farmer and
   chat, then verify the deleted chat is inaccessible through the API.
7. Upload a multi-image diagnosis case with no plant name and with an optional plant name. Until
   the evaluated classifier exists, expect the documented `LEAF_MODEL_NOT_CONFIGURED` outcome;
   never accept fabricated disease certainty.
8. Delete uploaded media records and confirm cleanup jobs remove their authenticated Cloudinary
   assets.
9. Run `uv run python -m scripts.provider_smoke` and `uv run python -m scripts.llm_response_eval` with production-equivalent
   credentials from a controlled environment.

Record request IDs and HTTP status codes, but never tokens, farmer data, or provider secrets.

## 6. Rollback

Render can roll back the application image. Database migrations are forward-only in production;
do not run Alembic downgrades without a reviewed data-preservation plan. If a provider fails,
restore its secret or configuration and verify readiness before rolling application traffic.
