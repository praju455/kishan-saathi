# Backend Verification Report

Date: 2026-09-29

No secret, bearer token, farmer record, or provider payload is recorded in this report.

## Automated HTTP and application checks

- The OpenAPI-derived HTTP audit exercised all 88 registered operations through the ASGI app.
  It verified that all 77 protected operations fail closed without a bearer token, all four
  health/shared operations are reachable without authentication, and the seven Cognito auth
  operations remain in the public auth surface.
- Contract tests cover auth, profile validation, health/readiness, and the non-diagnostic disease
  progression response.
- Integration tests cover chat persistence and idempotency, scoped memory, weather cache/fallback,
  diagnosis ownership and multi-image behavior, reminders, reports, farms, crops, activities,
  storage cleanup, usage accounting, and provider failure paths.
- Unit tests cover Bedrock transport/error mapping, structured reply/action reconstruction and
  strict tools, Cloudinary signed private storage behavior, Cognito verification, Mem0 scoping,
  configuration validation, rate limits, and safety policies.

Final local commands:

```text
uv run ruff check app tests scripts
uv run mypy app tests scripts
uv run pytest -q
uv run alembic upgrade head --sql
```

The generated Alembic SQL traversed all 18 revisions through `20260815_0018`.
Ruff passed, mypy reported no issues across 196 source files, and pytest passed all 141 tests.

## Live provider probes

The supplied credentials were passed only as process environment variables and were never written
to the repository.

| Provider | Result | Evidence |
| --- | --- | --- |
| Amazon Bedrock | Passed | Direct Converse returned HTTP 200; typed risk routing, core reply, action extraction, safety review, and a real two-image progression request completed with `global.anthropic.claude-sonnet-4-6` in `ap-south-1`. The synthetic progression probe returned `improving` at 0.55 confidence under the `visible_symptom_progression_only` scope. |
| Mem0 | Passed | Non-mutating scoped search completed successfully and returned zero results. |
| Cognito JWKS | Passed | The configured pool endpoint returned two signing keys. |
| Cognito auth flow | Blocked | The user-pool app-client ID was not supplied; a probe with an empty client ID returned the expected `InvalidParameterException`. |
| OpenWeather | Blocked | No API key was supplied; the adapter returned its controlled provider error. |
| Supabase PostgreSQL | Blocked | No project/session-pooler connection string was supplied, so live migration and CRUD verification were not possible. |
| Cloudinary | Blocked | Cloud name, API key, and API secret were not supplied, so real upload/read/delete verification was not possible. |

The first attempted all-in-one Bedrock schema was rejected by the provider as too complex. The
adapter now uses two smaller strict schemas—one for visible reply content and one for diagnosis,
retake, and reminder actions—and reconstructs the existing domain contract with Pydantic. Both
schemas passed live calls.

The live LLM evaluation set also passed all three scenarios: routine English chat, a Hindi safety
case that refused requested chemical/dosage instructions without repeating the dosage, and a
linked-plot forecast tool call that incorporated the trusted weather result.

## Environment limitations

- No evaluated leaf classifier or approved checkpoint was supplied. Diagnosis remains explicitly
  disabled and fails closed; the API does not fabricate disease results.
- No approved speech provider was supplied, so voice transcription and synthesis remain
  unavailable rather than silently calling OpenAI.
- Docker is installed on the verification host, but its daemon was unavailable, so the image could
  not be built locally. The Dockerfile, entrypoint, and Render Blueprint were checked statically,
  and `render.yaml` parsed successfully as YAML.
- Production HTTPS, cross-user ownership, real Supabase rows, authenticated Cloudinary delivery,
  browser CORS, and rollback checks must be completed after the missing service credentials and
  frontend origin are available. The ordered procedure is in `docs/DEPLOYMENT.md`.

The Bedrock and Mem0 credentials disclosed in chat should be rotated before any deployment.
