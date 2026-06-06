# Contract: Observability (wide events, OTel, sinks)

**Governs:** how every runtime in the project emits operational telemetry — the
wire format, what one event represents, the canonical fields, how a single unit
of work is traced across process/repo boundaries, where events are sent, and how
the schema is enforced. This is the interface that makes the whole system
(dashboard + training + serving) queryable from one event store.

> **Machine-readable companion:** the authoritative field list lives in
> [`observability.attributes.yaml`](./observability.attributes.yaml) — names,
> types, required/optional, and cardinality notes. The typed emit helpers and the
> CI schema test in each repo read **that** file; this document is the prose.

## Spec

### Wire format — OpenTelemetry, used with wide-event discipline

Telemetry is **OpenTelemetry over OTLP**. The discipline that matters more than
the SDK: **one fat root span per unit of work**, carrying a wide set of
high-cardinality attributes — *not* a swarm of thin spans or a stream of log
lines. Child spans are allowed for genuinely separable sub-steps (e.g. a training
run's data-pull vs. train vs. register), but the **root span is the canonical
wide event** for that unit of work and carries the full attribute set.

Choosing OTel buys vendor-neutrality: the same emit code targets SigNoz today and
any OTLP backend (e.g. Honeycomb free tier) tomorrow by changing one env var — see
**Sinks** below.

### Unit of work — the pattern (extensible without changing this contract)

A **runtime** is a process that exports telemetry. A **unit of work** is what one
wide event represents; one runtime emits several unit-of-work types. Committed
initial set:

| Runtime | Unit-of-work event types (root span each) |
|---|---|
| **Dashboard backend** (`terra-vigil`, FastAPI) | one HTTP request · one WebSocket "run" · one outbound inference call · one categorization pass |
| **Serving container** (pyfunc) | one inference invocation |
| **Training job** (`terra-incognita`) | the run lifecycle (child spans: data-pull, train, register) |

**Frontend RUM (browser) is explicitly deferred** to post-MVP. The set above is
*open*: any new kind of work instruments itself against this pattern (root span +
canonical fields + its type-specific fields), and does **not** require a contract
change to be added.

### Canonical fields — on every wide event

Every root span carries, at minimum (full list + types in the YAML registry):

- `environment` — `local` | `prod` (low-cardinality).
- `service.name` — the emitting runtime (OTel semconv).
- `git_sha` — provenance, joins to the deployed/trained artifact.
- `trace_id` / `span_id` — supplied by OTel; the cross-boundary join key.

Domain fields (`camtrap.*`) are attached per unit-of-work type — e.g.
`camtrap.model.version`, `camtrap.dataset.version`, `camtrap.run.id`,
`camtrap.image.id`, `camtrap.location.id`, `camtrap.error.type`. The registry is
authoritative.

### Trace-context propagation — a hard requirement

A dashboard **run** fans out into N inference calls to the **serving container** —
a different process, a different repo. The W3C `traceparent` header **must** be
propagated on every outbound inference call so the serving invocation's span joins
the run's trace. The result: a whole run — UI request → run → every inference →
categorization — is **one trace**, queryable end-to-end. This is the project's
headline observability capability; it is not optional.

### Boundary with MLflow — no overlap

Wide events own the **operational lifecycle** (did the run start, how long the S3
pull took, exit reason, spot-interrupted?, latency, counts). **MLflow owns ML
metrics** (mAP, precision/recall, per-class AP, loss curves). They **share keys**
(`camtrap.dataset.version`, `camtrap.model.version`, `git_sha`) so a run can be
joined across the two systems, but they **never duplicate payload**. Do not log
mAP into a wide event; do not log request latency into MLflow.

### Naming & cardinality

- **Naming:** OTel **semantic conventions** where they exist (`http.*`,
  `service.*`, …); a **`camtrap.*`** namespace for domain fields.
- **High-cardinality fields** (`camtrap.image.id`, `camtrap.run.id`, `trace_id`,
  `confidence`, `camtrap.model.version`, …) belong on **span attributes** — the
  event store is built to group/filter on them. They must **never** be used as
  **metric labels** (that explodes time-series cardinality). If a runtime also
  emits OTel metrics, their labels stay low-cardinality.

### Schema enforcement — strict in dev, never fatal in prod

- The attribute registry (YAML) is the single source of truth.
- Each repo wraps the OTel API in a **typed helper** (Pydantic-backed in Python)
  that imports field names from the registry — domain fields can only be set
  through typed functions, so wrong/missing fields are hard to write.
- **CI** asserts each canonical event carries its required fields (tested against
  the registry).
- **Strictness rule:** validate **loudly in dev and fail CI** on a missing
  required field — but **never let telemetry crash prod**. In prod, drop/warn
  instead. Telemetry must not take down the thing it observes.
- *Future stretch (not MVP):* codegen typed constants per language from the
  registry (OTel Weaver). Noted, not required.

### Sinks — same code everywhere, only the endpoint changes by env

Identical emit code in every environment; the **only** thing that differs is the
exporter endpoint — exactly the floci-vs-real-S3 parity pattern
([mlflow-topology.md](./mlflow-topology.md)), applied to telemetry. Concretely,
`OTEL_EXPORTER_OTLP_ENDPOINT` (+ an auth token for the deployed sink) is the whole
delta:

| | Local | Deployed (`prod`) |
|---|---|---|
| Emit | OTel SDK — *identical code* | *identical code* |
| Sink | **SigNoz in docker-compose** (always-on, alongside `mlflow` + `floci`) | **Self-hosted SigNoz on Hetzner** (CAX21 ARM), Terraform-provisioned; OTLP firewalled to the AWS egress IP — see [../infra/PLAN.md](../infra/PLAN.md) I8 |
| Differs by | `OTEL_EXPORTER_OTLP_ENDPOINT` only | same |

**Fallback:** because the wire format is OTLP, the deployed sink can be swapped to
the **Honeycomb free tier** with one env-var change and zero code change — the
concrete payoff of choosing vendor-neutral OTel.

## Depended on by

- **dashboard** (`terra-vigil`) — FastAPI middleware emits one wide event per
  request / run / inference call / categorization pass; propagates `traceparent`
  into the serving call. See [../dashboard/DESIGN.md](../dashboard/DESIGN.md).
- **training** (`terra-incognita`) — emits the run-lifecycle event (operational,
  distinct from MLflow autolog + provenance). See
  [../training/PLAN.md](../training/PLAN.md).
- **serving** (pyfunc container) — emits one event per inference invocation;
  continues the propagated trace.
- **infra** (`Terraform` repo) — provisions the deployed SigNoz on Hetzner
  (`hcloud` provider), firewalls the OTLP endpoint, and supplies the endpoint +
  auth token as config. Actionable delta: [../infra/PLAN.md](../infra/PLAN.md) I8.

## Rule

Do not redefine this elsewhere. Reference this file. A repo **never** invents its
own event schema or field names — it emits against
[`observability.attributes.yaml`](./observability.attributes.yaml). To change the
wire format, the canonical fields, propagation, the MLflow boundary, or the sink
strategy, change it **here**.
