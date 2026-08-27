# Logging and Environment Configuration

Operational logs and configuration should help diagnose behavior without exposing secrets.

## Environment variables

Use environment variables for values that vary by deployment, including service URLs, model names, timeouts, and secret references. Provide a `.env.example` containing names and safe placeholders, never working credentials. Validate numeric ranges and required values close to startup or when an optional feature is first invoked.

An empty string is not always equivalent to an unset value. Normalize optional secret fields so blank examples do not become apparently configured credentials. Keep production and local defaults distinct when their network addresses differ: an application on the host may use `http://localhost:6333`, while a container in Compose uses `http://qdrant:6333`.

## Structured logging

Use consistent event names and machine-readable key-value fields for important stages. Retrieval logs can include candidate counts, latency, collection, and embedding model. Generation logs can include provider, model, latency, and token usage when available. Do not log full prompts by default because retrieved documents and user questions may contain sensitive content.

## Correlation and latency

A request identifier lets operators connect API, retrieval, and generation events. Measure retrieval, generation, and total latency separately. Use monotonic clocks for elapsed durations. Wall-clock timestamps remain useful for cross-service correlation but should not calculate durations because system time can change.

## Secret handling

Secret values should use secret-aware configuration types and should be revealed only at the provider boundary. Avoid interpolating secret objects into exceptions or debug output. Rotate a credential immediately if it is committed, then remove it from the current tree; deleting it from one commit does not invalidate an exposed secret or remove every copy from history.

