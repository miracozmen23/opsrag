# REST API Error Handling

Consistent errors make operational incidents easier to diagnose and client behavior safer.

## Client and server errors

Use 4xx responses when the client can change the request to succeed. Validation errors, unsupported media types, and malformed identifiers belong in this category. Use 5xx responses when the server or a required dependency cannot complete an otherwise valid request. A temporary Qdrant or LLM outage can be represented as HTTP 503 so clients know the request may succeed later.

Do not expose raw exception messages from database drivers or model providers to clients. Those messages can contain internal hostnames, query fragments, or provider details. Return a stable public message and log the original exception with correlation data on the server.

## Timeouts and retries

Every remote call should have an explicit timeout. Retry only operations that are safe to repeat, and limit both attempts and total time. Add exponential backoff with jitter to prevent many workers from retrying at exactly the same moment. Authentication failures and invalid requests should not be retried automatically.

For generation requests, an ambiguous timeout may occur after the provider accepted work. If automatic retry could create cost or duplicate an external side effect, use an idempotency mechanism when the provider supports it or return a controlled error.

## Validation

Validate input at the API boundary. Strip accidental surrounding whitespace but reject an empty question. Set a maximum question length to limit accidental large prompts and resource use. Domain services should still validate their own invariants because they may be called outside HTTP routes.

## Error logging

Log the error category, request identifier, operation name, and elapsed time. Never log API keys, authorization headers, passwords, or complete connection strings. A stack trace is useful for unexpected server failures but usually unnecessary for expected validation errors.

