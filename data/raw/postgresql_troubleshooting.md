# PostgreSQL Troubleshooting

This guide covers common application-to-PostgreSQL failures in local and containerized deployments. Diagnose the network path and server readiness before changing credentials or application code.

## Connection refused

A connection-refused error means the operating system reached the target host but no process accepted the requested TCP port. Confirm that PostgreSQL is running, listening on the expected interface, and bound to the configured port. On the database host, check service status and the PostgreSQL logs. A database process can be running while still listening only on `localhost`, which prevents connections from another container or host.

In Docker Compose, applications must connect to the database service name and the container port, for example `postgres:5432`. `localhost` inside the application container refers to that application container, not the PostgreSQL container. A host-published port such as `5433:5432` is used by clients running on the host; sibling containers still use port `5432`.

Startup order is another frequent cause. `depends_on` can start the database container first, but process start is not the same as database readiness. Add a PostgreSQL health check, wait for healthy status where supported, and keep bounded retry logic with exponential backoff in the application. Do not use an unbounded retry loop because it hides persistent configuration failures.

## Authentication failed

An authentication error proves that a server answered, so focus on database name, user, password, and `pg_hba.conf`. Verify the effective connection string without printing the password. Environment variables supplied to an existing database container may not recreate roles after the data directory has already been initialized. Change the role explicitly or recreate only disposable development data with clear intent.

Check whether the application is connecting with SSL when the server expects plaintext, or vice versa. Managed services often require TLS and may provide a CA certificate. Certificate verification should not be disabled as a permanent fix.

## Too many connections

Connection exhaustion commonly comes from creating a new connection for every request, failing to return connections to a pool, or running too many workers with oversized pools. Estimate the total possible connections across every application replica and worker. Reserve capacity for administration and migrations, then set a bounded pool size and timeout. Inspect `pg_stat_activity` to distinguish active work from idle or leaked sessions.

## Slow queries and locks

Start with the slow query log and `EXPLAIN (ANALYZE, BUFFERS)` on a safe representative query. Avoid guessing that every slow request needs a new index. Look for missing selective indexes, sequential scans over large tables, stale statistics, lock waits, and application transactions that remain open. `pg_stat_activity` and `pg_locks` help identify blockers. Terminating sessions can roll back work, so identify the owning service and transaction first.

