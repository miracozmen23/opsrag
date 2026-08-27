# Docker and Compose Troubleshooting

Use this guide for container startup, networking, health, storage, and image problems. Begin with container state and logs, then verify configuration rendered by Compose.

## Container exits immediately

Run `docker compose ps -a` to see the exit code and `docker compose logs <service>` to inspect the process output. A container remains alive only while its configured main process is running. A shell script that starts a background service and exits will stop the container. Prefer an application process that stays in the foreground and propagates termination signals.

Exit code 127 usually indicates that the command was not found. Confirm the executable exists in the final image and that the entrypoint uses the correct path. Exit code 137 often indicates a forced kill and can be caused by an out-of-memory condition, although the surrounding runtime events should be checked before concluding that memory is the cause.

## Compose service networking

Services in the same Compose project can resolve one another by service name on the default network. An API service should call `http://qdrant:6333` or `postgres:5432` from inside its container. It should not use a host-published port or `localhost` for a sibling service. Published ports are primarily for traffic entering from the host.

If name resolution fails, confirm both services are attached to a common network and that the request uses the Compose service name rather than `container_name`. Inspect the fully rendered configuration with `docker compose config`; this reveals environment interpolation and merged override files. Network changes may require recreating containers, but deleting volumes is unrelated and risks data loss.

## Health checks and dependencies

A running container is not necessarily ready to serve requests. A health check should test a cheap operation that represents readiness, use a realistic interval and timeout, and allow enough startup time. Health checks must use tools actually present in the image. Minimal images may not contain `curl`; an application-native check can be more reliable.

Dependency ordering helps reduce startup races but applications should still tolerate brief downstream unavailability. Use bounded retries, explicit timeouts, and clear failure logs. Avoid long fixed sleeps because they are slow when dependencies are ready and unreliable when startup takes longer.

## Volumes and permissions

Named volumes persist after containers are removed. Rebuilding an image does not reset data stored in a volume. Inspect the mount configuration before deleting anything. Permission failures can occur when a process runs as a non-root user and the mounted directory is owned by another numeric user ID. Fix ownership deliberately in image or deployment setup rather than running the entire service as root.

## Image and build cache issues

Confirm the expected Dockerfile and build context. The build context controls which files can be copied, while `.dockerignore` controls exclusions. If a dependency file changed but a cached layer was reused unexpectedly, inspect the build output and rebuild the affected layer. Avoid routinely disabling all cache; good layer ordering makes builds both reproducible and fast.

