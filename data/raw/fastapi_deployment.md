# FastAPI Deployment Guide

This guide describes practical checks for running a FastAPI application behind a process manager, container platform, or reverse proxy.

## Application startup

Import the ASGI application with a stable module path such as `app.main:app`. Run startup validation in development and CI so missing modules or invalid settings fail before deployment. Configuration should come from environment variables or an injected settings file, while secrets must stay out of source control and logs.

Use Uvicorn directly for local development. In production, choose worker count from measured CPU and memory usage rather than copying a large default. Every worker loads its own Python process, embedding model, connection pools, and caches. A memory-heavy retrieval service may need fewer workers than a lightweight JSON API.

## Health endpoints

A liveness endpoint should answer quickly and prove that the web process can serve requests. It should not depend on Qdrant, the LLM provider, or another remote service; otherwise a downstream outage can cause the platform to restart healthy application processes repeatedly. A readiness check may validate required dependencies and should be exposed separately when the deployment platform can distinguish readiness from liveness.

Do not include secrets or detailed stack traces in health responses. Keep the liveness contract stable, for example `GET /health` returning HTTP 200 and `{"status":"ok"}`.

## Reverse proxy and timeouts

When a proxy terminates TLS, forward the original scheme and client information only from trusted proxies. Configure request timeouts consistently across the load balancer, proxy, Uvicorn, Qdrant client, and LLM client. The outer timeout must allow the inner operation to finish or fail cleanly. Unlimited timeouts can consume workers indefinitely.

For large request bodies, configure limits at the earliest network boundary and validate again in the application. RAG questions should have a reasonable maximum length. Return a structured client error for invalid input and a generic service-unavailable response for downstream failures, while preserving the detailed exception in server logs.

## Graceful shutdown

Deployment systems send a termination signal before stopping a container. Allow enough grace time for in-flight requests, stop accepting new work, and close reusable clients. The main process must receive signals directly; wrapper shell scripts should use `exec` or correctly forward signals. Keep shutdown bounded so deployments do not hang.

