# Release Notes — Nuclear-Entry Module Update

> Chinese original: [docs/zh/本次更新说明.md](docs/zh/本次更新说明.md)

Only the droplet transport simulation is kept. The old entry points, parameters, and backend implementations for the full-chain concentration field, the local nuclear-pore field, and compartment dynamics have been removed.

Queued jobs must be cancelled before restarting the computation service. Repeated submissions of a nuclear-entry task from the same browser are blocked while one is already queued or running; the number of pending jobs ahead is shown while waiting. Waiting tasks can be cancelled immediately.

This module follows the same operating model as the phase-separation app: CPU computation, progress preview, job history, and GIF/data export.
