# Offline operation and error handling

CondensateLab CPU simulations, frame generation and GIF export run locally. Keep the complete Windows folder, including `_internal`. Microsoft Edge WebView2 Runtime is a desktop prerequisite. A machine without WebView2 must install it before offline use. No Python or CUDA installation is required for the packaged release.

The executable was tested with `--self-test report.json --self-test-models --offline-check`. A Python audit hook denies external socket connections and permits loopback. All three computations completed and produced animated GIFs. See offline-verification.json. This test checks the packaged Python backend and served assets; it does not disable the computer's physical network adapter or audit WebView2's own background processes.

Test cases: phase N=32, steps=80; light N=32, on/off=5, cycles=1; transport duration=30. These are smoke tests, not full-size performance benchmarks or concurrent-user load tests.

Messages distinguish input, unstable numerical parameters, memory, disk space, folder permissions, missing files, runtime dependencies, interrupted sessions, queue capacity, authentication and expired results. Unknown calculation failures retain a neutral message and technical detail. Local service loss is not described as an Internet outage.

Results expire after 24 hours. Save exports before expiry. History is browser-session specific and does not synchronize between devices.
