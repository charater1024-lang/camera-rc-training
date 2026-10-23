# Security and operational boundaries

This is an offline research repository. No release is approved for controlling a real vehicle. Runtime examples default to dry-run; model validation flags must not be treated as a substitute for physical testing.

## Reporting

For a vulnerability, use GitHub's private vulnerability reporting feature if it is enabled for this repository. Otherwise open a minimal issue asking the maintainer for a private reporting channel. Do not post credentials, personal paths, vehicle logs, source images, labels, or a working exploit that could affect an exposed service. No monitored security email or response-time guarantee is claimed.

## Local execution

- Run code in an isolated environment without administrative privileges. Review commands that explicitly start training, queues, downloads, services, camera capture or serial communication.
- Load checkpoints only from a trusted source and verify the recorded SHA256 first. PyTorch/Ultralytics serialized models are not a safe format for arbitrary untrusted uploads. A hash verifies identity, not the trustworthiness or licensing of a file.
- Archive intake validates selected archives before extraction. Keep original ZIPs and failed outputs. Use a new output directory and do not disable path traversal, link, CRC or provenance checks to force acceptance.
- The progress server is intended for `127.0.0.1` only. It exposes local run metadata and has no authentication intended for internet use. Do not reverse proxy or bind it publicly; redact paths before sharing screenshots/logs.
- Do not commit `configs/local`, data, models, raw manifests/contracts, logs, tokens or environment files. `.gitignore` does not remove files already tracked by Git; inspect the final tracked-file list and history before publication.
- Downloads may require credentials and acceptance of source-specific terms. Supply credentials outside version control. A downloader or mirror does not grant redistribution rights.

## Tests and deployment

The CPU CI uses synthetic fixtures and mocked external operations. It does not connect to a vehicle, use CUDA, fetch datasets/weights or start recurring jobs. Dependency installation still accesses package registries. Software tests do not validate emergency braking, physical actuation or collision avoidance. The systemd and serial examples are explicit research integrations, not services that should be enabled by installation.

Known performance regressions and dataset limitations are documented in [RESULTS](docs/RESULTS.md) and [LIMITATIONS](docs/LIMITATIONS.md). A new checkpoint must retain its provenance and be evaluated before any separate deployment review. This project does not automatically replace a deployed model.
