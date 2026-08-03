# Security Policy

## Supported versions

Security fixes are provided for the latest published minor release and the current `main` branch.

| Version | Supported |
| --- | --- |
| `0.2.x` | Yes |
| `< 0.2` | No |

This laboratory is an educational reference implementation, not a hosted service. Its threat model and trust boundaries are documented in [`docs/threat-model.md`](docs/threat-model.md).

## Reporting a vulnerability

Please report suspected vulnerabilities through [GitHub private vulnerability reporting](https://github.com/ashirimi1019/knowledge-augmentation-lab/security/advisories/new). Do not include exploit details, credentials, personal data, or unpublished vulnerability information in a public issue or discussion.

Include, when available:

- The affected version or commit.
- The relevant component and public boundary.
- Reproduction steps or a minimal proof of concept.
- The expected and observed behavior.
- The potential impact and any known mitigations.

If GitHub private vulnerability reporting is unavailable, open a public issue containing no sensitive details and request a private contact channel.

## Response expectations

The project aims to:

- Acknowledge a complete report within three business days.
- Provide an initial triage decision within seven business days.
- Share progress updates at least every fourteen days while remediation is active.
- Coordinate disclosure after a fix and release plan are ready.

These are response targets, not guaranteed resolution deadlines. Timing depends on severity, reproducibility, and maintainer availability.

## Researcher expectations

Good-faith research should avoid privacy violations, destructive testing, denial of service, social engineering, and access to data beyond what is necessary to demonstrate the issue. Allow a reasonable remediation period before public disclosure.
