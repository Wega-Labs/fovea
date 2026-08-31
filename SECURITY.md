# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub Security Advisories][report]. Do not
open a public issue, discussion, or pull request containing exploit details, sensitive camera
data, or a model-integrity bypass.

Include the affected version or commit, platform, reproduction steps, impact, and any suggested
mitigation. Remove faces, frames, credentials, and other personal data before attaching evidence.

Maintainers aim to acknowledge a report within five working days. We will coordinate validation,
remediation, disclosure timing, and credit with the reporter. Response and release timing depend
on severity and the affected platforms; status updates will be provided while a report remains
open.

## In scope

- camera or landmark data retained, exposed, or transmitted contrary to Fovea's local-only model;
- model download, pinning, or checksum-integrity bypasses;
- command or control-message parsing that could enable code execution or unsafe file access;
- dependency or packaging behavior that makes installed applications execute untrusted content;
- any path that could exfiltrate frames, calibration data, or behavioral signals;
- permission, pause, or tracking-state failures with a meaningful security or privacy impact.

Ordinary accuracy bugs, feature requests, and crashes without a security or privacy impact belong
in the public issue tracker. When uncertain, report privately and let the maintainers triage it.

[report]: https://github.com/Wega-Labs/fovea/security/advisories/new
