# API Key Handling

API keys are credentials and should be treated like passwords. They belong in a secret manager, deployment environment, or local ignored environment file, not in source code, Docker image layers, issue comments, or log messages.

A service should receive a key at runtime and read it from an environment variable or secret-mounted file. Build steps should not use the key unless the build genuinely requires authenticated access. If a key appears in a repository, rotate it first and then remove the exposure.

Use separate keys for development, testing, and production. Limit each key's permissions and expiration where the provider supports it. Redacting values in error messages and request logs reduces accidental disclosure.

Build systems need special care because Docker layers, command output, cache exports, and failed workflow logs can preserve values longer than expected. Do not pass a credential as a Docker build argument merely because the application needs it at runtime. A workflow should grant only the permissions needed by its job and should avoid echoing environment variables while debugging.

Rotation should be a routine operation rather than an emergency-only procedure. Record which service uses each key, who owns it, when it expires, and how to replace it. If a provider reports a leaked key, revoke it immediately, inspect usage for unexpected requests, create a replacement, and update the runtime secret store. A placeholder such as `test-key` is safe only when it cannot authenticate anywhere.