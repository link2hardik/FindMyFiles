# A Safe Release Process

A release process reduces the chance that a change will surprise users. A common sequence is to create a small change, run automated checks, review it, build an immutable artifact, deploy it, and verify health after deployment.

Version tags make an artifact easy to identify. Deploying an exact image tag or digest is safer than deploying a mutable latest tag because the running code can be traced back to a commit.

A release should have a rollback plan. Keeping the previous image available and preserving persistent data makes rollback faster. Health checks should verify the application is ready, while logs and metrics help diagnose a failed release.

A release candidate should be tested with the same dependency lock file and Dockerfile used in production. Smoke tests can verify that the API starts, Redis is reachable, the worker can receive a task, and the frontend can contact the backend. They do not need to exercise every expensive language-model path, but they should detect missing commands, bad environment variables, and incompatible service URLs.

Deployment records should include the version, commit, image digest, configuration changes, and verification result. A rollback is not complete until the old version is healthy and persistent data remains usable. If a release changes a database or index format, plan the migration and rollback direction before deployment instead of discovering the compatibility problem during an outage.