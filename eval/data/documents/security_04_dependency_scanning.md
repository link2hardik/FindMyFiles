# Dependency Security Checks

Third-party libraries can introduce vulnerabilities even when application code is careful. Dependency scanning compares locked versions against published security advisories and helps identify packages that need an update.

A useful workflow checks dependencies during pull requests and schedules a regular update review. Lock files make builds reproducible, but they do not make old versions safe forever. Updates should be tested because a security fix can also change behavior.

Container scanning adds another layer by checking operating-system packages and image metadata. Findings should be prioritized by exploitability and whether the affected component is reachable. A small project benefits from fixing high-severity reachable issues first and documenting accepted low-risk findings.

Lock files make builds reproducible, but they do not make old versions safe forever. A useful process records the direct dependency that introduced a vulnerable transitive package and tests the smallest upgrade that resolves it. Remove unused dependencies when possible because every additional package increases the supply-chain and maintenance surface.

Container scanning should examine both operating-system packages and language packages. A finding in a development-only tool may be less urgent than a vulnerable parser reachable by an upload endpoint, but it should still be tracked. Pinning base images by reviewed version or digest improves repeatability. CI should fail on clearly defined high-severity policies rather than silently producing a supposedly trusted image.