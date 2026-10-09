# Effective Code Reviews

A code review should focus on correctness, security, maintainability, and missing tests. Reviewers should first understand the intended behavior, then inspect the changed code and its effects on nearby interfaces.

Specific comments are more useful than vague criticism. A good comment explains the risk, gives a concrete example, or asks a question about an observable behavior. Not every preference needs to block a change.

Small pull requests are easier to review. A description should mention the problem, the approach, test commands, and any known limitations. Automated checks handle formatting and repeatable tests so human reviewers can spend time on design and behavior.

Reviewers should inspect boundaries between modules, not only the changed lines. A small configuration change can alter startup behavior, environment-variable handling, or data compatibility. For ingestion code, pay attention to file handles, encoding errors, empty extraction results, cleanup on failure, and whether metadata still maps a result to its source document.

Useful review questions include: What happens when the external service is unavailable? Can the change expose secrets in logs or images? Is the behavior covered by a focused test? Does the change preserve backwards compatibility? A reviewer should distinguish a correctness defect from a follow-up improvement so the pull request remains actionable.