# Practical Backup Policy

A backup policy should state what is copied, how often it is copied, where copies are stored, and how restoration is tested. Copying data is not enough if nobody has confirmed that the copy can be restored.

A small application can begin with daily snapshots and a weekly off-site copy. Keep several historical versions so an accidental deletion or corrupted index is not immediately replicated everywhere. Encrypt backups when they contain uploaded files or personal information.

The recovery point objective describes how much recent data may be lost. The recovery time objective describes how quickly service should return. Database records, uploaded files, and search indexes may need different backup schedules because they have different rebuild costs.

These objectives determine backup frequency and restore procedures. A vector index may be rebuildable from source files, while an uploaded file may be impossible to recreate, so the original objects deserve stronger protection than derived search data. A database backup should use a consistency-aware method rather than copying a live database file while it is being modified.

A practical restore drill creates an isolated environment, restores the database and source files, rebuilds any derived index, and runs health checks plus representative searches. Keep several historical versions so corruption discovered later is not immediately replicated everywhere. Encrypt backups that contain personal information, protect encryption keys separately, and document who is responsible when a backup job fails.