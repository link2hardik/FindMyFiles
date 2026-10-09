# Application Observability

Observability helps a team understand what an application is doing from its outputs. The three common signals are logs, metrics, and traces. Logs describe events, metrics summarize behavior over time, and traces connect work across services.

Useful service metrics include request count, error rate, latency, queue depth, task duration, and ingestion failures. Logs should include a request or task identifier without exposing file contents, API keys, or personal data.

A dashboard is most useful when it supports a decision. Alerts should identify symptoms that need action, such as sustained errors or a growing worker queue. For a small application, structured logs and a few meaningful counters are a reasonable starting point.

An ingestion pipeline benefits from stage-level measurements. Record extraction duration, chunk count, embedding duration, queue wait time, and final task status. A document ID and task ID can connect these events without logging the document contents. For retrieval, track query latency, requested k, result count, and failures, but avoid recording sensitive query text unless the data policy explicitly permits it.

Metrics should have clear names and units. A histogram of request latency is more useful than an average alone because a small number of slow requests may affect users. Logs should be structured enough to filter by service and severity. Traces become especially valuable when a frontend request creates a queued task that later calls an external OCR or language-model provider.