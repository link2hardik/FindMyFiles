# Simple Threat Modeling

Threat modeling asks what an application protects, who might attack it, and which paths could be abused. The process does not require a complicated tool. A diagram of users, services, data stores, and trust boundaries is a useful starting point.

For a file-search service, important assets include uploaded files, extracted text, vector embeddings, API credentials, and user identity data. Threats may include unauthorized downloads, malicious files, prompt injection in document text, denial of service, and credential theft.

Controls should match the threat. File type validation, upload limits, authentication, authorization checks, malware scanning, rate limits, encrypted transport, and safe logging each address different risks. The model should be updated when the architecture changes.

Document-search systems also have content-level risks. An uploaded document may contain instructions intended to manipulate a language model, confidential text that should not be returned to another user, or a file designed to consume excessive CPU during parsing. Retrieval permissions must be applied before results are displayed, and extracted text should be treated as untrusted input rather than as application instructions.

Threat modeling works best when each risk has an owner and a proposed control. Rank issues by likelihood and impact, then test the most important controls with a small abuse case. Examples include uploading a file with a misleading extension, requesting another user's file ID, sending a very large query, or forcing repeated OCR work. Revisit the model after adding external APIs or new storage.