# Synthetic Evaluation Corpus

This corpus contains 24 synthetic Markdown documents across six topics, with four related but distinct documents per topic.

## Topics

- `cloud_storage`
- `cybersecurity`
- `project_management`
- `renewable_energy`
- `nutrition`
- `urban_gardening`

Use `manifest.json` as the source of stable document IDs and topic metadata. The filenames are identifiers for the fixture files, not relevance labels.

The documents are intentionally similar within each topic so evaluation queries can test semantic distinctions. For example, cloud-storage documents cover object storage, CDNs, backups, and cost rather than repeating one passage.

PDF, image/OCR, and audio fixtures can be added later with new manifest entries. Keep their document IDs stable and avoid changing existing text while comparing retrieval configurations.
