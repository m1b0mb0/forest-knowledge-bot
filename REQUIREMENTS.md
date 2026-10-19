# Forest Archive — Requirements Summary

> A concise record of the requirements discovered in the conversation with the fictional client, **Myshka**, for the Forest Knowledge Bot take-home assignment. This is a summary of the discussion, not a verbatim transcript. Private archive links, credentials, and source documents are intentionally excluded.

## 1. Problem and goal

Forest residents keep advice and rules in a Google Drive folder named **«Лісовий архів»** and share additional documents with a Telegram bot. As the collection grows, finding reliable information becomes difficult.

**Goal:** allow any resident to ask a question in natural language and receive a concise answer grounded in the available documents, with a clear indication of the source.

## 2. Requirements expressed by the client

| ID | Requirement | Expected behavior |
|---|---|---|
| R-01 | Question answering | Answer users' questions using the contents of the documents, not merely file names. |
| R-02 | Source attribution | Name the source document for every document-based answer; cite all relevant documents when more than one is used. |
| R-03 | Initial archive | Read documents from the Google Drive folder **«Лісовий архів»** without requiring users to resend them in Telegram. |
| R-04 | User uploads | Accept PDF, DOCX and TXT documents sent via Telegram and make their information available to all bot users. |
| R-05 | Upload metadata | Remember who sent each Telegram document and when it was sent; make that information available with its source attribution. |
| R-06 | Persistent knowledge | Retain uploaded documents so they remain usable for later questions, including after an application restart. |
| R-07 | Current advice only | When a newer version of the same advice for the same place supersedes an older one, use the newer version rather than the outdated advice. |
| R-08 | Equal access | Every user can ask questions and submit documents. No admin or privileged user role is required. |

## 3. Decisions proposed by the developer and accepted by the client

The client explicitly left the following situations to the developer's judgment; these behaviors were proposed and agreed in the conversation.

| Situation | Agreed behavior |
|---|---|
| No supported answer in the documents | State that the information was not found. Do not invent facts or rely on the LLM's general knowledge as a substitute for documents. |
| Question is ambiguous about a place | Ask which place the user means **when multiple applicable places could lead to different answers**. Answer directly when the context is unambiguous. |
| A user uploaded a document by mistake | Allow the uploader to remove their **own** Telegram documents. Other users' uploads cannot be removed by that user, and removed documents are excluded from future answers. |
| Telegram document storage | Keep Telegram uploads in the application's own storage; do **not** copy them into Google Drive. The bot uses both collections when answering. |
| Google Drive archive integrity | Access the archive read-only. Do not add, edit, or delete files in the original Drive folder via the bot. |

## 4. Engineering choices and known limitations

These are implementation decisions, **not additional requests made by the client**. They keep the take-home assignment manageable and are described more fully in the project's [README](README.md).

- **Archive updates:** check/import the configured Drive folder at startup; real-time synchronization or webhooks are out of scope.
- **Version detection:** identify likely versions using the advice topic and place, supported by filenames and optional document metadata. Prefer explicit versions/dates where available. If documents cannot be matched confidently, do not automatically discard one. This is a heuristic; the client did not provide a reliable naming or versioning convention.
- **Retrieval:** select likely relevant documents and provide their actual contents to the LLM to construct a grounded response. Accuracy may depend on descriptive filenames; the repository README documents the retrieval limits.
- **File processing:** parse PDF, DOCX and text files. OCR for PDF pages without extractable text was added as an implementation enhancement, not as an originally requested feature; OCR accuracy is not guaranteed.
- **Storage:** persist Telegram uploads and metadata locally. This is distinct from the original Drive archive.
- **Scope boundaries:** no administrative dashboard, custom user roles, Drive write access, or continuous Drive monitoring.

## 5. Technical constraints from the employer

- Implement the Telegram bot in **Python**.
- Use an LLM available in the **Groq free tier**.
- Start the entire application with one **Docker / Docker Compose** command.
- Configure tokens, API keys and other secrets via **environment variables**; do not commit credentials.
- Deliver the result as a **GitHub repository**.

## 6. Acceptance scenarios

These examples turn the discovered requirements into checks; they are not claims that every case has been independently tested.

1. A question answerable from the initial Drive archive returns the correct facts **and the source document name**.
2. A question with no supporting information receives an honest *not found* response, rather than an invented answer.
3. Where different places have different rules, an underspecified question prompts the user to clarify the place.
4. A user uploads a valid PDF, DOCX or TXT; another user can ask about its content, and the answer identifies the document, sender and timestamp.
5. Uploaded content remains available after restarting the Docker container.
6. A user can remove their own uploaded document but cannot remove someone else's; removed content is no longer used.
7. When a clearly identifiable newer version replaces advice for the same topic and place, the older version does not influence subsequent answers.
8. The bot can be started using the instructions in README and the documented environment variables, without manually installing Python dependencies on the host.

## 7. Interpretation

The core requirement is **trustworthy document-based answers with source attribution**, not a general-purpose chatbot. The main business rules are shared access, preserved provenance, avoidance of unsupported answers, and the preference for current advice. Where the client did not specify implementation details, the project favors small, documented decisions over extra features.
