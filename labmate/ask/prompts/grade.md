You judge search results for a question about a library of research documents.

Whole question: {question}
Search query: {query}

Search results:
{evidence}

- "relevant": the ids of the results that contain information that helps answer the
  search query (copy the ids exactly; leave out results that only share words with it).
- "sufficient": true only if the relevant results together answer the search query.
- "missing": if not sufficient, what information is still missing, in one short phrase.
