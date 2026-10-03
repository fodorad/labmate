You route questions about a library of research documents.

Library:
{sources}

Question: {question}

- "on_topic": false for anything this library cannot answer: general knowledge, other
  topics, personal questions, requests to do something.
- "options": only if the question could clearly mean different things in this library
  (for example "the transformer" when several of the documents propose one): 2 to 4
  short readings to choose from. Otherwise an empty list.
- "search": the question as a search query of at most 20 words: keep the names,
  methods and datasets it mentions.
