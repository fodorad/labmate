from labmate.ask.lc import LibraryRetriever, RecordedEmbeddings


def test_embeddings_and_retriever(indexed):
    embeddings = RecordedEmbeddings(indexed.embedder)
    assert len(embeddings.embed_query("blink")) == 64
    assert len(embeddings.embed_documents(["a", "b"])) == 2
    retriever = LibraryRetriever(index=indexed.index, embedder=indexed.embedder, tiers=(1,), k=2)
    docs = retriever.invoke("CEW ZJU datasets")
    assert len(docs) == 2 and all(d.metadata["tier"] == 1 for d in docs)
    assert docs[0].id and "score" in docs[0].metadata
    lexical = LibraryRetriever(index=indexed.index, embedder=indexed.embedder, method="bm25")
    assert "CEW" in lexical.invoke("CEW ZJU")[0].page_content
