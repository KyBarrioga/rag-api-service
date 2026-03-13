import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_SIZE = 1024
CHUNK_SIZE = 768
CHUNK_OVERLAP = 80
COLLECTION_NAME = "docs"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def get_qdrant_client() -> QdrantClient:
    mode = os.getenv("QDRANT_MODE", "local").strip().lower()

    if mode == "server":
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        return QdrantClient(host=host, port=port)

    if mode == "url":
        url = os.getenv("QDRANT_URL")
        if not url:
            raise ValueError("QDRANT_URL must be set when QDRANT_MODE=url")

        api_key = os.getenv("QDRANT_API_KEY")
        return QdrantClient(url=url, api_key=api_key)

    if mode == "local":
        path = os.getenv("QDRANT_PATH", "./qdrant_data")
        return QdrantClient(path=path)

    raise ValueError(
        "Unsupported QDRANT_MODE. Use one of: local, server, url."
    )


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )


def create_embedding(text: str) -> list[float]:
    # return client.embeddings.create(
    #     model=EMBEDDING_MODEL,
    #     input=text,
    # ).data[0].embedding
    return embed_model.get_text_embedding(text)


def load_pdf_chunks(documents_dir: Path) -> list[dict[str, object]]:
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {documents_dir.resolve()}")

    reader = SimpleDirectoryReader(
        input_files=[str(path) for path in pdf_paths])
    documents = reader.load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE,
                              chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    loaded_chunks: list[dict[str, object]] = []

    for chunk_index, node in enumerate(nodes, start=1):
        text = node.get_content().strip()
        if not text:
            continue

        metadata = node.metadata
        loaded_chunks.append(
            {
                "text": text,
                "source": metadata.get("file_name", "unknown"),
                "page": metadata.get("page_label", "unknown"),
                "chunk": chunk_index,
            }
        )

    if not loaded_chunks:
        raise ValueError(
            f"PDF files were found in {documents_dir.resolve()}, but no extractable text was detected."
        )

    return loaded_chunks
