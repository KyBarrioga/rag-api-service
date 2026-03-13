from helper import (
    get_qdrant_client, ensure_collection, load_pdf_chunks,
    create_embedding
)
from pathlib import Path
from openai import OpenAI
from qdrant_client.models import PointStruct

COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_SIZE = 1024
DOCUMENTS_DIR = Path("documents")
CHUNK_SIZE = 768
CHUNK_OVERLAP = 80

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the answer is not in the context, say you do not have enough information. "
    "Cite the source file and page when possible."
)


def main() -> None:
    openai_client = OpenAI()
    qdrant = get_qdrant_client()

    try:
        ensure_collection(qdrant)
        documents = load_pdf_chunks(DOCUMENTS_DIR)
        print(
            f"Loaded {len(documents)} text chunks from {DOCUMENTS_DIR.resolve()}")

        points = []

        for i, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=i,
                    # vector=create_embedding(openai_client, str(doc["text"])),
                    vector=create_embedding(str(doc["text"])),
                    payload={
                        "text": doc["text"],
                        "source": doc["source"],
                        "page": doc["page"],
                        "chunk": doc["chunk"],
                    },
                )
            )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        query = "Can you give me the syntax to create a loop in Python?"
        query_embedding = create_embedding(query)

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=5,
        ).points

        context_blocks = []
        for i, result in enumerate(results, start=1):
            payload = result.payload
            context_blocks.append(
                f"""[Source {i}]
        file: {payload.get("source")}
        page: {payload.get("page")}
        chunk: {payload.get("chunk")}
        text:
        {payload.get("text")}"""
            )

        context_text = "\n\n".join(context_blocks)

        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
                    Question:
                    {query}

                    Context:
                    {context_text}

                    Instructions:
                    - Answer only from the context above.
                    - Be concise and accurate.
                    - If the context is insufficient, say so.
                    - Include source citations like pdf name and page if available.
                    """.strip(),
                },
            ],
        )
        print(response.choices[0].message.content or "")

    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
