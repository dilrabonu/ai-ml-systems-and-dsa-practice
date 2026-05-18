import asyncio

async def call_llm(client, prompt):
    response = await client.message.create(
        model="claude-opus-4-7",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],

    )
    return response.content[0].text

async def summarize_documents(documents):
    client = AsyncAnthropic()
    tasks = [
        call_llm(client, f"Summarize: {doc}")
        for doc in documents
    ]
    summaries = await asyncio.gather(*tasks)
    return summaries

my_docs = ["doc1", "doc2", "doc3"]
summaries = asyncio.run(summarize_documents(my_docs))
print(summaries)


def file_stream(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()