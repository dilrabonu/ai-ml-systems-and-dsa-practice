import asyncio
from multiprocessing import current_process

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

# stream, async, generator
import asyncio
async def server_streami():
    """ It shows splitting info from server"""
    splits = ["Salom", "-this", "streaming", "through", "coming"]
    for split in splits:
        await asyncio.sleep(0.5)
        yield split
async def response_model(request):
    tokens = ["Python", "-", "is ", "wonderful", "programming" "language", "."]
    for token in tokens:
        await asyncio.sleep(0.3)
        yield token
    
async def main():
    request = "Tell about the Python?"
    print(f"Question: {request}")
    print(f"Answer: ", end="", flush=True)
    async for token in response_model(request):
        print(token, end="", flush=True)

    print()

asyncio.run(main())

# Iterator
class Count:
    def __init__(self, last):
        self.current = 0
        self.last = current

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.last:
            raise StopIteration
        value = self.current
        self.current += 1
        return value