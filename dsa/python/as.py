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

# List in ML in epoch, batch, gather all
losses = []
for epoch in range(50):
    loss = train.step()
    losses.append(loss)

# tuple immutable, order in ML tensor with shape
picture.shape # (224, 224, 3)- height, width, channels
batch.shape    # ( 32, 224, 224, 3) - 32 picture

for picture, batch in train_loader:
    preds = model(picture)

# several values in function
def info_give(data):
    return train_split, test_split

train, test = info_give(all_data)
# !!! tuple can be dict but list can't
values = {(0, 0): 1.5, (0, 1): 2.3} # tuple is key

# Set in Ml to switch off repetition samples
clean_text = set(all_text)
print(f"{len(all_text)} from {len(clean_text)} left.")

# Nlp word/unique category
words = set()
for sentence in text:
    words.update(sentence.split())  # unique words gathered here
print(f"Dictionary size: {len(words)}")

# fast check in set
stop_words = {"and", "with", "for", "too"} # set
clean = [s for s in words if s not in stop_words] # check with 0(1)
print(f"Clean size: {len(clean)}")

# dict in ML
# configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "hidden_size": 256,
}
loss = train(config["learning_rate"], config["batch_size"], config["epochs"], config["hidden_size"])

# give the name to metrics
results = {
    "acc": 0.94,
    "loss": 0.087,
    "f1": 0.95,
}
