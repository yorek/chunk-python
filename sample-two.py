from dotenv import load_dotenv
from chunkipy import TextChunker, TokenEstimator
from transformers import AutoTokenizer
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

print("Loading text...")
with open("./ai-gov-executive-order.txt", encoding='utf-8') as f:
    text = f.read()

print(f"Num of chars: {len(text)}")

print("Splitting...")
text_chunker = TextChunker(512, tokens=True, overlap_percent=0.3)
chunks = text_chunker.chunk(text)
print(f"Number of chunks {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i:000}, Length: {len(chunk)}")

print("Chuck 0:")
print(chunks[0])