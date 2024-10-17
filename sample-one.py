from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    openai_api_version="2023-05-15",
    show_progress_bar=True
)

print("Loading text...")
with open("./ai-gov-executive-order.txt", encoding='utf-8') as f:
    state_of_the_union = f.read()

print("Splitting...")
text_splitter = SemanticChunker(embeddings)
chunks = text_splitter.create_documents([state_of_the_union])

print(f"Number of chunks {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i:000}, Length: {len(chunk.page_content)}")

print("Chuck 0:")
print(chunks[0].page_content)