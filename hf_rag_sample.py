import ollama

#read file
dataset = []
with open("/Users/ravicharanketha/Documents/GitHub/LLM-Practice/cat-facts.txt", 'r') as file:
   dataset = file.readlines()
   print(f'Loaded {len(dataset)} entries')

#vector_db setup
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

vector_db = []

def add_chunk_to_db(chunk):
   embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
   vector_db.append((chunk, embedding))

#each line is considered a chunk
for i, chunk in enumerate(dataset):
   add_chunk_to_db(chunk)
   if (i+1)%50 == 0:
       print(f'added chunk {i+1}/{len(dataset)} to db')


def cosine_similarity(a, b):
   dot_product = sum([x * y for x, y in zip(a, b)])
   norm_a = sum([x ** 2 for x in a]) ** 0.5
   norm_b = sum([x ** 2 for x in b]) ** 0.5
   return dot_product/(norm_a * norm_b)

#retrieve
def retrieve(query, top_n):
    print(query)
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_emb, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

#generate
input_query = input('Ask me a question:')
retrieved_knwldge = retrieve(input_query, 5)

print("Retrieval")
for chunk, similarity in retrieved_knwldge:
   print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''Use only the  following pieces of context to answer the question.
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knwldge])}
'''

#chat
stream = ollama.chat(
   model=LANGUAGE_MODEL,
   messages=[
      {'role': 'system', 'content': instruction_prompt},
      {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
   print(chunk['message']['content'], end='', flush=True)