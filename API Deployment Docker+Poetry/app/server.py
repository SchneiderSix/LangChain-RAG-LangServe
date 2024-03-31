from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langserve import add_routes
import uvicorn
# from dotenv import load_dotenv
import os
import time
from operator import itemgetter
from starlette.middleware.base import BaseHTTPMiddleware
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# load_dotenv()

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


class TokenBucket:
    def __init__(self, capacity, refill_rate):
        # Initialize the token bucket
        self.capacity = capacity  # Maximum number of tokens in the bucket
        # Rate at which tokens are added to the bucket per second
        self.refill_rate = refill_rate
        self.tokens = capacity  # Start with the bucket full
        self.last_refill = time.time()  # Record the time when the bucket was last refilled

    def add_tokens(self):
        # Add tokens to the bucket based on the time elapsed since the last refill
        now = time.time()
        if self.tokens < self.capacity:
            # Calculate the number of tokens to add
            tokens_to_add = (now - self.last_refill) * self.refill_rate
            # Update the token count, ensuring it doesn't exceed the capacity
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now  # Update the last refill time

    def take_token(self):
        # Attempt to take a token from the bucket
        self.add_tokens()  # Ensure the bucket is refilled based on the elapsed time
        if self.tokens >= 1:
            self.tokens -= 1  # Deduct a token for the API call
            return True  # Indicate that the API call can proceed
        return False  # Indicate that the rate limit has been exceeded


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bucket: TokenBucket):
        super().__init__(app)
        self.bucket = bucket  # Initialize the middleware with a token bucket

    async def dispatch(self, request: Request, call_next):
        # Process each incoming request
        if self.bucket.take_token():
            # If a token is available, proceed with the request
            return await call_next(request)
        # If no tokens are available, return a 429 error (rate limit exceeded)
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# Initialize the token bucket with 3 tokens capacity and refill rate of 1 tokens/second
bucket = TokenBucket(capacity=3, refill_rate=1)

app = FastAPI(
    title="RGA Langchain Server Test",
    version="1.0",
    description="Now ChatGPT knows about Promptior!"
)

allowed_url = os.environ['FRONT_URL']
print('------\n'+allowed_url+'\n------')

# Enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add rate limiter
app.add_middleware(RateLimiterMiddleware, bucket=bucket)

# Get data from promptior.txt
loader = TextLoader(file_path="/code/app/promptior.txt", encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(data, embeddings)
retriever = vector_store.as_retriever()

# Specify llm
model = ChatOpenAI(
    temperature=0.5,
    model="gpt-3.5-turbo"
)

# Prompts for chains
"""Prompt for context_chain as ConversationalRetrievalChain with memory
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
---
Chat History: {chat_history}
---
Question: {question}
"""

template1 = """
Answer the question based only on the following context: {context}
---
Question: {question}
"""
prompt1 = ChatPromptTemplate.from_template(template1)

template2 = """
If base answer lacks of context then answer the question.
---
Base Answer: {answer_from_context}
---
Question: {question}
"""
prompt2 = ChatPromptTemplate.from_template(template2)

"""
# Create buffer for memory
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)

# context_chain as ConversationalRetrievalChain with memory
context_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    condense_question_prompt=prompt1
)
"""
# First chain, answer based on given context
context_chain = (
    {
        # Useful for searching many docuemnts, lambda x: retriever.get_relevant_documents(x["question"]),
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt1
    | model
    | StrOutputParser()
)

"""Second chain, 
answer based on context_chain's answer, 
if it didn't have enough context then 
a new answer will be made
"""
re_do_chain = (
    {
        "answer_from_context": context_chain,
        "question": itemgetter("question")
    }
    | prompt2
    | model
    | StrOutputParser()
)

# Print chain graph
re_do_chain.get_graph().print_ascii()

add_routes(
    app,
    re_do_chain,
    path="/qp"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
