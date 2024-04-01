# LangChain-RAG-LangServe
🦜
## Table of Contents

- [Introduction](#introduction)
- [Idea](#idea)
- [Usage](#usage)
- [Features](#features)
- [Contact](#contact)

## Introduction

LangChain RAG (Retrieval Augmented Generation) method in LangServe REST API Logic. Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts ([reference](https://aws.amazon.com/what-is/retrieval-augmented-generation/)).

## Idea

The first idea was simple, vectorize chunks of given text to generate a context and get an answer from OpenAI only based on that context. Thanks to LangChain this is **really easy**. The chain logic would be like this:

```
context_chain = ConversationalRetrievalChain.from_llm(
    llm=model,  <- Choose your OpenAI Model
    chain_type="stuff",  <- This option will be modified in the future
    retriever=retriever,  <- Return docs from vector store (context)
    memory=memory, <- Store question and answers in memory as buffer, doc, database, etc
    condense_question_prompt=prompt1  <- Prompt
)
```

Officially the prompt for this use case is this one:

```
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
---
Chat History: {chat_history}
---
Question: {question}
```

Having a chat history makes sense; it would follow some structure in each answer. However, this approach would function solely as a document searcher, and the context would need to be quite extensive. Combining context with chat history results in more tokens per OpenAI call. To create a more flexible solution, the final idea transformed into two chains. The first chain searches within the given context, and then the second chain rephrases the answer if the initial question lacked sufficient context. One notable drawback is that the chat history may introduce noise between the two prompts. However, having a trained GPT with additional context provides more options than a capped GPT that only knows about our given context (especially when dealing with small contexts). Take a look at the order of the [chains](https://github.com/SchneiderSix/LangChain-RAG-LangServe/blob/main/API/main.py#L116).

## Usage

```
# React.js app
cd .\Dummy Front
npm run start
# FastAPI app
cd .\API
py .\main.py
```

Open [http://localhost:4200](http://localhost:3000) with your browser to see the result.

## Features

- [x] Limit API calls, thank you [Jeremiah ](https://github.com/jeremiahtalamantes/fastapi-rate-limiter?tab=readme-ov-file)
- [x] API CORS
- [x] Document load for txt files
- [x] RAG using langChain
- [x] Classic Conversational Retrieval Chain example
- [x] Multichain example
- [x] Dummy call python script
- [x] Dummy react.js app
- [x] API [ready](https://github.com/SchneiderSix/LangChain-RAG-LangServe/tree/main/API%20Deployment%20Docker%2BPoetry) for deployment

## Contact

Ask me anything :smiley:

[Juan Matias Rossi](https://www.linkedin.com/in/jmrossi6/)
