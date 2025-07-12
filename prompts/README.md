# System Prompts for LocalRAG

This directory contains predefined system prompts for different use cases.

## Available Prompts

### `general_rag_assistant.txt`
Comprehensive assistant for document-based Q&A with context awareness. Best for general knowledge base applications.

### `technical_assistant.txt`
Specialized for technical documentation, code examples, and API references. Ideal for developer documentation systems.

### `simple_assistant.txt`
Basic assistant prompt for simple conversational interactions.

## Usage

### Method 1: Direct Text in .env
```bash
LLM_SYSTEM_PROMPT="Your system prompt text here"
```

### Method 2: File Reference in .env
```bash
LLM_SYSTEM_PROMPT_FILE=prompts/general_rag_assistant.txt
```

### Method 3: Multi-line in .env
```bash
LLM_SYSTEM_PROMPT="You are a helpful assistant.

Provide clear and accurate responses.
Use context when available."
```

## Creating Custom Prompts

1. Create a new `.txt` file in this directory
2. Write your system prompt
3. Reference it in your `.env` file using `LLM_SYSTEM_PROMPT_FILE`

## Best Practices

- Keep prompts focused and specific to your use case
- Include clear instructions for handling context and uncertainty
- Specify the desired tone and response format
- Test prompts with various types of queries
- Version control your prompts for different environments