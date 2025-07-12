"""
LLM Service for chat response generation with multiple provider support.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator

import httpx
from loguru import logger

from core.config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None, 
                              temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def stream_response(self, prompt: str, max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 endpoint: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        # Use custom endpoint if provided, otherwise default to OpenAI
        base_url = endpoint or "https://api.openai.com/v1"
        self.endpoint = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens or settings.llm_max_tokens,
                    "temperature": temperature or settings.llm_temperature,
                    "stream": False
                }
                
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream_response(self, prompt: str, max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens or settings.llm_max_tokens,
                    "temperature": temperature or settings.llm_temperature,
                    "stream": True
                }
                
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Use base URL for models endpoint
                base_url = self.endpoint.replace("/chat/completions", "")
                models_url = f"{base_url}/models"
                response = await client.get(
                    models_url,
                    headers=self.headers
                )
                return response.status_code == 200
        except:
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229",
                 endpoint: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens or settings.llm_max_tokens,
                    "temperature": temperature or settings.llm_temperature,
                    "stream": False
                }
                
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                return data["content"][0]["text"]
                
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def stream_response(self, prompt: str, max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens or settings.llm_max_tokens,
                    "temperature": temperature or settings.llm_temperature,
                    "stream": True
                }
                
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Simple health check
                test_payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1
                }
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=test_payload
                )
                return response.status_code == 200
        except:
            return False


class OllamaProvider(LLMProvider):
    """Local Ollama provider"""
    
    def __init__(self, model: str = "llama2", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
    
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature or settings.llm_temperature,
                        "num_predict": max_tokens or settings.llm_max_tokens
                    }
                }
                
                response = await client.post(
                    f"{self.endpoint}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                return data["response"]
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def stream_response(self, prompt: str, max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature or settings.llm_temperature,
                        "num_predict": max_tokens or settings.llm_max_tokens
                    }
                }
                
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    text = data["response"]
                                    if text:
                                        yield text
                                    
                                    # Check for completion
                                    if data.get("done", False):
                                        break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.endpoint}/api/tags")
                return response.status_code == 200
        except:
            return False


class LLMService:
    """Main LLM service that manages different providers"""
    
    def __init__(self):
        self.provider: Optional[LLMProvider] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the LLM provider based on configuration"""
        if not settings.enable_llm_chat:
            logger.info("LLM chat is disabled in configuration")
            return
        
        try:
            if settings.llm_provider == "openai":
                if not settings.llm_api_key:
                    logger.warning("OpenAI API key not provided")
                    return
                
                self.provider = OpenAIProvider(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    endpoint=settings.llm_endpoint
                )
                
            elif settings.llm_provider == "anthropic":
                if not settings.llm_api_key:
                    logger.warning("Anthropic API key not provided")
                    return
                
                self.provider = AnthropicProvider(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    endpoint=settings.llm_endpoint
                )
                
            elif settings.llm_provider == "ollama":
                self.provider = OllamaProvider(
                    model=settings.llm_model,
                    endpoint=settings.llm_endpoint or "http://localhost:11434"
                )
                
            else:
                logger.error(f"Unsupported LLM provider: {settings.llm_provider}")
                return
            
            # Mark as initialized (assume provider is available)
            self.initialized = True
            logger.info(f"LLM service initialized with {settings.llm_provider} provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            self.provider = None
    
    async def generate_response(self, prompt: str, context_chunks: List[Dict]) -> str:
        """Generate a response using the LLM"""
        if not self.initialized or not self.provider:
            raise ValueError("LLM service not initialized or not available")
        
        try:
            # Build context-enhanced prompt
            enhanced_prompt = self._build_chat_prompt(prompt, context_chunks)
            
            response = await self.provider.generate_response(
                enhanced_prompt,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system_prompt=settings.get_system_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise
    
    async def stream_response(self, prompt: str, context_chunks: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream a response using the LLM"""
        if not self.initialized or not self.provider:
            raise ValueError("LLM service not initialized or not available")
        
        try:
            # Build context-enhanced prompt
            enhanced_prompt = self._build_chat_prompt(prompt, context_chunks)
            
            async for chunk in self.provider.stream_response(
                enhanced_prompt,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system_prompt=settings.get_system_prompt
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Failed to stream LLM response: {e}")
            raise
    
    def _build_chat_prompt(self, user_message: str, context_chunks: List[Dict]) -> str:
        """Build a chat prompt with context for the LLM"""
        
        # Build context section
        context_parts = []
        if context_chunks:
            context_parts.append("Relevant information from documents:")
            context_parts.append("")
            
            for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
                context_parts.append(f"[Source {i}] (Relevance: {chunk.get('similarity', 0):.1%})")
                context_parts.append(chunk.get("content", ""))
                context_parts.append("")
        
        context_text = "\n".join(context_parts) if context_parts else "No specific context available."
        
        # Build the full prompt without system prompt (handled separately)
        prompt = f"""{context_text}

User: {user_message}

Instructions:
- Provide a helpful, conversational response based primarily on the context above
- If the context doesn't contain enough information, acknowledge this clearly
- Be specific and reference relevant information when appropriate
- Keep your response natural and engaging
- If you're uncertain about something, express that uncertainty
        
        Assistant: """
        
        return prompt
    
    async def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.initialized and self.provider is not None
    
    def get_provider_info(self) -> Dict:
        """Get information about the current provider"""
        if not self.initialized or not self.provider:
            return {"provider": None, "available": False}
        
        return {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "available": True,
            "streaming_enabled": settings.llm_streaming
        }


# Global instance
llm_service = LLMService()