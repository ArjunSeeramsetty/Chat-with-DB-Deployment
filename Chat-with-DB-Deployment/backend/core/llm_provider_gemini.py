"""
Gemini LLM Provider Implementation for Google Generative AI
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini LLM Provider"""
    api_key: str
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    top_p: float = 0.8
    top_k: int = 40
    safety_settings: Optional[Dict[str, Any]] = None
    generation_config: Optional[Dict[str, Any]] = None


class GeminiLLMProvider(LLMProvider):
    """Gemini LLM Provider using Google Generative AI API"""

    def __init__(
        self,
        config: GeminiConfig,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.config = config
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.top_p = config.top_p
        self.top_k = config.top_k
        
        # Initialize Gemini
        self._initialize_gemini()
        
        # Metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Cost per token (Gemini 2.5 Flash-Lite pricing as of 2024)
        self.cost_per_input_token = 0.000000125  # $0.125 per 1M input tokens
        self.cost_per_output_token = 0.000000375  # $0.375 per 1M output tokens

    def _initialize_gemini(self):
        """Initialize Gemini API client"""
        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self._get_default_generation_config(),
                safety_settings=self._get_default_safety_settings()
            )
            logger.info(f"Gemini LLM Provider initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _get_default_safety_settings(self) -> List[Dict[str, Any]]:
        """Get default safety settings for Gemini"""
        if self.config.safety_settings:
            return self.config.safety_settings
            
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        ]

    def _get_default_generation_config(self) -> Dict[str, Any]:
        """Get default generation configuration for Gemini"""
        if self.config.generation_config:
            return self.config.generation_config
            
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response from Gemini LLM"""
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, system_prompt)
            
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._generate_sync, 
                full_prompt
            )
            
            # Extract content and validate
            content = response.text if hasattr(response, 'text') else str(response)
            is_safe = self.validate_response(content)
            
            # Update metrics
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                self._update_metrics(input_tokens, output_tokens)
            
            return LLMResponse(
                content=content,
                is_safe=is_safe,
                usage={
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return LLMResponse(
                content="",
                is_safe=False,
                error=str(e)
            )

    def _generate_sync(self, prompt: str):
        """Synchronous generation method for executor"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response
        except Exception as e:
            logger.error(f"Gemini sync generation error: {e}")
            raise

    def _prepare_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Prepare the full prompt for Gemini"""
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def _update_metrics(self, input_tokens: int, output_tokens: int):
        """Update usage metrics and cost tracking"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        cost = self._calculate_cost(input_tokens, output_tokens)
        self.total_cost += cost
        
        logger.debug(f"Updated metrics: +{input_tokens} input, +{output_tokens} output, +${cost:.6f}")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        return input_cost + output_cost

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary and usage statistics"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "cost_per_input_token": self.cost_per_input_token,
            "cost_per_output_token": self.cost_per_output_token,
            "model": self.model,
            "provider": "gemini"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        return {
            "provider": "gemini",
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "supports_streaming": True,
            "supports_function_calling": True,
            "supports_multimodal": True
        }

    async def generate_with_functions(
        self, 
        prompt: str, 
        functions: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response with function calling support"""
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, system_prompt)
            
            # Create tools configuration for function calling
            tools = [{"function_declarations": functions}]
            
            # Run generation with tools
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_with_tools_sync,
                full_prompt,
                tools
            )
            
            # Extract content and validate
            content = response.text if hasattr(response, 'text') else str(response)
            is_safe = self.validate_response(content)
            
            # Check for function calls
            function_calls = []
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_calls.append(part.function_call)
            
            return LLMResponse(
                content=content,
                is_safe=is_safe,
                usage={
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                },
                function_calls=function_calls if function_calls else None
            )
            
        except Exception as e:
            logger.error(f"Gemini function generation error: {e}")
            return LLMResponse(
                content="",
                is_safe=False,
                error=str(e)
            )

    def _generate_with_tools_sync(self, prompt: str, tools: List[Dict[str, Any]]):
        """Synchronous generation with tools for executor"""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                tools=tools,
                generation_config=self._get_default_generation_config()
            )
            return response
        except Exception as e:
            logger.error(f"Gemini sync tools generation error: {e}")
            raise

    async def stream_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ):
        """Stream response from Gemini LLM"""
        try:
            full_prompt = self._prepare_prompt(prompt, system_prompt)
            
            # Run streaming generation in executor
            loop = asyncio.get_event_loop()
            response_stream = await loop.run_in_executor(
                None,
                self._stream_generate_sync,
                full_prompt
            )
            
            async for chunk in self._async_stream(response_stream):
                yield chunk
                
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield {"error": str(e)}

    def _stream_generate_sync(self, prompt: str):
        """Synchronous streaming generation for executor"""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                stream=True,
                generation_config=self._get_default_generation_config()
            )
            return response
        except Exception as e:
            logger.error(f"Gemini sync streaming error: {e}")
            raise

    async def _async_stream(self, response_stream):
        """Convert sync stream to async"""
        for chunk in response_stream:
            if hasattr(chunk, 'text'):
                yield {"content": chunk.text, "type": "content"}
            elif hasattr(chunk, 'candidates'):
                yield {"candidates": chunk.candidates, "type": "candidates"}
            else:
                yield {"chunk": str(chunk), "type": "raw"}

    def validate_response(self, response: str) -> bool:
        """Enhanced validation for Gemini responses"""
        # Call parent validation first
        if not super().validate_response(response):
            return False
            
        # Additional Gemini-specific validation
        if not response or len(response.strip()) == 0:
            return False
            
        # Check for Gemini error patterns
        error_patterns = [
            "I'm sorry, I can't",
            "I cannot",
            "I'm unable to",
            "This request violates",
            "Content blocked"
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern.lower() in response_lower:
                logger.warning(f"Gemini response contains error pattern: {pattern}")
                return False
                
        return True
