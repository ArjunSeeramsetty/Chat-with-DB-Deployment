"""
LLM Provider Interface and Implementations for SQL Generation Hooks
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import json

# Import OpenAI types
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response with safety checks"""

    content: str
    is_safe: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.banned_keywords = [
            "DROP",
            "UPDATE",
            "DELETE",
            "INSERT",
            "CREATE",
            "ALTER",
            "TRUNCATE",
        ]
        self.max_response_length = 300

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response from LLM"""
        pass

    def validate_response(self, response: str) -> bool:
        """Validate LLM response for safety"""
        if not response or len(response.strip()) == 0:
            return False

        if len(response) > self.max_response_length:
            logger.warning(f"LLM response too long: {len(response)} chars")
            return False

        # Check for banned SQL keywords
        response_upper = response.upper()
        for keyword in self.banned_keywords:
            if keyword in response_upper:
                logger.warning(f"LLM response contains banned keyword: {keyword}")
                return False

        return True

    def extract_sql_from_response(self, response: str) -> Optional[str]:
        """Extract SQL from LLM response, handling markdown code blocks"""
        if not response:
            return None

        # Remove markdown code blocks
        sql = response.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]

        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and when no LLM is configured"""

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate mock response based on prompt content"""
        try:
            # Analyze prompt to generate appropriate mock response
            prompt_lower = prompt.lower()
            
            # Entity extraction responses
            if "entity extraction" in prompt_lower or "extract entities" in prompt_lower:
                mock_response = {
                    "entities": ["energy", "shortage", "consumption"],
                    "tables": ["FactAllIndiaDailySummary"],
                    "relationships": ["energy_shortage", "energy_consumption"],
                    "aggregations": ["average", "sum", "count"]
                }
                return LLMResponse(
                    content=json.dumps(mock_response),
                    is_safe=True,
                    usage={"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20}
                )
            
            # SQL generation responses
            elif "sql generation" in prompt_lower or "generate sql" in prompt_lower:
                if "average" in prompt_lower and "shortage" in prompt_lower:
                    sql = "SELECT AVG(EnergyShortage) FROM FactAllIndiaDailySummary"
                elif "total" in prompt_lower and "consumption" in prompt_lower:
                    sql = "SELECT SUM(EnergyMet) FROM FactAllIndiaDailySummary"
                elif "compare" in prompt_lower:
                    sql = "SELECT Region, AVG(EnergyShortage) as avg_shortage, AVG(EnergyMet) as avg_consumption FROM FactAllIndiaDailySummary GROUP BY Region"
                else:
                    sql = "SELECT * FROM FactAllIndiaDailySummary LIMIT 10"
                
                return LLMResponse(
                    content=sql,
                    is_safe=True,
                    usage={"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20}
                )
            
            # MDL-aware SQL generation responses (Wren AI integration)
            elif "mdl" in prompt_lower or "semantic context" in prompt_lower or "business entities" in prompt_lower:
                # Generate SQL based on the context in the prompt
                if "energy consumption" in prompt_lower and "region" in prompt_lower:
                    sql = """SELECT r.RegionName, ROUND(SUM(f.EnergyMet), 2) AS TotalEnergyConsumption
FROM FactAllIndiaDailySummary f
JOIN DimRegions r ON f.RegionID = r.RegionID
JOIN DimDates d ON f.DateID = d.DateID
WHERE DATEPART(YEAR, d.ActualDate) = 2024
GROUP BY r.RegionName
ORDER BY TotalEnergyConsumption DESC;"""
                elif "energy demand" in prompt_lower and "region" in prompt_lower:
                    sql = """SELECT r.RegionName, ROUND(SUM(f.MaxDemandSCADA), 2) AS TotalEnergyDemand
FROM FactAllIndiaDailySummary f
JOIN DimRegions r ON f.RegionID = r.RegionID
JOIN DimDates d ON f.DateID = d.DateID
WHERE DATEPART(YEAR, d.ActualDate) = 2024
GROUP BY r.RegionName
ORDER BY TotalEnergyDemand DESC;"""
                elif "generation" in prompt_lower and "source" in prompt_lower:
                    sql = """SELECT dgs.SourceName, ROUND(SUM(fdgb.GenerationAmount), 2) AS TotalGeneration
FROM FactDailyGenerationBreakdown fdgb
JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
JOIN DimDates d ON fdgb.DateID = d.DateID
WHERE DATEPART(YEAR, d.ActualDate) = 2024
GROUP BY dgs.SourceName
ORDER BY TotalGeneration DESC;"""
                else:
                    # Default MDL-aware SQL
                    sql = """SELECT r.RegionName, ROUND(SUM(f.EnergyMet), 2) AS TotalEnergyMet
FROM FactAllIndiaDailySummary f
JOIN DimRegions r ON f.RegionID = r.RegionID
JOIN DimDates d ON f.DateID = d.DateID
WHERE DATEPART(YEAR, d.ActualDate) = 2024
GROUP BY r.RegionName
ORDER BY TotalEnergyMet DESC;"""
                
                return LLMResponse(
                    content=sql,
                    is_safe=True,
                    usage={"total_tokens": 150, "prompt_tokens": 120, "completion_tokens": 30}
                )
            
            # Complexity analysis responses
            elif "complexity" in prompt_lower or "analyze" in prompt_lower:
                if "average" in prompt_lower or "total" in prompt_lower:
                    complexity = "MODERATE"
                elif "compare" in prompt_lower or "vs" in prompt_lower:
                    complexity = "COMPLEX"
                else:
                    complexity = "SIMPLE"
                
                return LLMResponse(
                    content=f'{{"complexity": "{complexity}", "score": 2}}',
                    is_safe=True,
                    usage={"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10}
                )
            
            # Default response
            else:
                return LLMResponse(
                    content="Mock response for testing purposes",
                    is_safe=True,
                    usage={"total_tokens": 20, "prompt_tokens": 15, "completion_tokens": 5}
                )
                
        except Exception as e:
            logger.error(f"Mock LLM error: {str(e)}")
            return LLMResponse(
                content="",
                is_safe=False,
                error=f"Mock error: {str(e)}"
            )


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        try:
            import openai

            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Convert to proper message types for OpenAI API
            from typing import List, Union, cast

            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam,
            )

            typed_messages: List[
                Union[
                    ChatCompletionSystemMessageParam,
                    ChatCompletionUserMessageParam,
                    ChatCompletionAssistantMessageParam,
                    ChatCompletionToolMessageParam,
                    ChatCompletionFunctionMessageParam,
                ]
            ] = []
            for msg in messages:
                if msg["role"] == "system":
                    typed_messages.append(
                        ChatCompletionSystemMessageParam(
                            role="system", content=msg["content"]
                        )
                    )
                elif msg["role"] == "user":
                    typed_messages.append(
                        ChatCompletionUserMessageParam(
                            role="user", content=msg["content"]
                        )
                    )

            # Convert to list for API compatibility
            messages_list = list(typed_messages)

            response = await self.client.chat.completions.create(
                messages=messages_list,
                model=self.model or "gpt-3.5-turbo",
                max_tokens=self.max_response_length,
                temperature=0.1,
            )

            content = response.choices[0].message.content or ""
            is_safe = self.validate_response(content)

            return LLMResponse(
                content=content,
                is_safe=is_safe,
                usage=response.usage.dict() if response.usage else None,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return LLMResponse(content="", is_safe=False, error=str(e))


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM models"""

    def __init__(
        self,
        api_key: str = "ollama",
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        enable_gpu: bool = False,
        gpu_device: Optional[str] = None,
    ):
        super().__init__(api_key, model, base_url)
        # Increase max response length for Ollama models
        self.max_response_length = 2000  # Increased from 1000
        self.enable_gpu = enable_gpu
        self.gpu_device = gpu_device
        self.gpu_used = False

        try:
            import httpx

            self.client = httpx.AsyncClient()
        except ImportError:
            raise ImportError("httpx package not installed. Run: pip install httpx")

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using Ollama API with optional GPU acceleration and robust error handling"""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Prepare request parameters for Ollama API
            request_data: Dict[str, Any] = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {},
            }

            # Add GPU acceleration if enabled
            if self.enable_gpu:
                try:
                    # Check if GPU is available
                    if self._check_gpu_availability():
                        if "options" in request_data:
                            request_data["options"]["num_gpu"] = 1
                            request_data["options"]["num_thread"] = 4
                        self.gpu_used = True
                        logger.info(f"GPU acceleration enabled for model: {self.model}")
                    else:
                        logger.warning(
                            "GPU requested but not available, falling back to CPU"
                        )
                        self.gpu_used = False
                except Exception as e:
                    logger.warning(
                        f"GPU acceleration failed: {str(e)}, falling back to CPU"
                    )
                    self.gpu_used = False
            else:
                self.gpu_used = False

            # Try multiple API endpoints for compatibility
            endpoints_to_try = [
                f"{self.base_url}/api/generate",
                f"{self.base_url}/api/completions"
            ]
            
            response = None
            last_error = None
            
            for endpoint in endpoints_to_try:
                try:
                    logger.debug(f"Trying Ollama endpoint: {endpoint}")
                    response = await self.client.post(
                        endpoint, json=request_data, timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        break
                    elif response.status_code == 404:
                        logger.debug(f"Endpoint {endpoint} returned 404, trying next...")
                        continue
                    else:
                        logger.warning(f"Endpoint {endpoint} returned {response.status_code}")
                        last_error = f"API error: {response.status_code} - {response.text}"
                        continue
                        
                except Exception as e:
                    logger.debug(f"Failed to connect to {endpoint}: {str(e)}")
                    last_error = str(e)
                    continue
            
            if response is None or response.status_code != 200:
                error_msg = last_error or "All endpoints failed"
                logger.error(f"Ollama API error: {error_msg}")
                return LLMResponse(
                    content="",
                    is_safe=False,
                    error=f"API error: {error_msg}",
                )

            response_data = response.json()
            content = response_data.get("response", "")
            is_safe = self.validate_response(content)

            return LLMResponse(
                content=content,
                is_safe=is_safe,
                usage={
                    "total_tokens": response_data.get("eval_count", 0),
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                },
            )

        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            return LLMResponse(content="", is_safe=False, error=str(e))

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for Ollama"""
        try:
            import requests

            # Check Ollama GPU info
            base_url_clean = self.base_url.replace("/v1", "") if self.base_url else ""
            response = requests.get(f"{base_url_clean}/api/tags")
            if response.status_code == 200:
                # For now, assume GPU is available if Ollama is running
                # In a real implementation, you might want to check specific GPU info
                return True
            return False
        except Exception:
            return False

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        return {
            "gpu_enabled": self.enable_gpu,
            "gpu_used": self.gpu_used,
            "gpu_device": self.gpu_device,
            "model": self.model,
        }


def create_llm_provider(
    provider_type: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_gpu: bool = False,
    gpu_device: Optional[str] = None,
) -> LLMProvider:
    """Factory function to create LLM provider"""
    
    print(f"🔧 Creating LLM provider: {provider_type}")
    print(f"📊 API Key provided: {'*' * len(api_key) if api_key else 'None'}")
    print(f"📊 Model: {model}")
    print(f"📊 Base URL: {base_url}")

    if not provider_type:
        print("⚠️ No LLM provider configured, using mock provider")
        return MockLLMProvider()

    if provider_type.lower() == "openai":
        if not api_key:
            print("⚠️ OpenAI provider requires API key, using mock provider")
            return MockLLMProvider()
        print("✅ Creating OpenAI provider")
        return OpenAIProvider(api_key, model or "gpt-3.5-turbo")
    elif provider_type.lower() == "ollama":
        # Ollama doesn't require an API key
        print("✅ Creating Ollama provider")
        return OllamaProvider(
            api_key=api_key or "ollama",
            model=model or "llama3.2:3b",
            base_url=base_url or "http://localhost:11434/v1",
            enable_gpu=enable_gpu,
            gpu_device=gpu_device,
        )
    elif provider_type.lower() == "gemini":
        try:
            print("🔧 Attempting to create Gemini provider...")
            from .llm_provider_gemini import GeminiLLMProvider, GeminiConfig
            
            if not api_key:
                print("⚠️ Gemini provider requires API key, using mock provider")
                return MockLLMProvider()
                
            print(f"✅ Creating Gemini provider with API key: {'*' * len(api_key)}")
            config = GeminiConfig(
                api_key=api_key,
                model=model or "gemini-2.5-flash-lite"
            )
            provider = GeminiLLMProvider(config)
            print(f"✅ Gemini provider created successfully: {type(provider).__name__}")
            return provider
        except ImportError as e:
            print(f"❌ Failed to import Gemini provider: {e}")
            return MockLLMProvider()
        except Exception as e:
            print(f"❌ Failed to create Gemini provider: {e}")
            return MockLLMProvider()
    elif provider_type.lower() == "anthropic":
        # TODO: Implement Anthropic provider
        print("⚠️ Anthropic provider not yet implemented, using mock")
        return MockLLMProvider()
    elif provider_type.lower() == "local":
        # TODO: Implement local provider (Ollama, etc.)
        print("⚠️ Local provider not yet implemented, using mock")
        return MockLLMProvider()
    else:
        print(f"⚠️ Unknown LLM provider type: {provider_type}, using mock")
        return MockLLMProvider()
