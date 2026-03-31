"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json. The configuration is designed to be extensible for
future settings beyond NER.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, AnyHttpUrl, SecretStr, ConfigDict
from decide_ai_service_base.config import load_config


class NerConfig(BaseModel):
    """NER (Named Entity Recognition) configuration settings."""
    
    language: Literal["nl", "de", "en"] = Field(
        default="nl",
        description="Default language for NER extraction"
    )
    method: Literal["composite", "spacy", "huggingface", "flair", "regex", "title"] = Field(
        default="composite",
        description="Default NER extraction method"
    )
    post_process: bool = Field(
        default=True,
        description="Whether to apply post-processing to extracted entities"
    )
    labels: list[str] = Field(
        default_factory=lambda: ["CITY", "DOMAIN", "HOUSENUMBERS", "INTERSECTION", "POSTCODE", "PROVINCE", "ROAD", "STREET"],
        description="List of NER labels to extract"
    )
    enable_refinement: bool = Field(
        default=True,
        description="Whether to apply entity refinement to classify generic labels (DATE, LOCATION) into specific types"
    )


class AppSettingsConfig(BaseModel):
    """Application-level settings."""
    
    mode: Literal["development", "production", "staging", "test"] = Field(
        default="development",
        description="Application mode (development, production, etc.)"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="debug",
        description="Logging level (debug, info, warning, error)"
    )
    
    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize log level to lowercase and strip whitespace."""
        return v.strip().lower() if isinstance(v, str) else v


class ETranslationConfig(BaseModel):
    """eTranslation service configuration."""
    
    base_url: AnyHttpUrl = Field(
        default="https://language-tools.ec.europa.eu/etranslation/api",
        description="eTranslation API base URL"
    )
    bearer_token: SecretStr | None = Field(
        default=None,
        description="Bearer token for eTranslation API authentication"
    )
    username: str | None = Field(
        default=None,
        description="Username for eTranslation API basic authentication"
    )
    password: SecretStr | None = Field(
        default=None,
        description="Password for eTranslation API basic authentication"
    )
    domain: str = Field(
        default="GEN",
        description="Translation domain (e.g., GEN for general)"
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="HTTP request timeout in seconds"
    )
    callback_wait_timeout: float = Field(
        default=600.0,
        ge=1.0,
        description="Maximum time to wait for translation callback in seconds"
    )
    max_text_length: int = Field(
        default=4000,
        ge=100,
        description="Maximum text length per translation request"
    )
    callback_url: str | None = Field(
        default=None,
        description="Public callback URL for eTranslation to send results (e.g., https://your-host/etranslation)"
    )


class OllamaTranslationConfig(BaseModel):
    """Ollama translation service configuration."""

    base_url: AnyHttpUrl = Field(
        default="http://ollama:11434",
        description="Base URL for the Ollama API"
    )
    model: str = Field(
        default="mistral-nemo",
        description="Ollama model name to use for translation"
    )
    timeout_seconds: float = Field(
        default=180.0,
        ge=1.0,
        description="HTTP request timeout in seconds"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Generation temperature"
    )
    max_text_length: int = Field(
        default=6000,
        ge=200,
        description="Maximum characters per translation chunk sent to Ollama"
    )


class TranslationConfig(BaseModel):
    """Translation service configuration."""
    
    target_language: Literal["nl", "de", "en", "fr", "es"] = Field(
        default="en",
        description="Default target language for translations"
    )
    provider: Literal["huggingface", "etranslation", "ollama", "auto", "google", "microsoft", "deepl", "libre"] = Field(
        default="huggingface",
        description="Translation provider to use"
    )
    etranslation: ETranslationConfig = Field(
        default_factory=ETranslationConfig,
        description="eTranslation-specific settings"
    )
    ollama: OllamaTranslationConfig = Field(
        default_factory=OllamaTranslationConfig,
        description="Ollama-specific settings"
    )
    
    @field_validator('provider', mode='before')
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        """Normalize provider to lowercase and strip whitespace."""
        return v.strip().lower() if isinstance(v, str) else v


class SegmentationConfig(BaseModel):
    """Segmentation model configuration for document structure extraction."""
    
    model_name: str = Field(
        default="gpt-4.1",
        description="LLM deployment / model name"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the LLM endpoint"
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint URL for the LLM service"
    )
    max_new_tokens: int = Field(
        default=14000,
        ge=100,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more deterministic)"
    )

    max_gap: int = Field(
        default=5,
        description="Maximum character gap allowed when projecting segments back to original text. "
    )
    


class AppConfig(BaseModel):
    """Root application configuration model."""
    
    model_config = ConfigDict(extra="forbid")  # Reject extra fields not defined in the model
    
    app: AppSettingsConfig = Field(
        default_factory=AppSettingsConfig,
        description="Application-level settings"
    )
    ner: NerConfig = Field(
        default_factory=NerConfig,
        description="NER configuration settings"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig,
        description="Translation service configuration"
    )
    segmentation: SegmentationConfig = Field(
        default_factory=SegmentationConfig,
        description="Segmentation model configuration"
    )


def get_config() -> AppConfig:
    """
    Get the current configuration instance.
    
    If not yet loaded, attempts to load from default location.
    
    Returns:
        AppConfig instance
    """
    return load_config(AppConfig)
