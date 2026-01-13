"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json. The configuration is designed to be extensible for
future settings beyond NER.
"""

import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator, AnyHttpUrl, SecretStr, ConfigDict, ValidationError


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


class GeocodingConfig(BaseModel):
    """Geocoding service configuration."""
    
    nominatim_base_url: AnyHttpUrl = Field(
        description="Base URL for Nominatim geocoding service"
    )


class LlmConfig(BaseModel):
    """LLM (Large Language Model) configuration."""
    
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model name"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key"
    )


class TranslationConfig(BaseModel):
    """Translation service configuration."""
    
    target_language: Literal["nl", "de", "en", "fr", "es"] = Field(
        default="en",
        description="Default target language for translations"
    )
    provider: Literal["huggingface", "etranslation", "gemma", "auto", "google", "microsoft", "deepl", "libre"] = Field(
        default="huggingface",
        description="Translation provider to use"
    )
    
    @field_validator('provider', mode='before')
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        """Normalize provider to lowercase and strip whitespace."""
        return v.strip().lower() if isinstance(v, str) else v


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
    geocoding: GeocodingConfig = Field(
        description="Geocoding service configuration"
    )
    llm: LlmConfig = Field(
        default_factory=LlmConfig,
        description="LLM configuration"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig,
        description="Translation service configuration"
    )


# Global config instance (lazy-loaded)
_config: AppConfig | None = None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load and validate configuration from config.json.
    
    Args:
        config_path: Path to config.json file. If None, searches for config.json
                    in the project root (parent of src/ directory).
    
    Returns:
        Validated AppConfig instance
    
    Raises:
        FileNotFoundError: If config.json is not found
        json.JSONDecodeError: If config.json contains invalid JSON
        ValidationError: If configuration doesn't match the Pydantic model
    """
    global _config
    
    # Return cached config if already loaded
    if _config is not None:
        return _config
    
    # Determine config file path
    if config_path is None:
        src_dir = Path(__file__).resolve().parent
        project_root = src_dir.parent
        config_path = project_root / "config.json"
    else:
        config_path = Path(config_path).resolve()
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Please create config.json at the project root."
        )
    
    # Read and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in config file {config_path}: {e}"
        ) from e
    
    # Validate with Pydantic
    try:
        _config = AppConfig.model_validate(config_data)
    except ValidationError as e:
        raise ValueError(
            f"Configuration validation failed for {config_path}:\n{e}"
        ) from e
    
    return _config


def get_config() -> AppConfig:
    """
    Get the current configuration instance.
    
    If not yet loaded, attempts to load from default location.
    
    Returns:
        AppConfig instance
    """
    if _config is None:
        return load_config()
    return _config


def reset_config():
    """Reset the global config cache (useful for testing)."""
    global _config
    _config = None
