import json
from typing import Dict, List, Any, Optional, Union
import asyncio
from tqdm import tqdm
import os

# 1. Update Imports
from openai import AsyncAzureOpenAI, AsyncOpenAI 


class LLMAnalyzer:
    """
    A flexible analyzer class for processing text with Azure OpenAI OR Local LLMs (Ollama).
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 endpoint: Optional[str] = None, 
                 deployment: str = "gpt-4.1-nano",
                 api_version: str = "2024-10-21"):
        """
        Initialize the analyzer. Auto-detects provider based on endpoint URL.
        """
        self.deployment = deployment
        self.api_version = api_version
        
        # 2. Flexible Configuration
        # If no key provided for local, use dummy "ollama"
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY") or "ollama"
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "http://localhost:11434/v1"
        
        # 3. Client Selection Logic
        if "azure.com" in self.endpoint:
            print(f"Connecting to Azure OpenAI: {self.deployment}")
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
        else:
            print(f"Connecting to Local/OpenAI/Minstral via Openai API: {self.deployment}")
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint
            )
    
    async def analyze_single_entry(self, 
                                 text: str, 
                                 system_prompt: str, 
                                 user_prompt_template: str,
                                 expected_schema: Dict[str, Any],
                                 max_tokens: int = 8000,
                                 temperature: float = 0.1,
                                 text_limit: int = 8000) -> Dict[str, Any]:
        
        user_prompt = user_prompt_template.format(text=text[:text_limit])
        
        try:
            # 4. Minimal logic tweak for compatibility
            # O1/GPT-5 models often use 'max_completion_tokens', others use 'max_tokens'
            is_reasoning_model = self.deployment.startswith(("gpt-5", "o1"))
            
            completion_args = {
                "model": self.deployment,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"} # Ollama supports this in v1 API
            }

            if is_reasoning_model:
                completion_args["max_completion_tokens"] = max_tokens
            else:
                completion_args["max_tokens"] = max_tokens
                completion_args["temperature"] = temperature

            response = await self.client.chat.completions.create(**completion_args)

            # ... [Rest of your existing parsing logic remains exactly the same] ...
            result_text = response.choices[0].message.content
            result_text = result_text.strip()
            result = json.loads(result_text)
            return self._validate_result(result, expected_schema)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._create_error_result(expected_schema, f"Analysis failed: {str(e)}")

    def _validate_result(self, result: Dict[str, Any], expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and structure the result according to the expected schema.
        
        Args:
            result: Raw result from AI
            expected_schema: Expected structure with default values and types
        
        Returns:
            dict: Validated result matching schema
        """
        validated_result = {}
        
        
        for key, schema_info in expected_schema.items():
            
            if isinstance(schema_info, dict) and "default" in schema_info:
                default_value = schema_info["default"]
                expected_type = schema_info.get("type", type(default_value))
                
                # Clean the key and look for it in result
                clean_key = key.strip()
                value = None
                
                # Try to find the key (handle potential whitespace issues)
                for result_key in result.keys():
                    if result_key.strip() == clean_key:
                        value = result[result_key]
                        break
                
                if value is not None:
                    # Handle list type validation
                    if expected_type == list:
                        if isinstance(value, list):
                            validated_result[clean_key] = value
                        elif value:  # Convert non-empty single values to list
                            validated_result[clean_key] = [value]
                        else:
                            validated_result[clean_key] = default_value
                    else:
                        # Handle other types
                        validated_result[clean_key] = value if isinstance(value, expected_type) else default_value
                else:
                    validated_result[clean_key] = default_value
            else:
                # Simple default value
                clean_key = key.strip()
                # Try to find the key in result
                value = None
                for result_key in result.keys():
                    if result_key.strip() == clean_key:
                        value = result[result_key]
                        break
                validated_result[clean_key] = value if value is not None else schema_info
        
        return validated_result
    
    def _create_error_result(self, expected_schema: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Create an error result that matches the expected schema.
        
        Args:
            expected_schema: Expected structure
            error_message: Error message to include
        
        Returns:
            dict: Error result with schema structure
        """
        error_result = {}
        
        for key, schema_info in expected_schema.items():
            clean_key = key.strip()
            if isinstance(schema_info, dict) and "default" in schema_info:
                error_result[clean_key] = schema_info["default"]
            else:
                error_result[clean_key] = schema_info
        
        error_result["error"] = error_message
        return error_result
    
    async def analyze_batch(self, 
                          entries: List[Dict], 
                          text_field: str,
                          system_prompt: str,
                          user_prompt_template: str,
                          expected_schema: Dict[str, Any],
                          analysis_field: str = "analysis_result",
                          batch_size: int = 5, 
                          max_entries: Optional[int] = None, 
                          start_index: int = 0, 
                          output_file: Optional[str] = None,
                          **analysis_kwargs) -> List[Dict]:
        """
        Analyze multiple entries in batches with progress tracking and optional file output.
        
        Args:
            entries: List of entries to analyze
            text_field: Field name containing text to analyze
            system_prompt: System prompt for the AI
            user_prompt_template: User prompt template (should contain {text} placeholder)
            expected_schema: Dictionary defining the expected output structure
            analysis_field: Field name to store analysis results
            batch_size: Number of concurrent requests
            max_entries: Maximum number of entries to process
            start_index: Number of entries to skip from the beginning
            output_file: Optional path to save results progressively
            **analysis_kwargs: Additional arguments for analyze_single_entry
        
        Returns:
            list: Entries extended with analysis results
        """
        
        # Apply start_index and max_entries filtering
        if start_index > 0:
            entries = entries[start_index:]
        entries_to_process = entries[:max_entries] if max_entries else entries
        
        print(f"Analyzing {len(entries_to_process)} entries (batch size: {batch_size})")
        print(f"Processing from index {start_index} to {start_index + len(entries_to_process) - 1}")
        if output_file:
            print(f"Saving to: {output_file}")
        
        analyzed_entries = []
        successful_count = 0
        error_count = 0
        
        # Ensure output directory exists if output_file is specified
        if output_file:
            from pathlib import Path
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Open output file for writing if specified
        outfile = None
        if output_file:
            outfile = open(output_file, 'a', encoding='utf-8')
        
        try:
            # Process entries in batches with progress bar
            with tqdm(total=len(entries_to_process), desc="Analyzing entries", unit="entry") as pbar:
                for i in range(0, len(entries_to_process), batch_size):
                    batch = entries_to_process[i:i + batch_size]
                    
                    # Create tasks for concurrent processing
                    tasks = []
                    for entry in batch:
                        text = entry.get(text_field, "")
                        task = self.analyze_single_entry(
                            text=text,
                            system_prompt=system_prompt,
                            user_prompt_template=user_prompt_template,
                            expected_schema=expected_schema,
                            **analysis_kwargs
                        )
                        tasks.append(task)
                    
                    # Execute batch concurrently
                    try:
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process results and save immediately if output file specified
                        for entry, result in zip(batch, batch_results):
                            if isinstance(result, Exception):
                                entry[analysis_field] = self._create_error_result(
                                    expected_schema, str(result)
                                )
                                error_count += 1
                            else:
                                entry[analysis_field] = result
                                successful_count += 1
                            
                            # Save entry immediately to file if specified
                            if outfile:
                                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                outfile.flush()  # Ensure data is written
                            
                            analyzed_entries.append(entry)
                            pbar.update(1)
                    
                    except Exception as e:
                        # Handle batch errors
                        for entry in batch:
                            entry[analysis_field] = self._create_error_result(
                                expected_schema, f"Batch error: {str(e)}"
                            )
                            if outfile:
                                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                outfile.flush()
                            analyzed_entries.append(entry)
                            error_count += 1
                            pbar.update(1)
                    
                    # Rate limiting delay
                    if i + batch_size < len(entries_to_process):
                        await asyncio.sleep(1)
        
        finally:
            if outfile:
                outfile.close()
        
        # Final summary
        print(f"Complete: {successful_count} success, {error_count} errors")
        
        return analyzed_entries