from ..BasePlugin import BasePlugin # Assuming BasePlugin is in src/Plugins/BasePlugin.py
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingGeneratorPlugin(BasePlugin):
    plugin_type = "Data_Processing"
    plugin_name = "EmbeddingGenerator"

    def __init__(self, global_config=None): # Accept global_config
        super().__init__()
        self.global_config = global_config if global_config else {}
        # Allow global default model to be overridden by step config
        self.default_embedding_model = self.global_config.get("settings", {}).get("default_embedding_model", "all-MiniLM-L6-v2")
        self.models = {} # Cache for loaded models
        logger.info(f"EmbeddingGeneratorPlugin initialized. Default model: {self.default_embedding_model}")

    def _get_model(self, model_name):
        if model_name not in self.models:
            logger.info(f"Loading sentence transformer model: {model_name}")
            try:
                self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model {model_name}: {e}")
                raise
        return self.models[model_name]

    def execute_pipeline_step(self, step_config, context):
        logger.debug(f"Executing EmbeddingGenerator step with config: {step_config}")
        config_params = step_config.get("config", {})

        input_texts_key = config_params.get("input_texts_context_key")
        output_embeddings_key = config_params.get("output_embeddings_context_key", "generated_embeddings")
        # Model can be specified in step_config, overriding global default
        model_name = config_params.get("embedding_model", self.default_embedding_model)

        if not input_texts_key:
            logger.error("'input_texts_context_key' is required for EmbeddingGenerator plugin.")
            raise ValueError("'input_texts_context_key' is required for EmbeddingGenerator plugin.")

        texts_to_embed = context.get(input_texts_key)

        if texts_to_embed is None:
            logger.warning(f"Input texts key '{input_texts_key}' not found in context or is None. Returning empty list for embeddings.")
            return {output_embeddings_key: []}

        if not isinstance(texts_to_embed, list):
            logger.debug(f"Input texts for embedding is not a list (type: {type(texts_to_embed)}). Converting to list.")
            texts_to_embed = [texts_to_embed]

        if not texts_to_embed:
            logger.info("Input texts list is empty. Returning empty list for embeddings.")
            return {output_embeddings_key: []}

        # Ensure all items are strings
        processed_texts_to_embed = []
        for i, text in enumerate(texts_to_embed):
            if not isinstance(text, str):
                logger.warning(f"Item at index {i} in texts_to_embed is not a string (type: {type(text)}). Converting to string.")
                processed_texts_to_embed.append(str(text))
            else:
                processed_texts_to_embed.append(text)

        if not processed_texts_to_embed:
             logger.info("After processing, texts_to_embed is empty. Returning empty list for embeddings.")
             return {output_embeddings_key: []}

        try:
            model = self._get_model(model_name)
            logger.info(f"Generating embeddings for {len(processed_texts_to_embed)} text(s) using model {model_name}...")
            embeddings = model.encode(processed_texts_to_embed).tolist() # Convert to list for JSON serializability
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return {output_embeddings_key: embeddings}
        except Exception as e:
            logger.error(f"Error during embedding generation with model {model_name}: {e}")
            # Depending on desired behavior, either raise e or return an error indicator
            # For now, let's raise to make pipeline failures explicit
            raise

    def validate_config(self, step_config):
        config_params = step_config.get("config", {})
        if "input_texts_context_key" not in config_params:
            return False, "Missing 'input_texts_context_key' in EmbeddingGenerator config."
        return True, ""

# Example of how it might be registered or used by PluginManager
# This part is usually handled by PluginManager's discovery mechanism.
# For standalone testing, you might do something like this:
if __name__ == '__main__':
    # Mock global_config and context for testing
    mock_global_config = {"settings": {"default_embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}}
    plugin = EmbeddingGeneratorPlugin(global_config=mock_global_config)

    # Test case 1: Basic usage
    mock_context_1 = {"document_texts": ["Hello world", "This is a test sentence."]}
    mock_step_config_1 = {
        "plugin": "Data_Processing.EmbeddingGenerator",
        "name": "Generate Text Embeddings",
        "config": {
            "input_texts_context_key": "document_texts",
            "output_embeddings_context_key": "doc_embeddings"
        }
    }
    results_1 = plugin.execute_pipeline_step(mock_step_config_1, mock_context_1)
    print("Test Case 1 Results:", results_1)
    if results_1 and "doc_embeddings" in results_1:
        print(f"Number of embeddings: {len(results_1['doc_embeddings'])}")
        if results_1['doc_embeddings']:
            print(f"Dimension of first embedding: {len(results_1['doc_embeddings'][0])}")

    # Test case 2: Custom model and output key
    mock_context_2 = {"query_text": "What is the weather like?"} # Single string
    mock_step_config_2 = {
        "plugin": "Data_Processing.EmbeddingGenerator",
        "name": "Generate Query Embedding",
        "config": {
            "input_texts_context_key": "query_text",
            "output_embeddings_context_key": "query_vec",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }
    }
    results_2 = plugin.execute_pipeline_step(mock_step_config_2, mock_context_2)
    print("\nTest Case 2 Results:", results_2)
    if results_2 and "query_vec" in results_2:
        print(f"Number of embeddings: {len(results_2['query_vec'])}")
        if results_2['query_vec']:
            print(f"Dimension of first embedding: {len(results_2['query_vec'][0])}")

    # Test case 3: Empty input list
    mock_context_3 = {"empty_texts": []}
    mock_step_config_3 = {
        "config": {
            "input_texts_context_key": "empty_texts",
            "output_embeddings_context_key": "empty_embeddings"
        }
    }
    results_3 = plugin.execute_pipeline_step(mock_step_config_3, mock_context_3)
    print("\nTest Case 3 Results:", results_3) # Should be {'empty_embeddings': []}

    # Test case 4: Input key not in context
    mock_context_4 = {"some_other_key": "data"}
    mock_step_config_4 = {
        "config": {
            "input_texts_context_key": "non_existent_key",
        }
    }
    try:
        results_4 = plugin.execute_pipeline_step(mock_step_config_4, mock_context_4)
        print("\nTest Case 4 Results:", results_4) # Should be {'generated_embeddings': []}
    except Exception as e:
        print("\nTest Case 4 Error:", e) # Should not error out, but return empty or handle gracefully

    # Test case 5: Invalid model name (will raise an error during model loading)
    # mock_step_config_5 = {
    #     "config": {
    #         "input_texts_context_key": "document_texts",
    #         "embedding_model": "this-model-does-not-exist"
    #     }
    # }
    # try:
    #     plugin.execute_pipeline_step(mock_step_config_5, mock_context_1)
    # except Exception as e:
    #     print("\nTest Case 5 Error (expected):", e)

    # Test case 6: Non-string input in list
    mock_context_6 = {"mixed_data_types": ["This is a string", 123, {"not": "a string"}]}
    mock_step_config_6 = {
        "config": {
            "input_texts_context_key": "mixed_data_types",
            "output_embeddings_context_key": "mixed_embeddings"
        }
    }
    results_6 = plugin.execute_pipeline_step(mock_step_config_6, mock_context_6)
    print("\nTest Case 6 Results:", results_6)
    if results_6 and "mixed_embeddings" in results_6:
        print(f"Number of embeddings: {len(results_6['mixed_embeddings'])}")
        if results_6['mixed_embeddings']:
            print(f"Dimension of first embedding: {len(results_6['mixed_embeddings'][0])}")

```
