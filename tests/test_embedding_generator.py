import pytest
import os
import sys

# Ensure src directory is in Python path for imports
# This might need adjustment based on where pytest is run from
# Assuming pytest is run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from Plugins.Data_Processing.EmbeddingGenerator.EmbeddingGenerator import EmbeddingGeneratorPlugin # Adjusted import
from unittest.mock import MagicMock, patch

# Default model for testing if not overridden
TEST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# A smaller, faster model might be preferable for tests if available and compatible,
# but all-MiniLM-L6-v2 is relatively small.

@pytest.fixture
def plugin_config():
    """Global config mock for the plugin."""
    return {"settings": {"default_embedding_model": TEST_MODEL_NAME}}

@pytest.fixture
def embedding_plugin(plugin_config):
    """Fixture to create an EmbeddingGeneratorPlugin instance."""
    return EmbeddingGeneratorPlugin(global_config=plugin_config)

def test_plugin_initialization(embedding_plugin):
    assert embedding_plugin is not None
    assert embedding_plugin.default_embedding_model == TEST_MODEL_NAME

def test_get_model_loads_and_caches(embedding_plugin):
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model_instance = MagicMock()
        mock_st.return_value = mock_model_instance

        # First call - should load
        model1 = embedding_plugin._get_model(TEST_MODEL_NAME)
        mock_st.assert_called_once_with(TEST_MODEL_NAME)
        assert embedding_plugin.models[TEST_MODEL_NAME] == model1
        assert model1 == mock_model_instance

        # Second call - should be cached
        model2 = embedding_plugin._get_model(TEST_MODEL_NAME)
        mock_st.assert_called_once() # Not called again
        assert model2 == model1

def test_execute_pipeline_step_basic(embedding_plugin):
    texts = ["hello world", "another sentence"]
    context = {"input_docs": texts}
    step_config = {
        "config": {
            "input_texts_context_key": "input_docs",
            "output_embeddings_context_key": "output_vecs"
        }
    }

    # Mock the SentenceTransformer().encode().tolist() call chain
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]

    # Patch _get_model to return our mock_model
    with patch.object(embedding_plugin, '_get_model', return_value=mock_model) as mock_get_model:
        result = embedding_plugin.execute_pipeline_step(step_config, context)

        mock_get_model.assert_called_once_with(TEST_MODEL_NAME) # Uses default
        mock_model.encode.assert_called_once_with(texts)

        assert "output_vecs" in result
        assert result["output_vecs"] == [[0.1, 0.2], [0.3, 0.4]]

def test_execute_pipeline_step_custom_model(embedding_plugin):
    custom_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    texts = ["test string"]
    context = {"query": texts} # Note: plugin expects a list, even for single text
    step_config = {
        "config": {
            "input_texts_context_key": "query",
            "embedding_model": custom_model_name, # Override default
            "output_embeddings_context_key": "query_embedding"
        }
    }

    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.5, 0.6, 0.7]]
    with patch.object(embedding_plugin, '_get_model', return_value=mock_model) as mock_get_model:
        result = embedding_plugin.execute_pipeline_step(step_config, context)

        mock_get_model.assert_called_once_with(custom_model_name)
        mock_model.encode.assert_called_once_with(texts)
        assert "query_embedding" in result
        assert result["query_embedding"] == [[0.5, 0.6, 0.7]]


def test_execute_pipeline_step_single_text_input(embedding_plugin):
    text = "this is a single document"
    context = {"single_doc": text} # Plugin converts this to list
    step_config = {
        "config": {"input_texts_context_key": "single_doc"} # Use default output key
    }

    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1] * 384] # Example embedding
    with patch.object(embedding_plugin, '_get_model', return_value=mock_model):
        result = embedding_plugin.execute_pipeline_step(step_config, context)

        mock_model.encode.assert_called_once_with([text]) # Ensure it was converted to list
        assert "generated_embeddings" in result
        assert len(result["generated_embeddings"]) == 1
        assert len(result["generated_embeddings"][0]) == 384

def test_execute_pipeline_step_empty_text_list(embedding_plugin):
    context = {"empty_list_docs": []}
    step_config = {"config": {"input_texts_context_key": "empty_list_docs"}}

    result = embedding_plugin.execute_pipeline_step(step_config, context)
    assert "generated_embeddings" in result
    assert result["generated_embeddings"] == []

def test_execute_pipeline_step_input_key_not_in_context(embedding_plugin):
    context = {"some_other_key": ["data"]}
    step_config = {"config": {"input_texts_context_key": "non_existent_key"}}

    result = embedding_plugin.execute_pipeline_step(step_config, context)
    assert "generated_embeddings" in result
    assert result["generated_embeddings"] == [] # Should handle gracefully

def test_execute_pipeline_step_non_string_in_list(embedding_plugin):
    texts = ["string1", 123, {"key": "value"}, "string2"]
    processed_texts = ["string1", "123", "{'key': 'value'}", "string2"] # How plugin should convert
    context = {"mixed_input": texts}
    step_config = {"config": {"input_texts_context_key": "mixed_input"}}

    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1]] * len(processed_texts)
    with patch.object(embedding_plugin, '_get_model', return_value=mock_model):
        result = embedding_plugin.execute_pipeline_step(step_config, context)

        mock_model.encode.assert_called_once_with(processed_texts)
        assert "generated_embeddings" in result
        assert len(result["generated_embeddings"]) == len(processed_texts)

def test_execute_pipeline_step_missing_input_key_config(embedding_plugin):
    context = {"some_data": ["text"]}
    step_config = {"config": {"output_embeddings_context_key": "out_key"}} # Missing input_texts_context_key

    with pytest.raises(ValueError, match="'input_texts_context_key' is required"):
        embedding_plugin.execute_pipeline_step(step_config, context)

def test_validate_config_valid(embedding_plugin):
    step_config = {"config": {"input_texts_context_key": "some_key"}}
    is_valid, message = embedding_plugin.validate_config(step_config)
    assert is_valid
    assert message == ""

def test_validate_config_missing_input_key(embedding_plugin):
    step_config = {"config": {}} # Missing input_texts_context_key
    is_valid, message = embedding_plugin.validate_config(step_config)
    assert not is_valid
    assert "Missing 'input_texts_context_key'" in message

# Test actual model loading and embedding (can be slow, consider marking as integration or optional)
# For now, this will use the actual SentenceTransformer model.
# Ensure the model is downloaded or accessible if running in restricted environments.
@pytest.mark.slow  # Example of marking a slow test
def test_real_embedding_generation(embedding_plugin):
    texts = ["this is a real test", "Mimir-AIP is cool"]
    context = {"real_texts": texts}
    step_config = {
        "config": {
            "input_texts_context_key": "real_texts",
            "output_embeddings_context_key": "real_embeddings",
            "embedding_model": TEST_MODEL_NAME # Explicitly use the test model
        }
    }

    # No mocking here, use the actual plugin logic
    result = embedding_plugin.execute_pipeline_step(step_config, context)

    assert "real_embeddings" in result
    embeddings = result["real_embeddings"]
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0 # Check that embeddings are not empty lists
    assert isinstance(embeddings[0][0], float) # Check that elements are floats

    # Check if dimensions are consistent (e.g., all-MiniLM-L6-v2 produces 384 dimensions)
    # This depends on the model, so be careful if changing TEST_MODEL_NAME
    if TEST_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
```
