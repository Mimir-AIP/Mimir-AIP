import pytest
import os
import sys
import tempfile
import shutil # For cleaning up temp directories

# Ensure src directory is in Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from Plugins.VectorDatabases.ChromaDB.ChromaDB import ChromaDBPlugin # Adjusted import
from unittest.mock import MagicMock, patch, ANY

# Default model for testing if not overridden
TEST_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for persistent ChromaDB tests."""
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path) # Cleanup after test

@pytest.fixture
def plugin_global_config(temp_chroma_dir):
    """Global config mock for the plugin, using a temp dir for default path."""
    return {
        "settings": {
            "default_embedding_model": TEST_EMBED_MODEL,
            "chromadb_path": temp_chroma_dir # Default path for persistent client
        }
    }

@pytest.fixture
def chromadb_plugin(plugin_global_config):
    """Fixture to create a ChromaDBPlugin instance."""
    # Ensure the default_chroma_path directory exists for PersistentClient
    default_path = plugin_global_config["settings"]["chromadb_path"]
    os.makedirs(default_path, exist_ok=True)
    return ChromaDBPlugin(global_config=plugin_global_config)

@pytest.fixture
def mock_chroma_client():
    """Creates a mock ChromaDB client object."""
    client = MagicMock()
    client.get_or_create_collection.return_value = MagicMock(name="collection_mock")
    client.get_collection.return_value = MagicMock(name="collection_mock_get")
    client.list_collections.return_value = []
    client.delete_collection.return_value = None
    client.create_collection.return_value = MagicMock(name="collection_mock_create")
    return client

# --- Initialization Tests ---
def test_plugin_initialization(chromadb_plugin, plugin_global_config, temp_chroma_dir):
    assert chromadb_plugin is not None
    assert chromadb_plugin.default_embedding_model == TEST_EMBED_MODEL
    # Check if the default path is correctly set and made absolute
    expected_default_path = os.path.abspath(plugin_global_config["settings"]["chromadb_path"])
    assert chromadb_plugin.default_chroma_path == expected_default_path

@patch('chromadb.PersistentClient')
@patch('chromadb.Client')
def test_get_client_persistent_and_ephemeral(mock_ephemeral_client, mock_persistent_client, chromadb_plugin, temp_chroma_dir):
    # Test persistent client (default)
    client1 = chromadb_plugin._get_client(path=temp_chroma_dir, client_type="persistent")
    mock_persistent_client.assert_called_once_with(path=temp_chroma_dir)
    assert client1 == mock_persistent_client.return_value

    # Test ephemeral client
    client2 = chromadb_plugin._get_client(client_type="ephemeral")
    mock_ephemeral_client.assert_called_once()
    assert client2 == mock_ephemeral_client.return_value

    # Test persistent client caching
    client3 = chromadb_plugin._get_client(path=temp_chroma_dir, client_type="persistent")
    mock_persistent_client.assert_called_once() # Should not be called again for the same path
    assert client3 == client1


# --- Action: add ---
def test_action_add_documents_chroma_embeds(chromadb_plugin, mock_chroma_client):
    collection_name = "test_add_docs"
    docs = ["doc1", "doc2"]
    ids = ["id1", "id2"]
    metas = [{"src": "a"}, {"src": "b"}]
    context = {"texts": docs, "doc_ids": ids, "doc_metas": metas}
    step_config = {
        "config": {
            "action": "add", "collection_name": collection_name,
            "documents_context_key": "texts",
            "ids_context_key": "doc_ids",
            "metadatas_context_key": "doc_metas"
        }
    }
    # Patch _get_client to return our mock_chroma_client
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context)

        mock_chroma_client.get_or_create_collection.assert_called_once_with(
            name=collection_name, embedding_function=ANY, metadata=ANY
        )
        mock_collection = mock_chroma_client.get_or_create_collection.return_value
        mock_collection.add.assert_called_once_with(documents=docs, ids=ids, metadatas=metas)
        assert "chromadb_add_status" in result
        mock_collection.count.assert_called_once() # To include count in status

def test_action_add_precomputed_embeddings(chromadb_plugin, mock_chroma_client):
    collection_name = "test_add_embeds"
    embeds = [[0.1, 0.2], [0.3, 0.4]]
    ids = ["e_id1", "e_id2"]
    docs = ["text1", "text2"] # Optional but good to test
    context = {"vectors": embeds, "vector_ids": ids, "vector_docs": docs}
    step_config = {
        "config": {
            "action": "add", "collection_name": collection_name,
            "embeddings_context_key": "vectors",
            "ids_context_key": "vector_ids",
            "documents_context_key": "vector_docs" # Test with documents as well
        }
    }
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context)
        mock_collection = mock_chroma_client.get_or_create_collection.return_value
        mock_collection.add.assert_called_once_with(embeddings=embeds, ids=ids, documents=docs, metadatas=None)
        assert "chromadb_add_status" in result

def test_action_add_missing_docs_and_embeds_raises_error(chromadb_plugin, mock_chroma_client):
    step_config = {"config": {"action": "add", "collection_name": "test_fail"}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        with pytest.raises(ValueError, match="requires 'documents_context_key' or 'embeddings_context_key'"):
            chromadb_plugin.execute_pipeline_step(step_config, {})

def test_action_add_embeds_missing_ids_raises_error(chromadb_plugin, mock_chroma_client):
    context = {"vectors": [[0.1]]}
    step_config = {"config": {"action": "add", "collection_name": "test_fail", "embeddings_context_key": "vectors"}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        with pytest.raises(ValueError, match="also requires 'ids_context_key'"):
            chromadb_plugin.execute_pipeline_step(step_config, context)

# --- Action: query ---
def test_action_query_by_text(chromadb_plugin, mock_chroma_client):
    collection_name = "test_query_text"
    query_texts = ["find similar text"]
    context = {"q_texts": query_texts}
    step_config = {
        "config": {
            "action": "query", "collection_name": collection_name,
            "query_texts_context_key": "q_texts",
            "n_results": 3,
            "include": ["documents", "distances"],
            "where_filter": {"source": "news"}
        }
    }
    mock_query_results = {"documents": [["res1"]], "distances": [[0.1]]}
    mock_collection = mock_chroma_client.get_or_create_collection.return_value
    mock_collection.query.return_value = mock_query_results

    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context)

        mock_collection.query.assert_called_once_with(
            query_texts=query_texts, query_embeddings=None, n_results=3,
            where={"source": "news"}, where_document=None, include=["documents", "distances"]
        )
        assert "chromadb_query_results" in result
        assert result["chromadb_query_results"] == mock_query_results

def test_action_query_by_embedding(chromadb_plugin, mock_chroma_client):
    collection_name = "test_query_embed"
    query_embeds = [[0.5, 0.6]]
    context = {"q_embeds": query_embeds}
    step_config = {
        "config": {
            "action": "query", "collection_name": collection_name,
            "query_embeddings_context_key": "q_embeds",
            "output_context_key": "my_results"
        }
    }
    mock_collection = mock_chroma_client.get_or_create_collection.return_value
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        chromadb_plugin.execute_pipeline_step(step_config, context)
        mock_collection.query.assert_called_once_with(
            query_texts=None, query_embeddings=query_embeds, n_results=5, # Default n_results
            where=None, where_document=None, include=["metadatas", "documents", "distances"] # Default include
        )

def test_action_query_missing_query_input_raises_error(chromadb_plugin, mock_chroma_client):
    step_config = {"config": {"action": "query", "collection_name": "test_fail_q"}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        with pytest.raises(ValueError, match="requires 'query_texts_context_key' or 'query_embeddings_context_key'"):
            chromadb_plugin.execute_pipeline_step(step_config, {})

# --- Other Actions ---
def test_action_create_collection_only(chromadb_plugin, mock_chroma_client):
    collection_name = "new_coll_create"
    step_config = {"config": {"action": "create_collection_only", "collection_name": collection_name}}

    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context={})
        mock_chroma_client.create_collection.assert_called_once_with(
            name=collection_name, embedding_function=ANY, metadata=ANY
        )
        assert "chromadb_create_status" in result
        assert f"Collection '{collection_name}' created successfully" in result["chromadb_create_status"]

def test_action_delete_collection(chromadb_plugin, mock_chroma_client):
    collection_name = "to_delete_coll"
    step_config = {"config": {"action": "delete_collection", "collection_name": collection_name}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context={})
        mock_chroma_client.delete_collection.assert_called_once_with(name=collection_name)
        assert "chromadb_delete_status" in result

def test_action_list_collections(chromadb_plugin, mock_chroma_client):
    mock_coll_obj1 = MagicMock()
    mock_coll_obj1.name = "coll1"; mock_coll_obj1.id = "uuid1"; mock_coll_obj1.count.return_value = 10; mock_coll_obj1.metadata = {"m":1}
    mock_coll_obj2 = MagicMock()
    mock_coll_obj2.name = "coll2"; mock_coll_obj2.id = "uuid2"; mock_coll_obj2.count.return_value = 5; mock_coll_obj2.metadata = {"m":2}
    mock_chroma_client.list_collections.return_value = [mock_coll_obj1, mock_coll_obj2]

    step_config = {"config": {"action": "list_collections"}} # No collection_name needed
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context={})
        mock_chroma_client.list_collections.assert_called_once()
        assert "chromadb_collections_list" in result
        assert len(result["chromadb_collections_list"]) == 2
        assert result["chromadb_collections_list"][0]["name"] == "coll1"

def test_action_count_items(chromadb_plugin, mock_chroma_client):
    collection_name = "count_coll"
    mock_collection = mock_chroma_client.get_or_create_collection.return_value
    mock_collection.count.return_value = 42

    step_config = {"config": {"action": "count_items", "collection_name": collection_name}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context={})
        mock_collection.count.assert_called_once()
        assert "chromadb_collection_count" in result
        assert result["chromadb_collection_count"] == 42

def test_action_get_collection(chromadb_plugin, mock_chroma_client):
    collection_name = "get_this_coll"
    mock_coll_obj = mock_chroma_client.get_collection.return_value
    mock_coll_obj.name = collection_name
    mock_coll_obj.id = "uuid_get_coll" # Needs to be a string for str(uuid)
    mock_coll_obj.count.return_value = 100
    mock_coll_obj.metadata = {"type": "test"}

    step_config = {"config": {"action": "get_collection", "collection_name": collection_name}}
    with patch.object(chromadb_plugin, '_get_client', return_value=mock_chroma_client):
        result = chromadb_plugin.execute_pipeline_step(step_config, context={})
        mock_chroma_client.get_collection.assert_called_once_with(name=collection_name, embedding_function=ANY)
        assert "chromadb_collection_info" in result
        info = result["chromadb_collection_info"]
        assert info["name"] == collection_name
        assert info["id"] == "uuid_get_coll"
        assert info["count"] == 100
        assert info["metadata"] == {"type": "test"}


# --- Error Handling and Validation ---
def test_missing_action_raises_error(chromadb_plugin):
    step_config = {"config": {"collection_name": "some_coll"}} # Missing action
    with pytest.raises(ValueError, match="'action' is required"):
        chromadb_plugin.execute_pipeline_step(step_config, {})

def test_unsupported_action_raises_error(chromadb_plugin):
    step_config = {"config": {"action": "non_existent_action", "collection_name": "some_coll"}}
    with pytest.raises(ValueError, match="Unsupported ChromaDB action: non_existent_action"):
        chromadb_plugin.execute_pipeline_step(step_config, {})

def test_missing_collection_name_raises_error_for_relevant_actions(chromadb_plugin):
    actions_needing_collection = ["add", "query", "delete_collection", "count_items", "get_collection"]
    for action in actions_needing_collection:
        step_config = {"config": {"action": action}} # Missing collection_name
        with pytest.raises(ValueError, match=f"'collection_name' is required for ChromaDB action: {action}"):
            chromadb_plugin.execute_pipeline_step(step_config, {})

def test_validate_config_valid_cases(chromadb_plugin):
    valid_configs = [
        {"config": {"action": "add", "collection_name": "c", "documents_context_key": "d"}},
        {"config": {"action": "add", "collection_name": "c", "embeddings_context_key": "e", "ids_context_key": "i"}},
        {"config": {"action": "query", "collection_name": "c", "query_texts_context_key": "q"}},
        {"config": {"action": "list_collections"}},
        {"config": {"action": "delete_collection", "collection_name": "c"}},
    ]
    for sc in valid_configs:
        is_valid, msg = chromadb_plugin.validate_config(sc)
        assert is_valid, f"Config should be valid: {sc}, got msg: {msg}"
        assert msg == ""

def test_validate_config_invalid_cases(chromadb_plugin):
    invalid_configs_messages = [
        ({"config": {}}, "Missing 'action'"),
        ({"config": {"action": "dance"}}, "Unsupported action 'dance'"),
        ({"config": {"action": "add"}}, "Missing 'collection_name'"),
        ({"config": {"action": "add", "collection_name": "c"}}, "requires either 'documents_context_key' or 'embeddings_context_key'"),
        ({"config": {"action": "add", "collection_name": "c", "embeddings_context_key": "e"}}, "also requires 'ids_context_key'"),
        ({"config": {"action": "query", "collection_name": "c"}}, "requires either 'query_texts_context_key' or 'query_embeddings_context_key'"),
    ]
    for sc, expected_msg_part in invalid_configs_messages:
        is_valid, msg = chromadb_plugin.validate_config(sc)
        assert not is_valid, f"Config should be invalid: {sc}"
        assert expected_msg_part in msg, f"Msg '{msg}' did not contain '{expected_msg_part}' for {sc}"

# --- Real ChromaDB interaction tests (marked as slow, use temp_chroma_dir) ---
@pytest.mark.slow
def test_real_chromadb_workflow(plugin_global_config, temp_chroma_dir):
    # This test uses the actual ChromaDB library with a temporary persistent path.
    # It will create real DB files in temp_chroma_dir.
    real_plugin = ChromaDBPlugin(global_config=plugin_global_config)
    collection_name = "real_workflow_collection"

    # 1. Create collection explicitly (or ensure it's created by add)
    step_create = {"config": {"action": "create_collection_only", "collection_name": collection_name}}
    res_create = real_plugin.execute_pipeline_step(step_create, {})
    assert "created successfully" in res_create.get("chromadb_create_status", "") or \
           "already exists" in res_create.get("chromadb_create_status", "")


    # 2. Add documents (ChromaDB embeds them)
    docs_to_add = ["The sun is shining.", "AI is fascinating.", "ChromaDB stores vectors."]
    ids_to_add = ["sun1", "ai1", "chroma1"]
    metas_to_add = [{"topic": "weather"}, {"topic": "tech"}, {"topic": "db"}]
    context_add = {"texts": docs_to_add, "d_ids": ids_to_add, "d_metas": metas_to_add}
    step_add = {
        "config": {
            "action": "add", "collection_name": collection_name,
            "documents_context_key": "texts", "ids_context_key": "d_ids", "metadatas_context_key": "d_metas"
        }
    }
    res_add = real_plugin.execute_pipeline_step(step_add, context_add)
    assert "Added data" in res_add.get("chromadb_add_status", "")
    assert "Count: 3" in res_add.get("chromadb_add_status", "")

    # 3. Count items
    step_count = {"config": {"action": "count_items", "collection_name": collection_name}}
    res_count = real_plugin.execute_pipeline_step(step_count, {})
    assert res_count.get("chromadb_collection_count") == 3

    # 4. Query
    context_query = {"my_q": ["learn about databases"]}
    step_query = {
        "config": {
            "action": "query", "collection_name": collection_name,
            "query_texts_context_key": "my_q", "n_results": 1, "include": ["documents", "metadatas"]
        }
    }
    res_query = real_plugin.execute_pipeline_step(step_query, context_query)
    query_results = res_query.get("chromadb_query_results")
    assert query_results is not None
    assert len(query_results.get("documents", [[]])[0]) == 1
    # The most similar document should be "ChromaDB stores vectors."
    assert "ChromaDB stores vectors." in query_results["documents"][0]

    # 5. List collections
    step_list = {"config": {"action": "list_collections"}}
    res_list = real_plugin.execute_pipeline_step(step_list, {})
    assert any(c['name'] == collection_name for c in res_list.get("chromadb_collections_list",[]))

    # 6. Delete collection
    step_delete = {"config": {"action": "delete_collection", "collection_name": collection_name}}
    res_delete = real_plugin.execute_pipeline_step(step_delete, {})
    assert f"Collection '{collection_name}' deleted" in res_delete.get("chromadb_delete_status","")

    # Verify deletion
    res_list_after_delete = real_plugin.execute_pipeline_step(step_list, {})
    assert not any(c['name'] == collection_name for c in res_list_after_delete.get("chromadb_collections_list",[]))

    # The temp_chroma_dir fixture will handle cleanup of the directory.
```
