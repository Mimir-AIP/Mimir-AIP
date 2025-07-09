from ..BasePlugin import BasePlugin # Assuming BasePlugin is in src/Plugins/BasePlugin.py
import chromadb
from chromadb.utils import embedding_functions as ef # Alias for clarity
import logging
import os # For path joining

logger = logging.getLogger(__name__)

class ChromaDBPlugin(BasePlugin):
    plugin_type = "VectorDatabases"
    plugin_name = "ChromaDB"

    def __init__(self, global_config=None): # Accept global_config
        super().__init__()
        self.global_config = global_config if global_config else {}
        # Default path from global settings, can be overridden by step config
        # Ensure settings are accessed correctly from global_config
        settings = self.global_config.get("settings", {})
        self.default_chroma_path = settings.get("chromadb_path", "mimir_chroma_db")
        # Ensure the path is created relative to the project root if it's not absolute
        if not os.path.isabs(self.default_chroma_path):
            # Assuming this plugin file is src/Plugins/VectorDatabases/ChromaDB/ChromaDB.py
            # Project root is ../../../../ from here
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            self.default_chroma_path = os.path.join(project_root, self.default_chroma_path)

        self.default_embedding_model = settings.get("default_embedding_model", "all-MiniLM-L6-v2")
        self.clients = {} # Cache clients by path to allow multiple persistent dbs
        logger.info(f"ChromaDBPlugin initialized. Default DB path: {self.default_chroma_path}, Default model: {self.default_embedding_model}")

    def _get_client(self, path=None, client_type="persistent"):
        effective_path = path or self.default_chroma_path
        if client_type == "ephemeral":
            logger.info("Creating ephemeral ChromaDB client.")
            return chromadb.Client() # Ephemeral client

        # For persistent client, use a cached instance if available for the path
        if effective_path not in self.clients:
            logger.info(f"Creating persistent ChromaDB client at path: {effective_path}")
            # Ensure directory exists for persistent client
            os.makedirs(effective_path, exist_ok=True)
            self.clients[effective_path] = chromadb.PersistentClient(path=effective_path)
        return self.clients[effective_path]

    def execute_pipeline_step(self, step_config, context):
        logger.debug(f"Executing ChromaDB step with config: {step_config}")
        config_params = step_config.get("config", {})

        action = config_params.get("action")
        collection_name = config_params.get("collection_name")

        client_path_override = config_params.get("chroma_path")
        client_path = client_path_override or self.default_chroma_path
        if client_path_override and not os.path.isabs(client_path_override):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            client_path = os.path.join(project_root, client_path_override)

        client_type = config_params.get("client_type", "persistent")

        client = self._get_client(path=client_path, client_type=client_type)

        if not action:
            logger.error("'action' is required for ChromaDB plugin.")
            raise ValueError("'action' is required for ChromaDB plugin.")
        if action != "list_collections" and not collection_name: # collection_name not needed for list_collections
            logger.error("'collection_name' is required for ChromaDB action: {action}.")
            raise ValueError(f"'collection_name' is required for ChromaDB action: {action}.")

        # Embedding function for get_or_create_collection
        # This is used by ChromaDB if it needs to embed documents you pass directly to its .add()
        embedding_model_name = config_params.get("embedding_model", self.default_embedding_model)
        # Use the alias for embedding_functions
        chroma_embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

        distance_function = config_params.get("distance_function", "cosine") # l2, cosine, ip

        collection = None
        if action not in ["list_collections", "delete_collection", "create_collection_only"]: # These actions might not need to get/create
            try:
                # For most actions, we get or create the collection.
                # For create_collection_only, this is done explicitly below.
                if action != "create_collection_only":
                    logger.info(f"Getting or creating collection '{collection_name}' with model '{embedding_model_name}' and distance '{distance_function}'.")
                    collection = client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=chroma_embedding_function,
                        metadata={"hnsw:space": distance_function}
                    )
            except Exception as e:
                logger.error(f"Error getting or creating collection '{collection_name}': {e}")
                raise

        if action == "add":
            documents = context.get(config_params.get("documents_context_key"))
            ids = context.get(config_params.get("ids_context_key"))
            metadatas = context.get(config_params.get("metadatas_context_key"))
            embeddings = context.get(config_params.get("embeddings_context_key"))

            if not documents and not embeddings:
                logger.error("ChromaDB 'add' action requires 'documents_context_key' or 'embeddings_context_key'.")
                raise ValueError("ChromaDB 'add' action requires 'documents_context_key' or 'embeddings_context_key'.")

            # Ensure inputs are lists if not None
            if documents is not None and not isinstance(documents, list): documents = [documents]
            if ids is not None and not isinstance(ids, list): ids = [ids]
            if metadatas is not None and not isinstance(metadatas, list): metadatas = [metadatas]
            if embeddings is not None and not isinstance(embeddings, list): embeddings = [embeddings] # Embeddings should be list of lists

            # Auto-generate IDs if documents are provided but no IDs
            if documents and not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
                logger.info(f"Generated {len(ids)} IDs for documents.")

            # Validate IDs if embeddings are provided
            if embeddings and not ids:
                logger.error("ChromaDB 'add' action with 'embeddings_context_key' also requires 'ids_context_key'.")
                raise ValueError("ChromaDB 'add' action with 'embeddings_context_key' also requires 'ids_context_key'.")

            if embeddings and ids and len(embeddings) != len(ids):
                logger.error(f"Mismatch in length of embeddings ({len(embeddings)}) and IDs ({len(ids)}).")
                raise ValueError("Length of embeddings and IDs must match.")

            if documents and ids and len(documents) != len(ids):
                 logger.error(f"Mismatch in length of documents ({len(documents)}) and IDs ({len(ids)}).")
                 raise ValueError("Length of documents and IDs must match.")

            if metadatas and ids and len(metadatas) != len(ids):
                 logger.error(f"Mismatch in length of metadatas ({len(metadatas)}) and IDs ({len(ids)}).")
                 raise ValueError("Length of metadatas and IDs must match.")

            try:
                if embeddings:
                    logger.info(f"Adding {len(embeddings)} pre-computed embeddings to collection '{collection_name}'.")
                    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
                elif documents:
                    logger.info(f"Adding {len(documents)} documents to collection '{collection_name}' for ChromaDB to embed.")
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)

                output_key = config_params.get("output_context_key", "chromadb_add_status")
                return {output_key: f"Added data to collection '{collection_name}'. Count: {collection.count()}"}
            except Exception as e:
                logger.error(f"Error adding data to collection '{collection_name}': {e}")
                raise

        elif action == "query":
            query_texts = context.get(config_params.get("query_texts_context_key"))
            query_embeddings = context.get(config_params.get("query_embeddings_context_key"))
            n_results = int(config_params.get("n_results", 5))
            where_filter = config_params.get("where_filter")
            where_document_filter = config_params.get("where_document_filter")
            include_fields = config_params.get("include", ["metadatas", "documents", "distances"])

            if not query_texts and not query_embeddings:
                logger.error("ChromaDB 'query' action requires 'query_texts_context_key' or 'query_embeddings_context_key'.")
                raise ValueError("ChromaDB 'query' action requires 'query_texts_context_key' or 'query_embeddings_context_key'.")

            # Ensure queries are lists
            if query_texts is not None and not isinstance(query_texts, list): query_texts = [query_texts]
            if query_embeddings is not None and not isinstance(query_embeddings, list): query_embeddings = [query_embeddings]


            logger.info(f"Querying collection '{collection_name}' with n_results={n_results}.")
            try:
                results = collection.query(
                    query_texts=query_texts if query_texts else None,
                    query_embeddings=query_embeddings if query_embeddings else None,
                    n_results=n_results,
                    where=where_filter,
                    where_document=where_document_filter,
                    include=include_fields
                )
                output_key = config_params.get("output_context_key", "chromadb_query_results")
                return {output_key: results}
            except Exception as e:
                logger.error(f"Error querying collection '{collection_name}': {e}")
                raise

        elif action == "create_collection_only":
            try:
                logger.info(f"Explicitly creating collection '{collection_name}' with model '{embedding_model_name}' and distance '{distance_function}'.")
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=chroma_embedding_function,
                    metadata={"hnsw:space": distance_function}
                )
                output_key = config_params.get("output_context_key", "chromadb_create_status")
                return {output_key: f"Collection '{collection_name}' created successfully."}
            except Exception as e:
                # Handle cases where collection might already exist, depending on desired behavior
                # For now, re-raise if it's not a "collection already exists" type error.
                # ChromaDB's create_collection might raise if it exists, unlike get_or_create.
                logger.error(f"Error explicitly creating collection '{collection_name}': {e}")
                # Check if it's a "UniqueConstraintFailed: Collection ... already exists"
                if "UniqueConstraintFailed" in str(e) and f"Collection {collection_name} already exists" in str(e):
                     logger.warning(f"Collection '{collection_name}' already exists. Creation skipped.")
                     output_key = config_params.get("output_context_key", "chromadb_create_status")
                     return {output_key: f"Collection '{collection_name}' already exists."}
                raise

        elif action == "delete_collection":
            try:
                logger.info(f"Deleting collection '{collection_name}'.")
                client.delete_collection(name=collection_name)
                output_key = config_params.get("output_context_key", "chromadb_delete_status")
                return {output_key: f"Collection '{collection_name}' deleted."}
            except Exception as e:
                # Handle case where collection might not exist
                logger.error(f"Error deleting collection '{collection_name}': {e}")
                # If chromadb raises a specific error for not found, catch it.
                # For now, assume it might raise a general error or not error if not found.
                # Let's re-raise to see behavior.
                raise

        elif action == "get_collection":
            try:
                logger.info(f"Getting collection '{collection_name}'.")
                collection_obj = client.get_collection(
                    name=collection_name,
                    embedding_function=chroma_embedding_function # May be needed if collection exists but client is new
                )
                # We can't directly return the collection object easily via context if it's not serializable.
                # So, return info about it.
                info = {
                    "name": collection_obj.name,
                    "id": str(collection_obj.id), # UUID to string
                    "count": collection_obj.count(),
                    "metadata": collection_obj.metadata
                }
                output_key = config_params.get("output_context_key", "chromadb_collection_info")
                return {output_key: info}
            except Exception as e:
                logger.error(f"Error getting collection '{collection_name}': {e}")
                raise

        elif action == "list_collections":
            try:
                logger.info("Listing all collections.")
                collections = client.list_collections()
                # collections is a list of Collection objects. Extract names or more info.
                collection_details = [{"name": c.name, "id": str(c.id), "count": c.count(), "metadata":c.metadata} for c in collections]
                output_key = config_params.get("output_context_key", "chromadb_collections_list")
                return {output_key: collection_details}
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                raise

        elif action == "count_items": # Renamed from get_collection_count for clarity
            if not collection: # Should have been fetched above unless action was just create/delete
                 collection = client.get_collection(name=collection_name, embedding_function=chroma_embedding_function)
            count = collection.count()
            output_key = config_params.get("output_context_key", "chromadb_collection_count")
            logger.info(f"Collection '{collection_name}' has {count} items.")
            return {output_key: count}

        else:
            logger.error(f"Unsupported ChromaDB action: {action}.")
            raise ValueError(f"Unsupported ChromaDB action: {action}. Supported: add, query, create_collection_only, delete_collection, get_collection, list_collections, count_items.")

    def validate_config(self, step_config):
        config_params = step_config.get("config", {})
        action = config_params.get("action")
        if not action:
            return False, "Missing 'action' in ChromaDB config."

        supported_actions = ["add", "query", "create_collection_only", "delete_collection", "get_collection", "list_collections", "count_items"]
        if action not in supported_actions:
            return False, f"Unsupported action '{action}'. Supported actions are: {', '.join(supported_actions)}."

        if action != "list_collections" and "collection_name" not in config_params:
            return False, f"Missing 'collection_name' in ChromaDB config for action '{action}'."

        if action == "add":
            if not config_params.get("documents_context_key") and not config_params.get("embeddings_context_key"):
                return False, "Action 'add' requires either 'documents_context_key' or 'embeddings_context_key'."
            if config_params.get("embeddings_context_key") and not config_params.get("ids_context_key"):
                 return False, "Action 'add' with 'embeddings_context_key' also requires 'ids_context_key'."

        if action == "query":
            if not config_params.get("query_texts_context_key") and not config_params.get("query_embeddings_context_key"):
                return False, "Action 'query' requires either 'query_texts_context_key' or 'query_embeddings_context_key'."

        return True, ""


if __name__ == '__main__':
    # Mock global_config and context for testing
    # Create a temporary directory for ChromaDB persistence during tests
    import tempfile
    import shutil
    test_db_dir = tempfile.mkdtemp()
    print(f"Using temporary ChromaDB path for testing: {test_db_dir}")

    mock_global_config = {
        "settings": {
            "default_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chromadb_path": test_db_dir # Use temp dir for testing
        }
    }
    plugin = ChromaDBPlugin(global_config=mock_global_config)

    # Common context and configs
    collection_name = "test_collection_1"

    # Test Case 1: Create Collection (explicitly)
    step_config_create = {
        "config": {
            "action": "create_collection_only",
            "collection_name": collection_name,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", # optional, uses default
            "distance_function": "cosine" # optional, uses default
        }
    }
    print("\n--- Test Case 1: Create Collection ---")
    results_create = plugin.execute_pipeline_step(step_config_create, {})
    print("Create Results:", results_create)
    assert collection_name in results_create.get("chromadb_create_status", "")

    # Test Case 2: Add documents (ChromaDB embeds)
    context_add_docs = {
        "texts_to_add": ["doc1 text", "doc2 text is longer", "doc3 another one"],
        "text_ids": ["id1", "id2", "id3"],
        "text_meta": [{"source": "A"}, {"source": "B"}, {"source": "A"}]
    }
    step_config_add_docs = {
        "config": {
            "action": "add",
            "collection_name": collection_name,
            "documents_context_key": "texts_to_add",
            "ids_context_key": "text_ids",
            "metadatas_context_key": "text_meta"
        }
    }
    print("\n--- Test Case 2: Add Documents ---")
    results_add_docs = plugin.execute_pipeline_step(step_config_add_docs, context_add_docs)
    print("Add Docs Results:", results_add_docs)
    assert "Added data" in results_add_docs.get("chromadb_add_status", "")
    assert "Count: 3" in results_add_docs.get("chromadb_add_status", "")


    # Test Case 3: Query collection
    context_query = {"my_query": "find text similar to doc1"}
    step_config_query = {
        "config": {
            "action": "query",
            "collection_name": collection_name,
            "query_texts_context_key": "my_query",
            "n_results": 2,
            "include": ["documents", "distances", "metadatas"]
        }
    }
    print("\n--- Test Case 3: Query Collection ---")
    results_query = plugin.execute_pipeline_step(step_config_query, context_query)
    print("Query Results:", results_query.get("chromadb_query_results"))
    assert len(results_query.get("chromadb_query_results", {}).get("documents", [[]])[0]) == 2

    # Test Case 4: Count items
    step_config_count = {"config": {"action": "count_items", "collection_name": collection_name}}
    print("\n--- Test Case 4: Count Items ---")
    results_count = plugin.execute_pipeline_step(step_config_count, {})
    print("Count Results:", results_count)
    assert results_count.get("chromadb_collection_count") == 3

    # Test Case 5: Add pre-computed embeddings (requires EmbeddingGenerator first)
    # Let's quickly mock an embedding generator result
    from sentence_transformers import SentenceTransformer
    mock_embed_model = SentenceTransformer(mock_global_config["settings"]["default_embedding_model"])
    new_docs_for_embed = ["new doc 4", "new doc 5"]
    precomputed_embeddings = mock_embed_model.encode(new_docs_for_embed).tolist()

    context_add_embeds = {
        "new_docs": new_docs_for_embed, # For the documents field in Chroma
        "new_ids": ["id4", "id5"],
        "new_meta": [{"source": "C"}, {"source": "D"}],
        "pre_embeds": precomputed_embeddings
    }
    step_config_add_embeds = {
        "config": {
            "action": "add",
            "collection_name": collection_name,
            "documents_context_key": "new_docs", # Still useful to store original text
            "embeddings_context_key": "pre_embeds",
            "ids_context_key": "new_ids",
            "metadatas_context_key": "new_meta"
        }
    }
    print("\n--- Test Case 5: Add Pre-computed Embeddings ---")
    results_add_embeds = plugin.execute_pipeline_step(step_config_add_embeds, context_add_embeds)
    print("Add Embeds Results:", results_add_embeds)
    assert "Count: 5" in results_add_embeds.get("chromadb_add_status", "")

    # Test Case 6: List collections
    step_config_list = {"config": {"action": "list_collections"}}
    print("\n--- Test Case 6: List Collections ---")
    results_list = plugin.execute_pipeline_step(step_config_list, {})
    print("List Collections Results:", results_list)
    assert any(c['name'] == collection_name for c in results_list.get("chromadb_collections_list", []))

    # Test Case 7: Get specific collection info
    step_config_get_coll = {"config": {"action": "get_collection", "collection_name": collection_name}}
    print("\n--- Test Case 7: Get Collection Info ---")
    results_get_coll = plugin.execute_pipeline_step(step_config_get_coll, {})
    print("Get Collection Info Results:", results_get_coll)
    assert results_get_coll.get("chromadb_collection_info", {}).get("name") == collection_name
    assert results_get_coll.get("chromadb_collection_info", {}).get("count") == 5

    # Test Case 8: Delete collection
    step_config_delete = {"config": {"action": "delete_collection", "collection_name": collection_name}}
    print("\n--- Test Case 8: Delete Collection ---")
    results_delete = plugin.execute_pipeline_step(step_config_delete, {})
    print("Delete Results:", results_delete)
    assert f"Collection '{collection_name}' deleted" in results_delete.get("chromadb_delete_status", "")

    # Verify deletion by listing collections again
    results_list_after_delete = plugin.execute_pipeline_step(step_config_list, {})
    print("List Collections After Delete:", results_list_after_delete)
    assert not any(c['name'] == collection_name for c in results_list_after_delete.get("chromadb_collections_list", []))

    # Clean up the temporary directory
    try:
        shutil.rmtree(test_db_dir)
        print(f"\nCleaned up temporary ChromaDB directory: {test_db_dir}")
    except Exception as e:
        print(f"Error cleaning up temp directory {test_db_dir}: {e}")

```
