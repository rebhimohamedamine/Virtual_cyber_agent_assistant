from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Qdrant config
QDRANT_URL = "https://69b645bb-c816-4603-8168-c853b1ed8d04.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.a2djLlmlZy-BG6pkzT8NHpDnx9bVgAgX8moB7Xuau0g"
COLLECTION_NAME = "knowledge_base_v1"

# Qdrant & embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
QDRANT_REQUEST_TIMEOUT = 3600  # seconds
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_REQUEST_TIMEOUT)

# Collection creation
def create_qdrant_collection():
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created or recreated successfully.")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
        else:
            logger.error(f"Failed to create or recreate collection '{COLLECTION_NAME}': {e}")
            raise e

# Store function
def store_in_qdrant(chunks):
    points = []
    for chunk in chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Skip chunks with empty or very short content
        if not text or len(text.strip()) < 10:
            logger.warning(f"Skipping chunk {metadata.get('chunk_id', 'unknown')} - content too short")
            continue
        
        try:
            embedding = embedder.encode(text).tolist()
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": text,  # Store the actual content
                        **metadata  # Spread all metadata
                    }
                )
            )
            logger.debug(f"Encoded chunk {metadata.get('chunk_id', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to encode chunk {metadata.get('chunk_id', 'unknown')}: {e}")
            continue
    
    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Stored {len(points)} chunks in Qdrant.")
        return len(points)
    except Exception as e:
        logger.error(f"Failed to store chunks in Qdrant: {e}")
        raise e

# Convert chunks array into format for Qdrant storage
def process_chunks_from_json(json_data):
    """
    Process just the chunks array from the enhanced chunker.
    Expected structure: {"chunks": [{"id": 1, "content": "...", "section": "...", "level": 1, "metadata": {...}}]}
    """
    chunks_for_storage = []
    
    # Get chunks array from the JSON
    chunks = json_data.get("chunks", [])
    
    for chunk in chunks:
        # Extract content directly from chunk (not nested)
        content = chunk.get("content", "").strip()
        chunk_metadata = chunk.get("metadata", {})
        
        # Skip empty chunks
        if not content or len(content) < 10:
            logger.warning(f"Skipping empty chunk: {chunk.get('id', 'unknown')}")
            continue
        
        # Prepare metadata for storage using the actual structure
        storage_metadata = {
            "chunk_id": chunk.get("id"),
            "section": chunk.get("section", ""),
            "level": chunk.get("level", 1),
            "section_number": chunk_metadata.get("sectionNumber", ""),
            "title": chunk_metadata.get("title", ""),
            "parent_section": chunk_metadata.get("parentSection", ""),
            "parent_subsection": chunk_metadata.get("parentSubsection", ""),
            "chunk_part": chunk_metadata.get("chunkPart"),
            "total_parts": chunk_metadata.get("totalParts")
        }
        
        # Remove None values and empty strings to keep metadata clean
        storage_metadata = {k: v for k, v in storage_metadata.items() if v is not None and v != ""}
        
        chunks_for_storage.append({
            "text": content,
            "metadata": storage_metadata
        })
    
    logger.info(f"Processed {len(chunks_for_storage)} chunks from {len(chunks)} input chunks")
    return chunks_for_storage

# Process chunks directly from array
def process_chunks_directly(chunks_array):
    """
    Process chunks array directly
    """
    chunks_for_storage = []
    
    for chunk in chunks_array:
        # Extract content directly from chunk
        content = chunk.get("content", "").strip()
        chunk_metadata = chunk.get("metadata", {})
        
        # Skip empty chunks
        if not content or len(content) < 10:
            logger.warning(f"Skipping empty chunk: {chunk.get('id', 'unknown')}")
            continue
        
        # Prepare metadata for storage
        storage_metadata = {
            "chunk_id": chunk.get("id"),
            "section": chunk.get("section", ""),
            "level": chunk.get("level", 1),
            "section_number": chunk_metadata.get("sectionNumber", ""),
            "title": chunk_metadata.get("title", ""),
            "parent_section": chunk_metadata.get("parentSection", ""),
            "parent_subsection": chunk_metadata.get("parentSubsection", ""),
            "chunk_part": chunk_metadata.get("chunkPart"),
            "total_parts": chunk_metadata.get("totalParts")
        }
        
        # Remove None values and empty strings
        storage_metadata = {k: v for k, v in storage_metadata.items() if v is not None and v != ""}
        
        chunks_for_storage.append({
            "text": content,
            "metadata": storage_metadata
        })
    
    logger.info(f"Processed {len(chunks_for_storage)} chunks from {len(chunks_array)} input chunks")
    return chunks_for_storage

# Endpoint for storing chunks
@app.route("/store_chunks", methods=["POST"])
def store_chunks_api():
    """
    Store document chunks from the enhanced chunker.
    Flexible to handle multiple formats including single chunks
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Debug: Log what we actually received
    logger.info(f"Received data type: {type(data)}")
    logger.info(f"Received data keys (if dict): {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    logger.info(f"Received data sample: {str(data)[:500]}...")  # First 500 chars

    chunks_data = None

    # Handle multiple possible formats
    if isinstance(data, list):
        # Direct array format: [chunk1, chunk2, ...]
        chunks_data = data
        logger.info("Using direct array format")
    elif isinstance(data, dict):
        # Try different possible key names
        if "chunks" in data:
            chunks_data = data["chunks"]
            logger.info("Found 'chunks' key")
        elif "json" in data and isinstance(data["json"], dict) and "chunks" in data["json"]:
            chunks_data = data["json"]["chunks"]
            logger.info("Found 'chunks' in nested 'json' object")
        elif "sections" in data or ("chunks" in data and isinstance(data["chunks"], list)):
            # This looks like the full output from your chunker
            chunks_data = data.get("chunks", [])
            logger.info("Found chunker output format")
        else:
            # NEW: Check if this is a single chunk object
            if "content" in data and "id" in data:
                chunks_data = [data]  # Wrap single chunk in array
                logger.info("Treating single object as one chunk")

    if not chunks_data:
        return jsonify({
            "error": "Could not find chunks data",
            "received_keys": list(data.keys()) if isinstance(data, dict) else "not_a_dict",
            "data_type": str(type(data)),
            "help": "Send either: {'chunks': [chunk1, chunk2]} or [chunk1, chunk2] or single chunk object"
        }), 400

    if not isinstance(chunks_data, list):
        return jsonify({"error": "Chunks data must be an array"}), 400

    try:
        # Process the chunks array
        processed_chunks = process_chunks_directly(chunks_data)
        
        if not processed_chunks:
            return jsonify({"error": "No valid chunks found after processing"}), 400

        # Store in Qdrant
        stored_count = store_in_qdrant(processed_chunks)
        
        return jsonify({
            "message": f"Successfully stored {stored_count} chunks in Qdrant",
            "stored_chunks": stored_count,
            "processed_from": len(chunks_data)
        }), 200

    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        return jsonify({"error": f"Failed to process chunks: {str(e)}"}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    try:
        # Test Qdrant connection
        collections = qdrant_client.get_collections()
        return jsonify({
            "status": "healthy",
            "qdrant_connected": True,
            "collection_exists": COLLECTION_NAME in [c.name for c in collections.collections]
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Ensure the collection is created before the Flask app starts serving requests
    create_qdrant_collection()
    app.run(debug=True)