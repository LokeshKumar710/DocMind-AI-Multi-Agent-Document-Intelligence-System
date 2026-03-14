import base64
import os
import json
from io import BytesIO
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
import docx
import fitz  # PyMuPDF
import hashlib
from datetime import datetime

# Load environment variables
load_dotenv()

# Get OpenAI API Key
OPENAI_API_KEY = os.getenv("openai_api_key")
print("openai api key", OPENAI_API_KEY)  

# Initialize OpenAI client
model_name = os.getenv("model_name", "gpt-4.1")
print("model name", model_name)

# Create client instance
client = OpenAI(api_key=OPENAI_API_KEY)

# Cache file path for storing processed file results
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_cache.json")

# Cache file path for storing image analysis results (separate from document cache)
IMAGE_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_cache.json")


def load_cache():
    """
    Load the cache from the JSON file.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache file: {e}")
    return {}


def save_cache(cache):
    """
    Save the cache to the JSON file.
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save cache file: {e}")


def load_image_cache():
    """
    Load the image cache from the JSON file.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    try:
        if os.path.exists(IMAGE_CACHE_FILE):
            with open(IMAGE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load image cache file: {e}")
    return {}


def save_image_cache(cache):
    """
    Save the image cache to the JSON file.
    """
    try:
        with open(IMAGE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save image cache file: {e}")


def calculate_image_hash(image_bytes: bytes) -> str:
    """
    Calculate SHA256 hash of image bytes.
    
    Args:
        image_bytes: The raw image bytes
        
    Returns:
        Hexadecimal string of the SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_bytes)
    return sha256_hash.hexdigest()


def get_cached_image_result(image_hash: str):
    """
    Check if an image hash exists in the cache and return the cached analysis.
    
    Args:
        image_hash: The SHA256 hash of the image bytes
        
    Returns:
        The cached result if found, None otherwise
    """
    cache = load_image_cache()
    if image_hash in cache:
        print(f"🖼️ Image cache hit! Using cached analysis for hash: {image_hash[:16]}...")
        return cache[image_hash]["result"]
    return None


def save_image_to_cache(image_hash: str, result: str):
    """
    Save image analysis result to the cache.
    
    Args:
        image_hash: The SHA256 hash of the image bytes
        result: The analysis result from OpenAI
    """
    cache = load_image_cache()
    cache[image_hash] = {
        "result": result,
        "processed_at": datetime.now().isoformat(),
        "model": model_name
    }
    save_image_cache(cache)
    print(f"💾 Image analysis cached for hash: {image_hash[:16]}...")


def get_cached_result(file_hash):
    """
    Check if a file hash exists in the cache and return the cached result.
    
    Args:
        file_hash: The SHA256 hash of the file
        
    Returns:
        The cached result if found, None otherwise
    """
    cache = load_cache()
    if file_hash in cache:
        print(f"✅ Cache hit! Using cached result for hash: {file_hash[:16]}...")
        return cache[file_hash]["result"]
    return None


def save_to_cache(file_hash, result, filename=None):
    """
    Save a processing result to the cache.
    
    Args:
        file_hash: The SHA256 hash of the file
        result: The extracted text content
        filename: Optional original filename for reference
    """
    cache = load_cache()
    cache[file_hash] = {
        "result": result,
        "filename": filename,
        "processed_at": datetime.now().isoformat(),
        "character_count": len(result)
    }
    save_cache(cache)
    print(f"💾 Result cached for hash: {file_hash[:16]}...")


def process_document_with_cache(file_input, filename: str = None):
    """
    Process a document with caching support.
    If the file has been processed before (same hash), return cached result.
    Otherwise, process the file and cache the result.
    
    Args:
        file_input: Either bytes or a file path (str)
        filename: Optional filename to help determine file type when input is bytes
        
    Returns:
        Tuple of (extracted_text, was_cached, file_hash)
    """
    # Get file bytes for hashing
    if isinstance(file_input, str):
        with open(file_input, 'rb') as f:
            file_bytes = f.read()
        actual_filename = os.path.basename(file_input)
    else:
        file_bytes = file_input
        actual_filename = filename
    
    # Calculate hash
    file_hash = calculate_file_hash(file_bytes)
    print(f"📝 File hash: {file_hash}")
    
    # Check cache
    cached_result = get_cached_result(file_hash)
    if cached_result is not None:
        return cached_result, True, file_hash
    
    # Not in cache, process the document
    print(f"🔄 Processing new file...")
    result = process_document(file_input, filename)
    
    # Save to cache
    save_to_cache(file_hash, result, actual_filename)
    
    return result, False, file_hash


def clear_cache():
    """
    Clear all cached results (both file and image caches).
    """
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("🗑️ File cache cleared.")
    else:
        print("File cache is already empty.")
    
    if os.path.exists(IMAGE_CACHE_FILE):
        os.remove(IMAGE_CACHE_FILE)
        print("🗑️ Image cache cleared.")
    else:
        print("Image cache is already empty.")


def list_cache():
    """
    List all cached files with their metadata.
    """
    cache = load_cache()
    if not cache:
        print("Cache is empty.")
        return
    
    print(f"\n📦 Cached files ({len(cache)} entries):")
    print("-" * 60)
    for file_hash, data in cache.items():
        filename = data.get('filename', 'Unknown')
        processed_at = data.get('processed_at', 'Unknown')
        char_count = data.get('character_count', 0)
        print(f"  Hash: {file_hash[:16]}...")
        print(f"  File: {filename}")
        print(f"  Processed: {processed_at}")
        print(f"  Characters: {char_count}")
        print("-" * 60)
    
    # Also list image cache
    image_cache = load_image_cache()
    if image_cache:
        print(f"\n🖼️ Cached images ({len(image_cache)} entries):")
        print("-" * 60)
        for img_hash, data in image_cache.items():
            processed_at = data.get('processed_at', 'Unknown')
            model_used = data.get('model', 'Unknown')
            print(f"  Hash: {img_hash[:16]}...")
            print(f"  Model: {model_used}")
            print(f"  Processed: {processed_at}")
            print("-" * 60)


def analyze_image_with_openai(image_bytes):
    """Send image bytes to OpenAI and get detailed architecture diagram insights.
    Uses caching to avoid re-analyzing the same image."""
    
    # Calculate hash of image bytes FIRST
    image_hash = calculate_image_hash(image_bytes)
    print(f"🖼️ Image hash: {image_hash[:16]}...")
    
    # Check image cache
    cached_result = get_cached_image_result(image_hash)
    if cached_result is not None:
        return cached_result
    
    # Not in cache, call OpenAI API
    print(f"🔄 Analyzing new image with OpenAI...")
    
    # Comprehensive prompt for multi-agent document intelligence
    prompt = """
You are an expert in analyzing various types of images, including architecture diagrams, graphs, AI-generated visuals, and other types of images. Your task is to analyze the provided image and extract relevant insights based on its type.

INSTRUCTIONS:

1. **If the image is a non-architectural image** (e.g., a photograph, illustration, futuristic AI concept, abstract art, or any other type of non-architectural graphic):
   - Respond with: **"NON_ARCHITECTURAL_IMAGE"** and provide a general description of the image (e.g., theme, subjects, colors, style, etc.).

2. **If the image is an architecture diagram** (e.g., system architecture, network topology, data flow, microservices, cloud infrastructure, etc.):
   - Respond with the following structured analysis:

     **ANALYSIS FORMAT (for architecture diagrams):**
     1. **OVERVIEW**
        - Type of architecture diagram (e.g., system architecture, network topology, data flow, microservices, cloud infrastructure)
        - Overall purpose and scope of the system depicted

     2. **KEY COMPONENTS**
        - List each major component/service identified in the diagram
        - Briefly describe the role and responsibility of each component

     3. **DATA FLOW & INTERACTIONS**
        - Describe how data flows between components or services
        - Identify communication protocols or interaction patterns (e.g., REST, gRPC, message queues)

     4. **DEPENDENCIES & RELATIONSHIPS**
        - List dependencies between components, both internal and external
        - Identify any external systems or third-party services integrated with the architecture

     5. **RISKS & OPEN QUESTIONS**
        - Single points of failure
        - Missing or unclear components
        - Scalability or performance concerns
        - Security or privacy concerns

3. **If the image is a graph or data visualization** (e.g., bar chart, line chart, scatter plot, pie chart, etc.):
   - Extract the following insights:
     - Key trends or patterns visible in the graph
     - Significant data points (e.g., peaks, valleys, outliers)
     - Correlations, anomalies, or insights derived from the data
     - Conclusions that can be drawn based on the graph's data

4. **For all other types of images** (e.g., artistic works, creative designs, AI-generated art):
   - Provide a brief description or classification based on the visual elements such as the theme, subjects, and artistic style.

OUTPUT RULES:
- Provide a plain-text response formatted as outlined above.
- For non-architectural images, respond with **"NON_ARCHITECTURAL_IMAGE"** along with a general description.
- If the image is an architecture diagram or graph, provide detailed insights in the corresponding format.
- Keep the response concise, clear, and relevant to the image's content.
"""


    try:
        # Encode image bytes to base64 for OpenAI API
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use OpenAI's vision API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        result = response.choices[0].message.content.strip()
        final_result = f"Based on the diagram:\n{result}"
        
        # Save to image cache
        save_image_to_cache(image_hash, final_result)
        
        return final_result
    
    except Exception as e:
        print(f"Error analyzing image with OpenAI: {e}")
        return None


def process_docx(file_bytes: bytes):
    try:
        file_stream = BytesIO(file_bytes)
        file_stream.seek(0)
        doc = docx.Document(file_stream)
        text_data = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                text_data.append(text)

        # Extract images
        for rel in doc.part.rels.values():
            if hasattr(rel, 'target_ref') and "image" in rel.target_ref:
                image_data = getattr(rel.target_part, 'blob', None)
                if image_data:
                    desc = analyze_image_with_openai(image_data)
                    if desc:  # Only add if it's an architectural image
                        text_data.append(f"[Image Analysis]: {desc}")

        return "\n".join(text_data)
    except Exception as e:
        print(f"Error processing docx from bytes: {e}")
        return ""


def process_pdf(file_input):
    """
    Process a PDF file from a file path or bytes, extracting text and image descriptions
    in visual order (top-to-bottom, left-to-right).
    """
    # Open PDF from bytes or file path
    if isinstance(file_input, bytes):
        file_stream = BytesIO(file_input)
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    else:
        doc = fitz.open(file_input)

    output = []

    for page in doc:
        # Get text blocks: (x0, y0, x1, y1, "text")
        text_blocks = [
            (b[0], b[1], b[2], b[3], b[4])
            for b in page.get_text("blocks")
            if b[4].strip()  # ignore empty
        ]

        # Get image blocks with coordinates
        image_blocks = []
        for img in page.get_images(full=True):
            xref = img[0]
            bbox = page.get_image_bbox(img)  # (x0, y0, x1, y1)
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            desc = analyze_image_with_openai(image_bytes)
            if desc:  # Only add if it's an architectural image
                image_blocks.append((bbox[0], bbox[1], bbox[2], bbox[3], f"[Image Analysis]: {desc}"))

        # Merge text & image blocks
        all_blocks = text_blocks + image_blocks

        # Sort top-to-bottom, then left-to-right
        all_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

        # Append content in visual order
        for block in all_blocks:
            output.append(block[4])

    return "\n".join(output)


def process_text(file_input):
    """
    Process a plain text file from a file path or bytes.
    """
    try:
        if isinstance(file_input, bytes):
            text = file_input.decode('utf-8')
        else:
            with open(file_input, 'r', encoding='utf-8') as f:
                text = f.read()
        
        return text.strip()
    except Exception as e:
        print(f"Error processing text file: {e}")
        return ""


def process_vtt(file_input):
    """
    Process a VTT (WebVTT) subtitle file, extracting only the text content.
    Removes timestamps and metadata, keeping just the spoken content.
    """
    try:
        if isinstance(file_input, bytes):
            content = file_input.decode('utf-8')
        else:
            with open(file_input, 'r', encoding='utf-8') as f:
                content = f.read()
        
        lines = content.split('\n')
        text_lines = []
        skip_next = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip WEBVTT header
            if line.startswith('WEBVTT'):
                continue
            
            # Skip NOTE comments
            if line.startswith('NOTE'):
                skip_next = True
                continue
            
            if skip_next:
                skip_next = False
                continue
            
            # Skip cue identifiers (lines that are just numbers or contain -->)
            if line.isdigit():
                continue
            
            # Skip timestamp lines (contain -->)
            if '-->' in line:
                continue
            
            # Skip lines that look like cue settings
            if line.startswith('align:') or line.startswith('position:'):
                continue
            
            # This should be actual subtitle text
            # Remove HTML-like tags if present (e.g., <v Speaker>)
            import re
            clean_line = re.sub(r'<[^>]+>', '', line)
            
            if clean_line.strip():
                text_lines.append(clean_line.strip())
        
        return "\n".join(text_lines)
    except Exception as e:
        print(f"Error processing VTT file: {e}")
        return ""


def process_document(file_input, filename: str = None):
    """
    Unified document processor that accepts PDF, DOCX, TXT, or VTT files.
    Automatically detects file type and routes to the appropriate processor.
    
    Args:
        file_input: Either bytes or a file path (str)
        filename: Optional filename to help determine file type when input is bytes
        
    Returns:
        Extracted text content from the document
    """
    # Determine file extension
    if isinstance(file_input, str):
        # It's a file path
        ext = os.path.splitext(file_input)[1].lower()
    elif filename:
        # Use provided filename for extension
        ext = os.path.splitext(filename)[1].lower()
    else:
        raise ValueError("When passing bytes, you must provide a filename to determine file type")
    
    # Route to appropriate processor
    if ext == '.pdf':
        return process_pdf(file_input)
    elif ext == '.docx':
        if isinstance(file_input, str):
            # Read file as bytes first
            with open(file_input, 'rb') as f:
                file_input = f.read()
        return process_docx(file_input)
    elif ext == '.txt':
        return process_text(file_input)
    elif ext == '.vtt':
        return process_vtt(file_input)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .docx, .txt, .vtt")

def calculate_file_hash(file_input):
    """
    Calculate the SHA256 hash of a file.
    
    Args:
        file_input: Either bytes or a file path (str)
        
    Returns:
        Hexadecimal string of the SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    
    if isinstance(file_input, str):
        # It's a file path
        with open(file_input, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    else:
        # It's bytes
        sha256_hash.update(file_input)
    
    return sha256_hash.hexdigest()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Handle cache management commands
        if arg == "--clear-cache":
            clear_cache()
        elif arg == "--list-cache":
            list_cache()
        else:
            # Process file with caching
            filepath = arg
            print(f"Processing: {filepath}")
            
            # Use the cached version
            result, was_cached, file_hash = process_document_with_cache(filepath)
            
            # Generate output filename (same name as input but with .txt extension)
            base_name = os.path.splitext(filepath)[0]
            output_filename = f"{base_name}_extracted.txt"
            
            # Save to text file
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(result)
            
            if was_cached:
                print(f"\n⚡ Result retrieved from cache (instant!)")
            else:
                print(f"\n🔄 File was processed and cached for future use")
            
            print(f"\n✅ Extracted content saved to: {output_filename}")
            print(f"📄 Total characters: {len(result)}")
            print(f"🔑 File hash: {file_hash}")
            
            # Also print preview
            print("\n--- Preview (first 500 chars) ---\n")
            print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print("Usage: python extract.py <file_path>")
        print("       python extract.py --clear-cache   (clear all cached results)")
        print("       python extract.py --list-cache    (list all cached files)")
        print("Supported formats: .pdf, .docx, .txt, .vtt")