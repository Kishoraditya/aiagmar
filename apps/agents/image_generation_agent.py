"""
Image Generation Agent

This agent generates images or diagrams based on research findings, enhancing visual representation.
It relies on EverArt MCP for AI image generation, leveraging creative capabilities.
"""

import os
import uuid
import json
import time
import re
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Import MCP clients
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.mcps.filesystem_mcp import FilesystemMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_generation_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class ImageGenerationRequest(TypedDict):
    """Structure for image generation requests."""
    prompt: str
    style: Optional[str]
    aspect_ratio: Optional[str]
    num_images: Optional[int]
    context: Optional[str]
    reference_text: Optional[str]
    reference_images: Optional[List[str]]
    save_path: Optional[str]
    namespace: Optional[str]
    session_id: Optional[str]


class ImageEnhancementRequest(TypedDict):
    """Structure for image enhancement requests."""
    image_url: str
    prompt: str
    strength: Optional[float]
    save_path: Optional[str]
    namespace: Optional[str]
    session_id: Optional[str]


class ImageDescriptionRequest(TypedDict):
    """Structure for image description requests."""
    image_url: str
    detail_level: Optional[str]
    namespace: Optional[str]
    session_id: Optional[str]


class GeneratedImage(TypedDict):
    """Structure for generated image information."""
    image_url: str
    prompt: str
    style: Optional[str]
    aspect_ratio: str
    generation_time: str
    local_path: Optional[str]


class EnhancedImage(TypedDict):
    """Structure for enhanced image information."""
    original_url: str
    enhanced_url: str
    prompt: str
    strength: float
    enhancement_time: str
    local_path: Optional[str]


class ImageDescription(TypedDict):
    """Structure for image description information."""
    image_url: str
    description: str
    detail_level: str
    description_time: str


class ImageGenerationResponse(TypedDict):
    """Structure for image generation responses."""
    images: List[GeneratedImage]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class ImageEnhancementResponse(TypedDict):
    """Structure for image enhancement responses."""
    enhanced_image: EnhancedImage
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class ImageDescriptionResponse(TypedDict):
    """Structure for image description responses."""
    description: ImageDescription
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the Image Generation Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.everart_mcp = None
        self.memory_mcp = None
        self.filesystem_mcp = None
    
    def get_everart_mcp(self) -> EverArtMCP:
        """Get or create EverArt MCP client."""
        if self.everart_mcp is None:
            api_key = os.environ.get("EVERART_API_KEY")
            if not api_key:
                raise ValueError("EVERART_API_KEY environment variable is required")
            self.everart_mcp = EverArtMCP(api_key=api_key)
        return self.everart_mcp
    
    def get_memory_mcp(self) -> MemoryMCP:
        """Get or create Memory MCP client."""
        if self.memory_mcp is None:
            storage_path = os.environ.get("MEMORY_STORAGE_PATH", "./memory_storage")
            self.memory_mcp = MemoryMCP(storage_path=storage_path)
        return self.memory_mcp
    
    def get_filesystem_mcp(self) -> FilesystemMCP:
        """Get or create Filesystem MCP client."""
        if self.filesystem_mcp is None:
            workspace_dir = os.environ.get("WORKSPACE_DIR", "./workspace")
            self.filesystem_mcp = FilesystemMCP(workspace_dir=workspace_dir)
        return self.filesystem_mcp
    
    def close_all(self):
        """Close all MCP clients."""
        if self.everart_mcp:
            self.everart_mcp.close()
        if self.memory_mcp:
            self.memory_mcp.close()
        if self.filesystem_mcp:
            self.filesystem_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Image Generation Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.7):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def enhance_image_prompt(prompt: str, context: Optional[str] = None, 
                        reference_text: Optional[str] = None) -> str:
    """
    Enhance an image generation prompt to create better images.
    
    Args:
        prompt: Base image prompt
        context: Optional context for the image
        reference_text: Optional reference text to incorporate
        
    Returns:
        Enhanced image prompt
    """
    llm = create_llm(temperature=0.7)
    
    # Create prompt for enhancing the image prompt
    system_message = """You are an expert at creating detailed, effective prompts for AI image generation. 
    Your task is to enhance the given basic prompt into a more detailed, descriptive prompt that will 
    produce high-quality, visually appealing images.
    
    Consider these aspects in your enhancement:
    1. Visual details (colors, lighting, composition)
    2. Style and mood
    3. Subject details and positioning
    4. Background elements
    
    Do not add inappropriate content. Keep the enhanced prompt focused on the original intent but make it 
    more descriptive and effective for image generation. Return only the enhanced prompt text without 
    explanations or formatting."""
    
    # Add context and reference text to system message if provided
    if context:
        system_message += f"\n\nContext for the image: {context}"
    if reference_text:
        system_message += f"\n\nReference text to incorporate: {reference_text}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content="Basic prompt: {prompt}")
    ])
    
    # Execute the prompt
    chain = prompt_template | llm | StrOutputParser()
    enhanced_prompt = chain.invoke({"prompt": prompt})
    
    return enhanced_prompt


def download_image(image_url: str, save_path: str) -> str:
    """
    Download an image from a URL and save it locally.
    
    Args:
        image_url: URL of the image to download
        save_path: Directory path to save the image
        
    Returns:
        Local file path of the saved image
    """
    try:
        # Parse URL to get filename
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        
        # If filename doesn't have an extension, add .png
        if not os.path.splitext(filename)[1]:
            filename += ".png"
        
        # If filename is empty or invalid, generate a random one
        if not filename or filename == ".png":
            filename = f"image_{uuid.uuid4()}.png"
        
        # Create full path
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, filename)
        
        # Download the image
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading image from {image_url}: {e}")
        raise


def save_image_with_filesystem_mcp(image_url: str, directory: str, filename: Optional[str] = None) -> str:
    """
    Save an image using the Filesystem MCP.
    
    Args:
        image_url: URL of the image to save
        directory: Directory to save the image in
        filename: Optional filename to use
        
    Returns:
        Path where the image was saved
    """
    try:
        # Get Filesystem MCP client
        fs_mcp = mcp_manager.get_filesystem_mcp()
        
        # Create directory if it doesn't exist
        try:
            fs_mcp.create_directory(directory)
        except Exception as e:
            # Directory might already exist
            logger.debug(f"Note when creating directory: {e}")
        
        # Generate filename if not provided
        if not filename:
            # Parse URL to get filename
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            
            # If filename doesn't have an extension, add .png
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            
            # If filename is empty or invalid, generate a random one
            if not filename or filename == ".png":
                filename = f"image_{uuid.uuid4()}.png"
        
        # Full path
        file_path = os.path.join(directory, filename)
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Convert image data to base64 for storage
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        # Save image data using Filesystem MCP
        # Note: This is a workaround since we can't directly write binary data
        # In a real implementation, you might want to use a different approach
        fs_mcp.write_file(file_path, image_data)
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving image from {image_url}: {e}")
        raise


def extract_image_urls_from_response(response_text: str) -> List[str]:
    """
    Extract image URLs from EverArt MCP response text.
    
    Args:
        response_text: Response text from EverArt MCP
        
    Returns:
        List of image URLs
    """
    # Common URL patterns
    url_patterns = [
        r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)(?:\?\S*)?',  # Standard image URLs
        r'https?://\S+/images/\S+',  # Common image path pattern
        r'https?://\S+/image/\S+',   # Alternative image path
        r'https?://\S+/img/\S+'      # Another common pattern
    ]
    
    # Extract URLs using patterns
    all_urls = []
    for pattern in url_patterns:
        urls = re.findall(pattern, response_text, re.IGNORECASE)
        all_urls.extend(urls)
    
    # If no URLs found with patterns, try a more general approach
    if not all_urls:
        # Look for URLs in specific formats from the response
        lines = response_text.split('\n')
        for line in lines:
            if 'http' in line and ('image' in line.lower() or 'url' in line.lower()):
                # Extract URL from line
                url_match = re.search(r'(https?://\S+)', line)
                if url_match:
                    all_urls.append(url_match.group(1))
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in all_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    return unique_urls


def store_image_metadata(image_info: Union[GeneratedImage, EnhancedImage, ImageDescription], 
                        namespace: str, key_prefix: str) -> str:
    """
    Store image metadata in memory.
    
    Args:
        image_info: Image information to store
        namespace: Namespace for storage
        key_prefix: Prefix for the memory key
        
    Returns:
        Memory key where metadata is stored
    """
    memory_mcp = mcp_manager.get_memory_mcp()
    
    # Generate a key
    image_id = str(uuid.uuid4())
    key = f"{key_prefix}_{image_id}"
    
    # Store the metadata
    memory_mcp.store_memory(
        key=key,
        value=json.dumps(image_info),
        namespace=namespace
    )
    
    return key


# -----------------------------------------------------------------------------
# Image Generation Agent Class
# -----------------------------------------------------------------------------

class ImageGenerationAgent:
    """
    Image Generation Agent that creates images based on prompts and research findings.
    """
    
    def __init__(self):
        """Initialize the Image Generation Agent."""
        pass
    
    def generate_images(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate images based on a prompt.
        
        Args:
            request: Image generation request
            
        Returns:
            Image generation response
        """
        start_time = time.time()
        
        # Initialize response
        response: ImageGenerationResponse = {
            "images": [],
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "prompt_enhancement_performed": False,
                "num_images_requested": request.get("num_images", 1),
                "num_images_generated": 0
            },
            "errors": []
        }
        
        try:
            # Get EverArt MCP client
            everart_mcp = mcp_manager.get_everart_mcp()
            
            # Enhance prompt if context or reference text is provided
            original_prompt = request["prompt"]
            enhanced_prompt = original_prompt
            
            if request.get("context") or request.get("reference_text"):
                try:
                    enhanced_prompt = enhance_image_prompt(
                        prompt=original_prompt,
                        context=request.get("context"),
                        reference_text=request.get("reference_text")
                    )
                    response["execution_stats"]["prompt_enhancement_performed"] = True
                except Exception as e:
                    logger.error(f"Error enhancing prompt: {e}")
                    response["errors"].append({
                        "type": "prompt_enhancement_error",
                        "error": str(e)
                    })
            
            # Set default values if not provided
            style = request.get("style")
            aspect_ratio = request.get("aspect_ratio", "1:1")
            num_images = request.get("num_images", 1)
            
            # Generate images
            logger.info(f"Generating {num_images} images with prompt: {enhanced_prompt[:50]}...")
            
            generation_result = everart_mcp.generate_image(
                prompt=enhanced_prompt,
                style=style,
                aspect_ratio=aspect_ratio,
                num_images=num_images
            )
            
            # Extract image URLs from the response
            image_urls = extract_image_urls_from_response(generation_result)
            
            # Process each generated image
            for i, image_url in enumerate(image_urls):
                try:
                    # Save image locally if save_path is provided
                    local_path = None
                    if request.get("save_path"):
                        try:
                            # Create directory structure: save_path/YYYY-MM-DD/images/
                            date_dir = time.strftime("%Y-%m-%d")
                            image_dir = os.path.join(request["save_path"], date_dir, "images")
                            
                            # Save using Filesystem MCP if available
                            try:
                                filename = f"image_{time.strftime('%H%M%S')}_{i+1}.png"
                                local_path = save_image_with_filesystem_mcp(
                                    image_url=image_url,
                                    directory=image_dir,
                                    filename=filename
                                )
                            except Exception as e:
                                # Fallback to direct download
                                logger.warning(f"Filesystem MCP save failed, falling back to direct download: {e}")
                                local_path = download_image(image_url, image_dir)
                        
                        except Exception as e:
                            logger.error(f"Error saving image: {e}")
                            response["errors"].append({
                                "type": "image_save_error",
                                "error": str(e)
                            })
                    
                    # Create image info
                    image_info: GeneratedImage = {
                        "image_url": image_url,
                        "prompt": enhanced_prompt,
                        "style": style,
                        "aspect_ratio": aspect_ratio,
                        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "local_path": local_path
                    }
                    
                    # Store metadata in memory if namespace provided
                    if request.get("namespace"):
                        try:
                            store_image_metadata(
                                image_info=image_info,
                                namespace=request["namespace"],
                                key_prefix=f"generated_image_{request.get('session_id', '')}"
                            )
                        except Exception as e:
                            logger.error(f"Error storing image metadata: {e}")
                            response["errors"].append({
                                "type": "metadata_storage_error",
                                "error": str(e)
                            })
                    
                    # Add to response
                    response["images"].append(image_info)
                
                except Exception as e:
                    logger.error(f"Error processing image {i+1}: {e}")
                    response["errors"].append({
                        "type": "image_processing_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["num_images_generated"] = len(response["images"])
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def enhance_image(self, request: ImageEnhancementRequest) -> ImageEnhancementResponse:
        """
        Enhance an existing image based on a prompt.
        
        Args:
            request: Image enhancement request
            
        Returns:
            Image enhancement response
        """
        start_time = time.time()
        
        # Initialize response
        response: ImageEnhancementResponse = {
            "enhanced_image": {
                "original_url": request["image_url"],
                "enhanced_url": "",
                "prompt": request["prompt"],
                "strength": request.get("strength", 0.5),
                "enhancement_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "local_path": None
            },
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get EverArt MCP client
            everart_mcp = mcp_manager.get_everart_mcp()
            
            # Set default strength if not provided
            strength = request.get("strength", 0.5)
            
            # Enhance image
            logger.info(f"Enhancing image with prompt: {request['prompt'][:50]}...")
            
            enhancement_result = everart_mcp.enhance_image(
                image_url=request["image_url"],
                prompt=request["prompt"],
                strength=strength
            )
            
            # Extract image URL from the response
            enhanced_urls = extract_image_urls_from_response(enhancement_result)
            
            if not enhanced_urls:
                raise ValueError("No enhanced image URL found in the response")
            
            enhanced_url = enhanced_urls[0]
            
            # Save image locally if save_path is provided
            local_path = None
            if request.get("save_path"):
                try:
                    # Create directory structure: save_path/YYYY-MM-DD/enhanced/
                    date_dir = time.strftime("%Y-%m-%d")
                    image_dir = os.path.join(request["save_path"], date_dir, "enhanced")
                    
                    # Save using Filesystem MCP if available
                    try:
                        filename = f"enhanced_{time.strftime('%H%M%S')}.png"
                        local_path = save_image_with_filesystem_mcp(
                            image_url=enhanced_url,
                            directory=image_dir,
                            filename=filename
                        )
                    except Exception as e:
                        # Fallback to direct download
                        logger.warning(f"Filesystem MCP save failed, falling back to direct download: {e}")
                        local_path = download_image(enhanced_url, image_dir)
                
                except Exception as e:
                    logger.error(f"Error saving enhanced image: {e}")
                    response["errors"].append({
                        "type": "image_save_error",
                        "error": str(e)
                    })
            
            # Update response
            response["enhanced_image"]["enhanced_url"] = enhanced_url
            response["enhanced_image"]["local_path"] = local_path
            
            # Store metadata in memory if namespace provided
            if request.get("namespace"):
                try:
                    store_image_metadata(
                        image_info=response["enhanced_image"],
                        namespace=request["namespace"],
                        key_prefix=f"enhanced_image_{request.get('session_id', '')}"
                    )
                except Exception as e:
                    logger.error(f"Error storing enhanced image metadata: {e}")
                    response["errors"].append({
                        "type": "metadata_storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in image enhancement: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def describe_image(self, request: ImageDescriptionRequest) -> ImageDescriptionResponse:
        """
        Generate a description of an image.
        
        Args:
            request: Image description request
            
        Returns:
            Image description response
        """
        start_time = time.time()
        
        # Initialize response
        response: ImageDescriptionResponse = {
            "description": {
                "image_url": request["image_url"],
                "description": "",
                "detail_level": request.get("detail_level", "medium"),
                "description_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get EverArt MCP client
            everart_mcp = mcp_manager.get_everart_mcp()
            
            # Set default detail level if not provided
            detail_level = request.get("detail_level", "medium")
            
            # Describe image
            logger.info(f"Generating description for image with detail level: {detail_level}...")
            
            description_result = everart_mcp.describe_image(
                image_url=request["image_url"],
                detail_level=detail_level
            )
            
            # Update response
            response["description"]["description"] = description_result
            
            # Store metadata in memory if namespace provided
            if request.get("namespace"):
                try:
                    store_image_metadata(
                        image_info=response["description"],
                        namespace=request["namespace"],
                        key_prefix=f"image_description_{request.get('session_id', '')}"
                    )
                except Exception as e:
                    logger.error(f"Error storing image description metadata: {e}")
                    response["errors"].append({
                        "type": "metadata_storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in image description: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def generate_diagram_from_text(self, text: str, diagram_type: str = "flowchart", 
                                 save_path: Optional[str] = None, 
                                 namespace: Optional[str] = None,
                                 session_id: Optional[str] = None) -> ImageGenerationResponse:
        """
        Generate a diagram based on text content.
        
        Args:
            text: Text to visualize as a diagram
            diagram_type: Type of diagram to generate
            save_path: Optional path to save the diagram
            namespace: Optional namespace for storing metadata
            session_id: Optional session ID
            
        Returns:
            Image generation response
        """
        # Create a diagram prompt based on the text and diagram type
        llm = create_llm(temperature=0.3)
        
        # Create prompt for generating a diagram prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert at creating prompts for AI image generation, 
            specifically for {diagram_type} diagrams. Your task is to create a detailed prompt that will 
            produce a clear, professional {diagram_type} based on the provided text.
            
            The prompt should:
            1. Specify that this is a {diagram_type} diagram
            2. Include key elements and their relationships from the text
            3. Request a clean, professional style with good readability
            4. Specify any necessary labels, arrows, or connections
            
            Return only the image generation prompt without explanations or formatting."""),
            HumanMessage(content="Text to visualize: {text}")
        ])
        
        # Execute the prompt
        chain = prompt_template | llm | StrOutputParser()
        diagram_prompt = chain.invoke({"text": text})
        
        # Create image generation request
        request: ImageGenerationRequest = {
            "prompt": diagram_prompt,
            "style": "diagram",
            "aspect_ratio": "16:9",
            "num_images": 1,
            "context": f"Creating a {diagram_type} diagram based on text",
            "reference_text": text,
            "save_path": save_path,
            "namespace": namespace,
            "session_id": session_id
        }
        
        # Generate the diagram
        return self.generate_images(request)
    
    def generate_visualization_from_data(self, data: Dict[str, Any], chart_type: str = "bar", 
                                       title: Optional[str] = None,
                                       save_path: Optional[str] = None, 
                                       namespace: Optional[str] = None,
                                       session_id: Optional[str] = None) -> ImageGenerationResponse:
        """
        Generate a data visualization based on structured data.
        
        Args:
            data: Data to visualize
            chart_type: Type of chart to generate
            title: Optional title for the visualization
            save_path: Optional path to save the visualization
            namespace: Optional namespace for storing metadata
            session_id: Optional session ID
            
        Returns:
            Image generation response
        """
        # Convert data to a string representation
        data_str = json.dumps(data, indent=2)
        
        # Create a visualization prompt based on the data and chart type
        llm = create_llm(temperature=0.3)
        
        # Create prompt for generating a visualization prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert at creating prompts for AI image generation, 
            specifically for data visualizations. Your task is to create a detailed prompt that will 
            produce a clear, professional {chart_type} chart based on the provided data.
            
            The prompt should:
            1. Specify that this is a {chart_type} chart visualization
            2. Include the key data points and their relationships
            3. Request a clean, professional style with good readability
            4. Specify necessary labels, legends, and axes
            
            Return only the image generation prompt without explanations or formatting."""),
            HumanMessage(content="Data to visualize: {data}\nTitle: {title}")
        ])
        
        # Execute the prompt
        chain = prompt_template | llm | StrOutputParser()
        visualization_prompt = chain.invoke({
            "data": data_str,
            "title": title or f"{chart_type.capitalize()} Chart Visualization"
        })
        
        # Create image generation request
        request: ImageGenerationRequest = {
            "prompt": visualization_prompt,
            "style": "data_visualization",
            "aspect_ratio": "16:9",
            "num_images": 1,
            "context": f"Creating a {chart_type} chart visualization",
            "reference_text": data_str,
            "save_path": save_path,
            "namespace": namespace,
            "session_id": session_id
        }
        
        # Generate the visualization
        return self.generate_images(request)
    
    def generate_concept_illustration(self, concept: str, style: Optional[str] = None,
                                    reference_text: Optional[str] = None,
                                    save_path: Optional[str] = None, 
                                    namespace: Optional[str] = None,
                                    session_id: Optional[str] = None) -> ImageGenerationResponse:
        """
        Generate an illustration of a concept.
        
        Args:
            concept: Concept to illustrate
            style: Optional style for the illustration
            reference_text: Optional reference text for context
            save_path: Optional path to save the illustration
            namespace: Optional namespace for storing metadata
            session_id: Optional session ID
            
        Returns:
            Image generation response
        """
        # Create a concept illustration prompt
        llm = create_llm(temperature=0.5)
        
        # Create prompt for generating a concept illustration prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at creating prompts for AI image generation, 
            specifically for concept illustrations. Your task is to create a detailed prompt that will 
            produce a clear, visually appealing illustration of the provided concept.
            
            The prompt should:
            1. Describe the concept in visual terms
            2. Include key visual elements that represent the concept
            3. Specify mood, lighting, and composition
            4. Request a clear, engaging visual style
            
            Return only the image generation prompt without explanations or formatting."""),
            HumanMessage(content="Concept to illustrate: {concept}")
        ])
        
        # Add reference text if provided
        if reference_text:
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert at creating prompts for AI image generation, 
                specifically for concept illustrations. Your task is to create a detailed prompt that will 
                produce a clear, visually appealing illustration of the provided concept.
                
                The prompt should:
                1. Describe the concept in visual terms
                2. Include key visual elements that represent the concept
                3. Specify mood, lighting, and composition
                4. Request a clear, engaging visual style
                
                Use the reference text to inform your prompt, extracting relevant details.
                
                Return only the image generation prompt without explanations or formatting."""),
                HumanMessage(content="Concept to illustrate: {concept}"),
                SystemMessage(content="Reference text: {reference_text}")
            ])
        
        # Execute the prompt
        chain = prompt_template | llm | StrOutputParser()
        illustration_prompt = chain.invoke({
            "concept": concept,
            "reference_text": reference_text or ""
        })
        
        # Create image generation request
        request: ImageGenerationRequest = {
            "prompt": illustration_prompt,
            "style": style or "illustration",
            "aspect_ratio": "4:3",
            "num_images": 1,
            "context": f"Creating an illustration of the concept: {concept}",
            "reference_text": reference_text,
            "save_path": save_path,
            "namespace": namespace,
            "session_id": session_id
        }
        
        # Generate the illustration
        return self.generate_images(request)
    
    def cleanup(self):
        """Clean up resources used by the Image Generation Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Image Generation Agent")
    parser.add_argument("--mode", type=str, choices=["generate", "enhance", "describe", "diagram", "visualization", "concept"], 
                      default="generate", help="Operation mode")
    parser.add_argument("--prompt", type=str, help="Image generation prompt")
    parser.add_argument("--image-url", type=str, help="URL of image to enhance or describe")
    parser.add_argument("--style", type=str, help="Style for image generation")
    parser.add_argument("--aspect-ratio", type=str, default="1:1", help="Aspect ratio for image generation")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength for image enhancement")
    parser.add_argument("--detail-level", type=str, default="medium", choices=["low", "medium", "high"], 
                      help="Detail level for image description")
    parser.add_argument("--save-path", type=str, help="Path to save generated images")
    parser.add_argument("--text", type=str, help="Text for diagram generation")
    parser.add_argument("--diagram-type", type=str, default="flowchart", 
                      choices=["flowchart", "sequence", "mindmap", "concept"], help="Type of diagram to generate")
    parser.add_argument("--data-file", type=str, help="JSON file with data for visualization")
    parser.add_argument("--chart-type", type=str, default="bar", 
                      choices=["bar", "line", "pie", "scatter"], help="Type of chart to generate")
    parser.add_argument("--title", type=str, help="Title for visualization")
    parser.add_argument("--concept", type=str, help="Concept to illustrate")
    parser.add_argument("--reference-text", type=str, help="Reference text for context")
    parser.add_argument("--namespace", type=str, default="default", help="Namespace for storing in memory")
    parser.add_argument("--session-id", type=str, help="Session ID for continuity")
    args = parser.parse_args()
    
    # Set environment variables if needed
    if args.save_path:
        os.environ["WORKSPACE_DIR"] = os.path.abspath(args.save_path)
    
    # Create the Image Generation Agent
    agent = ImageGenerationAgent()
    
    try:
        if args.mode == "generate":
            if not args.prompt:
                print("Error: --prompt is required for generate mode")
                sys.exit(1)
            
            # Create request
            request: ImageGenerationRequest = {
                "prompt": args.prompt,
                "style": args.style,
                "aspect_ratio": args.aspect_ratio,
                "num_images": args.num_images,
                "reference_text": args.reference_text,
                "save_path": args.save_path,
                "namespace": args.namespace,
                "session_id": args.session_id
            }
            
            # Generate images
            print(f"Generating {args.num_images} images with prompt: {args.prompt}")
            response = agent.generate_images(request)
            
            # Print results
            print(f"\nGenerated {len(response['images'])} images:")
            for i, image in enumerate(response['images']):
                print(f"{i+1}. URL: {image['image_url']}")
                if image['local_path']:
                    print(f"   Saved to: {image['local_path']}")
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
        
        elif args.mode == "enhance":
            if not args.image_url or not args.prompt:
                print("Error: --image-url and --prompt are required for enhance mode")
                sys.exit(1)
            
            # Create request
            request: ImageEnhancementRequest = {
                "image_url": args.image_url,
                "prompt": args.prompt,
                "strength": args.strength,
                "save_path": args.save_path,
                "namespace": args.namespace,
                "session_id": args.session_id
            }
            
            # Enhance image
            print(f"Enhancing image with prompt: {args.prompt}")
            response = agent.enhance_image(request)
            
            # Print results
            print(f"\nEnhanced image:")
            print(f"Original URL: {response['enhanced_image']['original_url']}")
            print(f"Enhanced URL: {response['enhanced_image']['enhanced_url']}")
            if response['enhanced_image']['local_path']:
                print(f"Saved to: {response['enhanced_image']['local_path']}")
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
        
        elif args.mode == "describe":
            if not args.image_url:
                print("Error: --image-url is required for describe mode")
                sys.exit(1)
            
            # Create request
            request: ImageDescriptionRequest = {
                "image_url": args.image_url,
                "detail_level": args.detail_level,
                "namespace": args.namespace,
                "session_id": args.session_id
            }
            
            # Describe image
            print(f"Generating description for image with detail level: {args.detail_level}")
            response = agent.describe_image(request)
            
            # Print results
            print(f"\nImage description:")
            print(response['description']['description'])
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
        
        elif args.mode == "diagram":
            if not args.text:
                print("Error: --text is required for diagram mode")
                sys.exit(1)
            
            # Generate diagram
            print(f"Generating {args.diagram_type} diagram from text")
            response = agent.generate_diagram_from_text(
                text=args.text,
                diagram_type=args.diagram_type,
                save_path=args.save_path,
                namespace=args.namespace,
                session_id=args.session_id
            )
            
            # Print results
            print(f"\nGenerated diagram:")
            if response['images']:
                print(f"URL: {response['images'][0]['image_url']}")
                if response['images'][0]['local_path']:
                    print(f"Saved to: {response['images'][0]['local_path']}")
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
        
        elif args.mode == "visualization":
            if not args.data_file:
                print("Error: --data-file is required for visualization mode")
                sys.exit(1)
            
            # Load data from file
            try:
                with open(args.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading data file: {e}")
                sys.exit(1)
            
            # Generate visualization
            print(f"Generating {args.chart_type} chart visualization")
            response = agent.generate_visualization_from_data(
                data=data,
                chart_type=args.chart_type,
                title=args.title,
                save_path=args.save_path,
                namespace=args.namespace,
                session_id=args.session_id
            )
            
            # Print results
            print(f"\nGenerated visualization:")
            if response['images']:
                print(f"URL: {response['images'][0]['image_url']}")
                if response['images'][0]['local_path']:
                    print(f"Saved to: {response['images'][0]['local_path']}")
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
        
        elif args.mode == "concept":
            if not args.concept:
                print("Error: --concept is required for concept mode")
                sys.exit(1)
            
            # Generate concept illustration
            print(f"Generating illustration for concept: {args.concept}")
            response = agent.generate_concept_illustration(
                concept=args.concept,
                style=args.style,
                reference_text=args.reference_text,
                save_path=args.save_path,
                namespace=args.namespace,
                session_id=args.session_id
            )
            
            # Print results
            print(f"\nGenerated concept illustration:")
            if response['images']:
                print(f"URL: {response['images'][0]['image_url']}")
                if response['images'][0]['local_path']:
                    print(f"Saved to: {response['images'][0]['local_path']}")
            
            # Print stats
            print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
            if response["errors"]:
                print(f"Encountered {len(response['errors'])} errors")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        agent.cleanup()


# -----------------------------------------------------------------------------
# Interactive Mode
# -----------------------------------------------------------------------------

def run_interactive_mode():
    """Run the Image Generation Agent in interactive mode."""
    print("Image Generation Agent - Interactive Mode")
    print("========================================")
    print("This agent will help you generate, enhance, and describe images.")
    print("Type 'exit' to quit.\n")
    
    # Create the Image Generation Agent
    agent = ImageGenerationAgent()
    
    # Initialize session
    session_id = str(uuid.uuid4())
    namespace = "interactive_session"
    
    try:
        while True:
            # Show menu
            print("\nSelect an operation:")
            print("1. Generate images")
            print("2. Enhance an image")
            print("3. Describe an image")
            print("4. Generate a diagram")
            print("5. Generate a data visualization")
            print("6. Illustrate a concept")
            print("0. Exit")
            
            # Get user choice
            choice = input("\nEnter your choice (0-6): ")
            
            # Check for exit command
            if choice == "0" or choice.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            # Process choice
            if choice == "1":  # Generate images
                prompt = input("Enter image generation prompt: ")
                style = input("Enter style (optional): ")
                aspect_ratio = input("Enter aspect ratio (default 1:1): ") or "1:1"
                num_images_str = input("Enter number of images (default 1): ") or "1"
                save_path = input("Enter save path (optional): ")
                
                try:
                    num_images = int(num_images_str)
                except ValueError:
                    num_images = 1
                
                # Create request
                request: ImageGenerationRequest = {
                    "prompt": prompt,
                    "style": style or None,
                    "aspect_ratio": aspect_ratio,
                    "num_images": num_images,
                    "save_path": save_path or None,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                # Generate images
                print(f"\nGenerating {num_images} images with prompt: {prompt}")
                response = agent.generate_images(request)
                
                # Print results
                print(f"\nGenerated {len(response['images'])} images:")
                for i, image in enumerate(response['images']):
                    print(f"{i+1}. URL: {image['image_url']}")
                    if image['local_path']:
                        print(f"   Saved to: {image['local_path']}")
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            elif choice == "2":  # Enhance an image
                image_url = input("Enter URL of image to enhance: ")
                prompt = input("Enter enhancement prompt: ")
                strength_str = input("Enter enhancement strength (0.0-1.0, default 0.5): ") or "0.5"
                save_path = input("Enter save path (optional): ")
                
                try:
                    strength = float(strength_str)
                except ValueError:
                    strength = 0.5
                
                # Create request
                request: ImageEnhancementRequest = {
                    "image_url": image_url,
                    "prompt": prompt,
                    "strength": strength,
                    "save_path": save_path or None,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                # Enhance image
                print(f"\nEnhancing image with prompt: {prompt}")
                response = agent.enhance_image(request)
                
                # Print results
                print(f"\nEnhanced image:")
                print(f"Original URL: {response['enhanced_image']['original_url']}")
                print(f"Enhanced URL: {response['enhanced_image']['enhanced_url']}")
                if response['enhanced_image']['local_path']:
                    print(f"Saved to: {response['enhanced_image']['local_path']}")
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            elif choice == "3":  # Describe an image
                image_url = input("Enter URL of image to describe: ")
                detail_level = input("Enter detail level (low/medium/high, default medium): ") or "medium"
                
                # Validate detail level
                if detail_level not in ["low", "medium", "high"]:
                    print("Invalid detail level. Using 'medium'.")
                    detail_level = "medium"
                
                # Create request
                request: ImageDescriptionRequest = {
                    "image_url": image_url,
                    "detail_level": detail_level,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                # Describe image
                print(f"\nGenerating description for image with detail level: {detail_level}")
                response = agent.describe_image(request)
                
                # Print results
                print(f"\nImage description:")
                print(response['description']['description'])
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            elif choice == "4":  # Generate a diagram
                text = input("Enter text to visualize as a diagram: ")
                diagram_type = input("Enter diagram type (flowchart/sequence/mindmap/concept, default flowchart): ") or "flowchart"
                save_path = input("Enter save path (optional): ")
                
                # Validate diagram type
                if diagram_type not in ["flowchart", "sequence", "mindmap", "concept"]:
                    print("Invalid diagram type. Using 'flowchart'.")
                    diagram_type = "flowchart"
                
                # Generate diagram
                print(f"\nGenerating {diagram_type} diagram from text")
                response = agent.generate_diagram_from_text(
                    text=text,
                    diagram_type=diagram_type,
                    save_path=save_path or None,
                    namespace=namespace,
                    session_id=session_id
                )
                
                # Print results
                print(f"\nGenerated diagram:")
                if response['images']:
                    print(f"URL: {response['images'][0]['image_url']}")
                    if response['images'][0]['local_path']:
                        print(f"Saved to: {response['images'][0]['local_path']}")
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            elif choice == "5":  # Generate a data visualization
                print("Enter data in JSON format (end with an empty line):")
                data_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    data_lines.append(line)
                
                data_str = "\n".join(data_lines)
                
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON data: {e}")
                    continue
                
                chart_type = input("Enter chart type (bar/line/pie/scatter, default bar): ") or "bar"
                title = input("Enter chart title (optional): ")
                save_path = input("Enter save path (optional): ")
                
                # Validate chart type
                if chart_type not in ["bar", "line", "pie", "scatter"]:
                    print("Invalid chart type. Using 'bar'.")
                    chart_type = "bar"
                
                # Generate visualization
                print(f"\nGenerating {chart_type} chart visualization")
                response = agent.generate_visualization_from_data(
                    data=data,
                    chart_type=chart_type,
                    title=title or None,
                    save_path=save_path or None,
                    namespace=namespace,
                    session_id=session_id
                )
                
                # Print results
                print(f"\nGenerated visualization:")
                if response['images']:
                    print(f"URL: {response['images'][0]['image_url']}")
                    if response['images'][0]['local_path']:
                        print(f"Saved to: {response['images'][0]['local_path']}")
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            elif choice == "6":  # Illustrate a concept
                concept = input("Enter concept to illustrate: ")
                style = input("Enter style (optional): ")
                reference_text = input("Enter reference text (optional): ")
                save_path = input("Enter save path (optional): ")
                
                # Generate concept illustration
                print(f"\nGenerating illustration for concept: {concept}")
                response = agent.generate_concept_illustration(
                    concept=concept,
                    style=style or None,
                    reference_text=reference_text or None,
                    save_path=save_path or None,
                    namespace=namespace,
                    session_id=session_id
                )
                
                # Print results
                print(f"\nGenerated concept illustration:")
                if response['images']:
                    print(f"URL: {response['images'][0]['image_url']}")
                    if response['images'][0]['local_path']:
                        print(f"Saved to: {response['images'][0]['local_path']}")
                
                # Print stats
                print(f"\nProcessing took {response['execution_stats']['duration_seconds']} seconds")
                if response["errors"]:
                    print(f"Encountered {len(response['errors'])} errors:")
                    for error in response["errors"]:
                        print(f"- {error['type']}: {error['error']}")
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    except Exception as e:
        print(f"Error in interactive mode: {e}")
    
    finally:
        # Clean up
        agent.cleanup()


if __name__ == "__main__" and len(sys.argv) == 1:
    # If no arguments provided, run in interactive mode
    run_interactive_mode()
