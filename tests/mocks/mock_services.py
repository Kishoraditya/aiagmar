"""
Mock service implementations for testing purposes.
"""

import json
import os
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from unittest.mock import MagicMock, patch

# Import MCP mocks for reference
from tests.mocks.mock_mcps import (
    MockBraveSearchMCP,
    MockEverArtMCP,
    MockFetchMCP,
    MockFilesystemMCP,
    MockMemoryMCP
)


class MockLLMService:
    """Mock implementation of a Language Model service for testing."""
    
    def __init__(self, model_name: str = "mock-model"):
        """
        Initialize the mock LLM service.
        
        Args:
            model_name: Name of the model to simulate
        """
        self.model_name = model_name
        
        # Track method calls
        self.generate_text_called = False
        self.generate_text_args = None
        self.generate_text_kwargs = None
        self.generate_text_result = "This is a mock response from the language model."
        
        self.chat_called = False
        self.chat_args = None
        self.chat_kwargs = None
        self.chat_result = "This is a mock chat response from the language model."
        
        # Configurable response generators
        self.text_generator = None
        self.chat_generator = None
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text using the language model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        self.generate_text_called = True
        self.generate_text_args = (prompt,)
        self.generate_text_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Use custom generator if provided
        if self.text_generator and callable(self.text_generator):
            return self.text_generator(prompt, max_tokens, temperature, **kwargs)
        
        return self.generate_text_result
    
    def set_generate_text_result(self, result: str):
        """Set the result to be returned by generate_text()."""
        self.generate_text_result = result
    
    def set_text_generator(self, generator: Callable):
        """
        Set a custom function to generate text responses.
        
        Args:
            generator: Function that takes the same arguments as generate_text and returns a string
        """
        self.text_generator = generator
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 100,
            temperature: float = 0.7, **kwargs) -> str:
        """
        Generate a chat response using the language model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            Generated chat response
        """
        self.chat_called = True
        self.chat_args = (messages,)
        self.chat_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Use custom generator if provided
        if self.chat_generator and callable(self.chat_generator):
            return self.chat_generator(messages, max_tokens, temperature, **kwargs)
        
        return self.chat_result
    
    def set_chat_result(self, result: str):
        """Set the result to be returned by chat()."""
        self.chat_result = result
    
    def set_chat_generator(self, generator: Callable):
        """
        Set a custom function to generate chat responses.
        
        Args:
            generator: Function that takes the same arguments as chat and returns a string
        """
        self.chat_generator = generator


class MockAPIService:
    """Mock implementation of a generic API service for testing."""
    
    def __init__(self, base_url: str = "https://api.example.com", api_key: Optional[str] = "mock-api-key"):
        """
        Initialize the mock API service.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # Track method calls
        self.get_called = False
        self.get_args = None
        self.get_kwargs = None
        self.get_result = {"status": "success", "data": {"message": "Mock GET response"}}
        
        self.post_called = False
        self.post_args = None
        self.post_kwargs = None
        self.post_result = {"status": "success", "data": {"message": "Mock POST response", "id": "12345"}}
        
        self.put_called = False
        self.put_args = None
        self.put_kwargs = None
        self.put_result = {"status": "success", "data": {"message": "Mock PUT response"}}
        
        self.delete_called = False
        self.delete_args = None
        self.delete_kwargs = None
        self.delete_result = {"status": "success", "data": {"message": "Mock DELETE response"}}
        
        # Mock response handlers
        self.response_handlers = {}
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response data
        """
        self.get_called = True
        self.get_args = (endpoint,)
        self.get_kwargs = {
            "params": params or {},
            "headers": headers or {}
        }
        
        # Check for custom handler
        key = f"GET:{endpoint}"
        if key in self.response_handlers and callable(self.response_handlers[key]):
            return self.response_handlers[key](endpoint, params, headers)
        
        return self.get_result
    
    def set_get_result(self, result: Dict[str, Any]):
        """Set the result to be returned by get()."""
        self.get_result = result
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            headers: Request headers
            
        Returns:
            Response data
        """
        self.post_called = True
        self.post_args = (endpoint,)
        self.post_kwargs = {
            "data": data or {},
            "headers": headers or {}
        }
        
        # Check for custom handler
        key = f"POST:{endpoint}"
        if key in self.response_handlers and callable(self.response_handlers[key]):
            return self.response_handlers[key](endpoint, data, headers)
        
        return self.post_result
    
    def set_post_result(self, result: Dict[str, Any]):
        """Set the result to be returned by post()."""
        self.post_result = result
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            headers: Request headers
            
        Returns:
            Response data
        """
        self.put_called = True
        self.put_args = (endpoint,)
        self.put_kwargs = {
            "data": data or {},
            "headers": headers or {}
        }
        
        # Check for custom handler
        key = f"PUT:{endpoint}"
        if key in self.response_handlers and callable(self.response_handlers[key]):
            return self.response_handlers[key](endpoint, data, headers)
        
        return self.put_result
    
    def set_put_result(self, result: Dict[str, Any]):
        """Set the result to be returned by put()."""
        self.put_result = result
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform a DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response data
        """
        self.delete_called = True
        self.delete_args = (endpoint,)
        self.delete_kwargs = {
            "params": params or {},
            "headers": headers or {}
        }
        
        # Check for custom handler
        key = f"DELETE:{endpoint}"
        if key in self.response_handlers and callable(self.response_handlers[key]):
            return self.response_handlers[key](endpoint, params, headers)
        
        return self.delete_result
    
    def set_delete_result(self, result: Dict[str, Any]):
        """Set the result to be returned by delete()."""
        self.delete_result = result
    
    def register_handler(self, method: str, endpoint: str, handler: Callable):
        """
        Register a custom response handler for a specific method and endpoint.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            handler: Function that takes the same arguments as the method and returns a response
        """
        key = f"{method.upper()}:{endpoint}"
        self.response_handlers[key] = handler


class MockDatabaseService:
    """Mock implementation of a database service for testing."""
    
    def __init__(self, db_name: str = "mock_db"):
        """
        Initialize the mock database service.
        
        Args:
            db_name: Name of the database
        """
        self.db_name = db_name
        
        # Mock database storage
        self.collections = {}
        
        # Track method calls
        self.insert_called = False
        self.insert_args = None
        self.insert_kwargs = None
        self.insert_result = {"id": "mock_id_12345"}
        
        self.find_called = False
        self.find_args = None
        self.find_kwargs = None
        self.find_result = []
        
        self.update_called = False
        self.update_args = None
        self.update_kwargs = None
        self.update_result = {"modified_count": 1}
        
        self.delete_called = False
        self.delete_args = None
        self.delete_kwargs = None
        self.delete_result = {"deleted_count": 1}
    
    def insert(self, collection: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a document into a collection.
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            Result with document ID
        """
        self.insert_called = True
        self.insert_args = (collection, document)
        self.insert_kwargs = {}
        
        # Update mock database
        if collection not in self.collections:
            self.collections[collection] = []
        
        # Add ID if not present
        if "_id" not in document:
            document["_id"] = str(uuid.uuid4())
        
        self.collections[collection].append(document.copy())
        
        return {"id": document["_id"]}
    
    def set_insert_result(self, result: Dict[str, Any]):
        """Set the result to be returned by insert()."""
        self.insert_result = result
    
    def find(self, collection: str, query: Dict[str, Any] = None, 
            sort: Optional[List[tuple]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find documents in a collection.
        
        Args:
            collection: Collection name
            query: Query filter
            sort: Sort specification
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        self.find_called = True
        self.find_args = (collection,)
        self.find_kwargs = {
            "query": query or {},
            "sort": sort,
            "limit": limit
        }
        
        # Query mock database
        if collection not in self.collections:
            return []
        
        # Simple filtering (exact matches only)
        results = self.collections[collection]
        if query:
            results = [doc for doc in results if all(
                key in doc and doc[key] == value 
                for key, value in query.items()
            )]
        
        # Apply limit
        if limit is not None and limit > 0:
            results = results[:limit]
        
        return results
    
    def set_find_result(self, result: List[Dict[str, Any]]):
        """Set the result to be returned by find()."""
        self.find_result = result
    
    def update(self, collection: str, query: Dict[str, Any], 
              update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update documents in a collection.
        
        Args:
            collection: Collection name
            query: Query filter
            update: Update operations
            
        Returns:
            Result with modified count
        """
        self.update_called = True
        self.update_args = (collection, query, update)
        self.update_kwargs = {}
        
        # Update mock database
        if collection not in self.collections:
            return {"modified_count": 0}
        
        modified_count = 0
        for doc in self.collections[collection]:
            if all(key in doc and doc[key] == value for key, value in query.items()):
                # Simple update (replace fields)
                for key, value in update.items():
                    doc[key] = value
                modified_count += 1
        
        return {"modified_count": modified_count}
    
    def set_update_result(self, result: Dict[str, Any]):
        """Set the result to be returned by update()."""
        self.update_result = result
    
    def delete(self, collection: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete documents from a collection.
        
        Args:
            collection: Collection name
            query: Query filter
            
        Returns:
            Result with deleted count
        """
        self.delete_called = True
        self.delete_args = (collection, query)
        self.delete_kwargs = {}
        
        # Delete from mock database
        if collection not in self.collections:
            return {"deleted_count": 0}
        
        original_count = len(self.collections[collection])
        self.collections[collection] = [
            doc for doc in self.collections[collection]
            if not all(key in doc and doc[key] == value for key, value in query.items())
        ]
        deleted_count = original_count - len(self.collections[collection])
        
        return {"deleted_count": deleted_count}
    def set_delete_result(self, result: Dict[str, Any]):
        """Set the result to be returned by delete()."""
        self.delete_result = result
    
    def drop_collection(self, collection: str) -> bool:
        """
        Drop a collection from the database.
        
        Args:
            collection: Collection name
            
        Returns:
            True if successful, False otherwise
        """
        if collection in self.collections:
            del self.collections[collection]
            return True
        return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        return list(self.collections.keys())


class MockStorageService:
    """Mock implementation of a file storage service for testing."""
    
    def __init__(self, base_path: str = "/mock/storage"):
        """
        Initialize the mock storage service.
        
        Args:
            base_path: Base storage path
        """
        self.base_path = base_path
        
        # Mock storage
        self.files = {}
        
        # Track method calls
        self.upload_file_called = False
        self.upload_file_args = None
        self.upload_file_kwargs = None
        self.upload_file_result = {"url": "https://storage.example.com/mock-file.txt"}
        
        self.download_file_called = False
        self.download_file_args = None
        self.download_file_kwargs = None
        self.download_file_result = b"Mock file content"
        
        self.delete_file_called = False
        self.delete_file_args = None
        self.delete_file_kwargs = None
        self.delete_file_result = True
        
        self.list_files_called = False
        self.list_files_args = None
        self.list_files_kwargs = None
        self.list_files_result = ["file1.txt", "file2.jpg", "file3.pdf"]
    
    def upload_file(self, file_path: str, content: Union[str, bytes], 
                   content_type: Optional[str] = None) -> Dict[str, str]:
        """
        Upload a file to storage.
        
        Args:
            file_path: Path where the file will be stored
            content: File content (string or bytes)
            content_type: MIME type of the file
            
        Returns:
            Dictionary with file URL
        """
        self.upload_file_called = True
        self.upload_file_args = (file_path, content)
        self.upload_file_kwargs = {"content_type": content_type}
        
        # Store in mock storage
        self.files[file_path] = {
            "content": content if isinstance(content, bytes) else content.encode('utf-8'),
            "content_type": content_type or "application/octet-stream"
        }
        
        # Generate mock URL
        url = f"https://storage.example.com/{os.path.basename(file_path)}"
        
        return {"url": url}
    
    def set_upload_file_result(self, result: Dict[str, str]):
        """Set the result to be returned by upload_file()."""
        self.upload_file_result = result
    
    def download_file(self, file_path: str) -> bytes:
        """
        Download a file from storage.
        
        Args:
            file_path: Path of the file to download
            
        Returns:
            File content as bytes
        """
        self.download_file_called = True
        self.download_file_args = (file_path,)
        self.download_file_kwargs = {}
        
        # Retrieve from mock storage
        if file_path in self.files:
            return self.files[file_path]["content"]
        
        raise FileNotFoundError(f"File not found: {file_path}")
    
    def set_download_file_result(self, result: bytes):
        """Set the result to be returned by download_file()."""
        self.download_file_result = result
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        self.delete_file_called = True
        self.delete_file_args = (file_path,)
        self.delete_file_kwargs = {}
        
        # Delete from mock storage
        if file_path in self.files:
            del self.files[file_path]
            return True
        
        return False
    
    def set_delete_file_result(self, result: bool):
        """Set the result to be returned by delete_file()."""
        self.delete_file_result = result
    
    def list_files(self, directory: str = "") -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            List of file paths
        """
        self.list_files_called = True
        self.list_files_args = (directory,)
        self.list_files_kwargs = {}
        
        # List from mock storage
        prefix = directory
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        if not prefix:
            return list(self.files.keys())
        
        return [path for path in self.files.keys() if path.startswith(prefix)]
    
    def set_list_files_result(self, result: List[str]):
        """Set the result to be returned by list_files()."""
        self.list_files_result = result
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path of the file to check
            
        Returns:
            True if the file exists, False otherwise
        """
        return file_path in self.files
    
    def get_file_url(self, file_path: str) -> str:
        """
        Get the URL for a file.
        
        Args:
            file_path: Path of the file
            
        Returns:
            File URL
        """
        if file_path in self.files:
            return f"https://storage.example.com/{os.path.basename(file_path)}"
        
        raise FileNotFoundError(f"File not found: {file_path}")


class MockEmailService:
    """Mock implementation of an email service for testing."""
    
    def __init__(self):
        """Initialize the mock email service."""
        # Track sent emails
        self.sent_emails = []
        
        # Track method calls
        self.send_email_called = False
        self.send_email_args = None
        self.send_email_kwargs = None
        self.send_email_result = {"status": "sent", "message_id": "mock-message-id"}
    
    def send_email(self, to: Union[str, List[str]], subject: str, body: str,
                  from_email: Optional[str] = None, cc: Optional[Union[str, List[str]]] = None,
                  bcc: Optional[Union[str, List[str]]] = None, 
                  attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        Send an email.
        
        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body
            from_email: Sender email address
            cc: Carbon copy recipient(s)
            bcc: Blind carbon copy recipient(s)
            attachments: List of attachment dictionaries
            
        Returns:
            Dictionary with status and message ID
        """
        self.send_email_called = True
        self.send_email_args = (to, subject, body)
        self.send_email_kwargs = {
            "from_email": from_email,
            "cc": cc,
            "bcc": bcc,
            "attachments": attachments
        }
        
        # Normalize recipients to lists
        to_list = to if isinstance(to, list) else [to]
        cc_list = cc if isinstance(cc, list) else ([cc] if cc else [])
        bcc_list = bcc if isinstance(bcc, list) else ([bcc] if bcc else [])
        
        # Store sent email
        email = {
            "to": to_list,
            "subject": subject,
            "body": body,
            "from": from_email,
            "cc": cc_list,
            "bcc": bcc_list,
            "attachments": attachments or [],
            "timestamp": uuid.uuid4().hex,
            "message_id": f"mock-message-{uuid.uuid4().hex[:8]}"
        }
        
        self.sent_emails.append(email)
        
        return {"status": "sent", "message_id": email["message_id"]}
    
    def set_send_email_result(self, result: Dict[str, str]):
        """Set the result to be returned by send_email()."""
        self.send_email_result = result
    
    def get_sent_emails(self, to: Optional[str] = None, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get sent emails, optionally filtered by recipient or subject.
        
        Args:
            to: Filter by recipient
            subject: Filter by subject
            
        Returns:
            List of matching emails
        """
        results = self.sent_emails
        
        if to:
            results = [email for email in results if to in email["to"]]
        
        if subject:
            results = [email for email in results if subject in email["subject"]]
        
        return results
    
    def clear_sent_emails(self):
        """Clear the list of sent emails."""
        self.sent_emails = []


class MockWebhookService:
    """Mock implementation of a webhook service for testing."""
    
    def __init__(self):
        """Initialize the mock webhook service."""
        # Track received webhooks
        self.received_webhooks = []
        
        # Track sent webhooks
        self.sent_webhooks = []
        
        # Track method calls
        self.send_webhook_called = False
        self.send_webhook_args = None
        self.send_webhook_kwargs = None
        self.send_webhook_result = {"status": "sent", "id": "mock-webhook-id"}
        
        self.register_webhook_called = False
        self.register_webhook_args = None
        self.register_webhook_kwargs = None
        self.register_webhook_result = {"status": "registered", "id": "mock-registration-id"}
    
    def send_webhook(self, url: str, payload: Dict[str, Any], 
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Send a webhook to a URL.
        
        Args:
            url: Webhook URL
            payload: Webhook payload
            headers: Request headers
            
        Returns:
            Dictionary with status and webhook ID
        """
        self.send_webhook_called = True
        self.send_webhook_args = (url, payload)
        self.send_webhook_kwargs = {"headers": headers}
        
        # Store sent webhook
        webhook = {
            "url": url,
            "payload": payload,
            "headers": headers or {},
            "timestamp": uuid.uuid4().hex,
            "id": f"mock-webhook-{uuid.uuid4().hex[:8]}"
        }
        
        self.sent_webhooks.append(webhook)
        
        return {"status": "sent", "id": webhook["id"]}
    
    def set_send_webhook_result(self, result: Dict[str, str]):
        """Set the result to be returned by send_webhook()."""
        self.send_webhook_result = result
    
    def register_webhook(self, event_type: str, url: str, 
                        secret: Optional[str] = None) -> Dict[str, str]:
        """
        Register a webhook for an event type.
        
        Args:
            event_type: Type of event to listen for
            url: Webhook URL
            secret: Webhook secret for signature verification
            
        Returns:
            Dictionary with status and registration ID
        """
        self.register_webhook_called = True
        self.register_webhook_args = (event_type, url)
        self.register_webhook_kwargs = {"secret": secret}
        
        registration_id = f"mock-registration-{uuid.uuid4().hex[:8]}"
        
        return {"status": "registered", "id": registration_id}
    
    def set_register_webhook_result(self, result: Dict[str, str]):
        """Set the result to be returned by register_webhook()."""
        self.register_webhook_result = result
    
    def simulate_webhook(self, event_type: str, payload: Dict[str, Any]):
        """
        Simulate receiving a webhook.
        
        Args:
            event_type: Type of event
            payload: Webhook payload
        """
        webhook = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": uuid.uuid4().hex,
            "id": f"mock-received-{uuid.uuid4().hex[:8]}"
        }
        
        self.received_webhooks.append(webhook)
    
    def get_received_webhooks(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get received webhooks, optionally filtered by event type.
        
        Args:
            event_type: Filter by event type
            
        Returns:
            List of matching webhooks
        """
        if event_type:
            return [webhook for webhook in self.received_webhooks if webhook["event_type"] == event_type]
        
        return self.received_webhooks
    
    def get_sent_webhooks(self, url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get sent webhooks, optionally filtered by URL.
        
        Args:
            url: Filter by URL
            
        Returns:
            List of matching webhooks
        """
        if url:
            return [webhook for webhook in self.sent_webhooks if webhook["url"] == url]
        
        return self.sent_webhooks
    
    def clear_webhooks(self):
        """Clear the lists of received and sent webhooks."""
        self.received_webhooks = []
        self.sent_webhooks = []


# Factory class for creating mock service instances
class MockServiceFactory:
    """Factory for creating mock service instances."""
    
    @staticmethod
    def create_llm_service(model_name: str = "mock-model") -> MockLLMService:
        """Create a mock LLM service instance."""
        return MockLLMService(model_name=model_name)
    
    @staticmethod
    def create_api_service(base_url: str = "https://api.example.com", 
                          api_key: Optional[str] = "mock-api-key") -> MockAPIService:
        """Create a mock API service instance."""
        return MockAPIService(base_url=base_url, api_key=api_key)
    
    @staticmethod
    def create_database_service(db_name: str = "mock_db") -> MockDatabaseService:
        """Create a mock database service instance."""
        return MockDatabaseService(db_name=db_name)
    
    @staticmethod
    def create_storage_service(base_path: str = "/mock/storage") -> MockStorageService:
        """Create a mock storage service instance."""
        return MockStorageService(base_path=base_path)
    @staticmethod
    def create_email_service() -> MockEmailService:
        """Create a mock email service instance."""
        return MockEmailService()
    
    @staticmethod
    def create_webhook_service() -> MockWebhookService:
        """Create a mock webhook service instance."""
        return MockWebhookService()


# Patch functions for testing
def patch_services():
    """Patch all service classes with their mock counterparts."""
    patches = [
        # Add patches for actual service classes when they are implemented
        # For example:
        # patch('apps.services.llm_service.LLMService', MockLLMService),
        # patch('apps.services.api_service.APIService', MockAPIService),
        # etc.
    ]
    
    for p in patches:
        p.start()
    
    return patches

def stop_patches(patches):
    """Stop all patches."""
    for p in patches:
        p.stop()


# If this module is run directly, perform a simple test
if __name__ == "__main__":
    # Create mock services
    llm = MockLLMService()
    api = MockAPIService()
    db = MockDatabaseService()
    storage = MockStorageService()
    email = MockEmailService()
    webhook = MockWebhookService()
    
    # Test LLM service
    result = llm.generate_text("Tell me about AI")
    print(f"LLM generate_text result: {result[:50]}...")
    
    # Test API service
    result = api.get("/users")
    print(f"API get result: {result}")
    
    # Test database service
    db.insert("users", {"name": "Test User", "email": "test@example.com"})
    result = db.find("users", {"name": "Test User"})
    print(f"Database find result: {result}")
    
    # Test storage service
    result = storage.upload_file("test.txt", "Hello, world!")
    print(f"Storage upload_file result: {result}")
    
    # Test email service
    result = email.send_email("test@example.com", "Test Subject", "Test Body")
    print(f"Email send_email result: {result}")
    
    # Test webhook service
    result = webhook.send_webhook("https://example.com/webhook", {"event": "test"})
    print(f"Webhook send_webhook result: {result}")
