"""
Research Example

This example demonstrates how to use the research workflow to perform
automated research on a topic.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apps.workflows.research_workflow import ResearchWorkflow, ResearchRequest

def main():
    """Run a complete research workflow example."""
    print("Research Workflow Example")
    print("========================\n")
    
    # Set up workspace directory
    workspace_dir = os.path.join(os.getcwd(), "example_workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Create the workflow
    workflow = ResearchWorkflow(workspace_dir=workspace_dir)
    
    # Example research topics
    topics = [
        "The impact of artificial intelligence on healthcare",
        "Sustainable energy solutions for developing countries",
        "The future of remote work after the pandemic"
    ]
    
    # Let user choose a topic
    print("Available research topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    print(f"{len(topics) + 1}. Custom topic")
    
    choice = int(input("\nSelect a topic (1-4): "))
    
    if choice <= len(topics):
        query = topics[choice - 1]
    else:
        query = input("Enter your research topic: ")
    
    # Prepare the request
    request: ResearchRequest = {
        "query": query,
        "session_id": f"example-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "parameters": {
            "max_sources": 3,
            "include_images": True,
            "verify_facts": True,
            "save_results": True,
            "research_depth": "standard",
            "output_format": "markdown",
            "workspace_dir": workspace_dir
        }
    }
    
    print(f"\nStarting research on: {query}")
    print("This may take a few minutes...\n")
    
    try:
        # Execute the workflow
        response = workflow.execute(request)
        
        # Print the results
        if response["success"]:
            print(f"Research completed in {response['execution_stats']['duration_seconds']} seconds")
            
            # Print summary
            print("\n=== SUMMARY ===\n")
            print(response["summary"][:500] + "..." if len(response["summary"]) > 500 else response["summary"])
            
            # Print sources
            print("\n=== SOURCES ===\n")
            for i, source in enumerate(response["sources"], 1):
                print(f"{i}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
            
            # Print file locations
            print("\n=== FILES ===\n")
            for file in response["files"]:
                print(f"- {file.get('path', 'Unknown path')}")
            
            print(f"\nFull results saved to: {response['files'][-1]['path'] if response['files'] else 'N/A'}")
            
            # Open the final output file
            if response["files"] and os.path.exists(response["files"][-1]["path"]):
                open_file = input("\nOpen the final output file? (y/n): ")
                if open_file.lower() == "y":
                    # Cross-platform file opening
                    if sys.platform == "win32":
                        os.startfile(response["files"][-1]["path"])
                    elif sys.platform == "darwin":
                        os.system(f"open {response['files'][-1]['path']}")
                    else:
                        os.system(f"xdg-open {response['files'][-1]['path']}")
        else:
            print(f"Research failed: {response['errors'][0]['error'] if response['errors'] else 'Unknown error'}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up
        workflow.cleanup()
        print("\nExample completed.")

if __name__ == "__main__":
    main()
