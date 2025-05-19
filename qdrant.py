#to see qdrant data
from qdrant_client import QdrantClient
import portalocker.exceptions
from collections import defaultdict

def view_interactions():
    client = None
    try:
        # Initialize QdrantClient with local storage
        client = QdrantClient(path="data/qdrant_db")
        
        # Get collections
        collections = client.get_collections().collections
        print("Collections:", [c.name for c in collections])

        # Scroll through all interactions
        result, _ = client.scroll(collection_name="interactions", limit=1000)  # Adjust limit as needed
        
        # Group interactions by user
        user_interactions = defaultdict(list)
        for point in result:
            user_id = point.payload.get("user_id", "Unknown")  # Adjust key based on your payload structure
            user_interactions[user_id].append(point)
        
        # Display interactions for each user
        for user_id, points in user_interactions.items():
            print(f"\nInteractions for User: {user_id}")
            print("=" * 50)
            for point in points:
                print(f"ID: {point.id}")
                
                if point.vector is not None:
                    print("Vector length:", len(point.vector))
                else:
                    print("No vector data found for this point.")
                
                print("Payload:", point.payload)
                print("-" * 40)
            
    except portalocker.exceptions.AlreadyLocked:
        print("Error: The Qdrant storage folder 'data/qdrant_db' is already in use by another instance.")
        print("Please ensure no other QdrantClient is accessing this folder.")
        print("Alternatively, consider using a Qdrant server for concurrent access.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # Ensure client connection is closed
        if client is not None:
            client.close()

if __name__ == "__main__":
    view_interactions()