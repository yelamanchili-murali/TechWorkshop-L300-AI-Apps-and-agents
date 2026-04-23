import os
import sys
import requests
from functools import lru_cache

from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

EMBEDDING_ENDPOINT = os.environ.get("embedding_endpoint")
EMBEDDING_DEPLOYMENT = os.environ.get("embedding_deployment")
EMBEDDING_API_VERSION = os.environ.get("embedding_api_version")

if not COSMOS_ENDPOINT:
    raise ValueError("COSMOS_ENDPOINT environment variable is not set")
if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME environment variable is not set")
if not CONTAINER_NAME:
    raise ValueError("CONTAINER_NAME environment variable is not set")


@lru_cache
def get_credential():
    return DefaultAzureCredential()


@lru_cache
def get_container():
    client = CosmosClient(COSMOS_ENDPOINT, credential=get_credential())
    database = client.get_database_client(DATABASE_NAME)
    return database.get_container_client(CONTAINER_NAME)


def test_cosmos_token():
    token = get_credential().get_token("https://cosmos.azure.com/.default")
    return {"expires_on": token.expires_on}


def get_request_embedding(text: str) -> list[float] | None:
    if not EMBEDDING_ENDPOINT or not EMBEDDING_DEPLOYMENT or not EMBEDDING_API_VERSION:
        raise ValueError(
            "Embedding endpoint configuration missing. "
            "Set EMBEDDING_ENDPOINT, EMBEDDING_DEPLOYMENT, EMBEDDING_API_VERSION"
        )

    url = (
        EMBEDDING_ENDPOINT.rstrip("/")
        + f"/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={EMBEDDING_API_VERSION}"
    )

    token = get_credential().get_token("https://cognitiveservices.azure.com/.default")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.token}",
    }
    payload = {"input": text}

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [{}])[0].get("embedding")


def product_recommendations(question: str, top_k: int = 8):
    query_vector = get_request_embedding(question)
    if query_vector is None:
        raise RuntimeError("Failed to generate query embedding")

    container = get_container()

    query = (
        "SELECT c.id, c.ProductID, c.ProductName, c.ProductCategory, c.ProductDescription, "
        "c.ImageURL, c.ProductPunchLine, c.Price "
        "FROM c "
        "ORDER BY VectorDistance(c.request_vector, @vector) "
        "OFFSET 0 LIMIT @top"
    )

    parameters = [
        {"name": "@vector", "value": query_vector},
        {"name": "@top", "value": top_k},
    ]

    items = list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
            max_item_count=top_k,
        )
    )

    return [
        {
            "id": item.get("ProductID"),
            "name": item.get("ProductName"),
            "type": item.get("ProductCategory"),
            "description": item.get("ProductDescription"),
            "imageURL": item.get("ImageURL"),
            "punchLine": item.get("ProductPunchLine"),
            "price": item.get("Price"),
        }
        for item in items
    ]