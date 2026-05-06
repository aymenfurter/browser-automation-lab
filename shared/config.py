"""Shared configuration for both agents."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

SLIDEFINDER_URL = os.getenv("SLIDEFINDER_URL", "https://example.com/")

SEARCH_QUERIES = [
    "Azure Kubernetes Service",
    "Azure DevOps Pipelines",
    "Azure Bicep Infrastructure as Code",
    "Azure Container Apps",
    "Azure Functions Serverless",
    "Azure API Management",
    "Azure Service Bus Messaging",
    "Azure Monitor Observability",
    "Azure Key Vault Secrets",
    "Azure Front Door CDN",
    "Azure Cosmos DB",
    "Azure Virtual Network Security",
    "Azure AI Search",
    "Azure Logic Apps Integration",
    "Azure Event Grid",
    "Azure Static Web Apps",
    "Azure Machine Learning",
    "Azure Defender for Cloud",
    "Azure Load Testing",
    "GitHub Copilot for Azure",
]

HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"

# Azure OpenAI configuration (deployed via infra/main.bicep)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
