targetScope = 'subscription'

@description('The location for all resources')
param location string = 'swedencentral'

@description('A unique suffix for resource names')
param environmentName string

@description('The principal ID to grant Cognitive Services User role')
param principalId string = ''

var resourceGroupName = 'rg-${environmentName}'
var tags = {
  'azd-env-name': environmentName
  project: 'browser-automation-lab'
}

// Resource Group
resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

// Azure OpenAI (for LangGraph agent LLM calls)
module openai 'modules/openai.bicep' = {
  name: 'openai'
  scope: rg
  params: {
    location: location
    tags: tags
    principalId: principalId
  }
}

// Storage Account (for agent result output)
module storage 'modules/storage.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    location: location
    tags: tags
    environmentName: environmentName
  }
}

output AZURE_OPENAI_ENDPOINT string = openai.outputs.endpoint
output AZURE_OPENAI_DEPLOYMENT string = openai.outputs.deploymentName
output AZURE_OPENAI_API_VERSION string = '2024-12-01-preview'
output AZURE_STORAGE_ACCOUNT string = storage.outputs.accountName
output AZURE_RESOURCE_GROUP string = rg.name
