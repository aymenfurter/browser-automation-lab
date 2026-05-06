@description('Location for the OpenAI resource')
param location string

@description('Tags for the resource')
param tags object

@description('Principal ID to grant Cognitive Services User role')
param principalId string = ''

var accountName = 'oai-${uniqueString(resourceGroup().id)}'

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: accountName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: accountName
    publicNetworkAccess: 'Enabled'
  }
}

resource gpt4_1 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'gpt-4.1'
  sku: {
    name: 'GlobalStandard'
    capacity: 30
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1'
      version: '2025-04-14'
    }
  }
}

// Grant Cognitive Services User role to the specified principal
resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(openai.id, principalId, 'a97b65f3-24c7-4388-baec-2e87135dc908')
  scope: openai
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')
    principalType: 'User'
  }
}

output endpoint string = openai.properties.endpoint
output deploymentName string = gpt4_1.name
output accountName string = openai.name
