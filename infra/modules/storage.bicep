@description('Location for the storage account')
param location string

@description('Tags for the resource')
param tags object

@description('Environment name for uniqueness')
param environmentName string

var accountName = 'st${replace(environmentName, '-', '')}${uniqueString(resourceGroup().id)}'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take(accountName, 24)
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource resultsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'agent-results'
  properties: {
    publicAccess: 'None'
  }
}

output accountName string = storage.name
output containerName string = resultsContainer.name
