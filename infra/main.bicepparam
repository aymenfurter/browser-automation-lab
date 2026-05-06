using './main.bicep'

param environmentName = readEnvironmentVariable('AZURE_ENV_NAME', 'browser-auto-lab')
param location = readEnvironmentVariable('AZURE_LOCATION', 'swedencentral')
param principalId = readEnvironmentVariable('AZURE_PRINCIPAL_ID', '')
