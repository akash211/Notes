# NOTES FOR AZURE Machine Learning with DP100 certification

Most of the notes are taken from [Microsoft learn](https://learn.microsoft.com/en-us/training/courses/dp-100t01).
Except section 7 which is taken from [Microsoft learn challenge](https://learn.microsoft.com/en-in/collections/67pkuoq3xp58?WT.mc_id=cloudskillschallenge_764de7ca-8b6d-4b3a-a491-1942af389d8c).

## Section 1. Design a machine learning solution

### Chapter 1. Design a data ingestion strategy for machine learning projects

To train a machine learning model there are 6 steps and it is iterative process:

1. Define the problem: Decide on what the model should predict and when it's successful.
2. Get the data: Find data sources and get access.
3. Prepare the data: Explore the data. Clean and transform the data based on the model's requirements.
4. Train the model: Choose an algorithm and hyperparameter values based on trial and error.
5. Integrate the model: Deploy the model to an endpoint to generate predictions.
6. Monitor the model: Track the model's performance.

To use data in Azure ML (or Azure Databricks or Azure Synapse Analytics) most of the time below 3 services are used:

1. Azure Blob Storage
2. Azure Data Lake Storage (Basically Blob storage with hierarchical namespace, helps in giving access properly)
3. Azure SQL

This image shows all types of services used on Azure for storing data.
[Saving data on Azure] (<https://learn.microsoft.com/en-us/azure/architecture/guide/technology-choices/images/data-store-decision-tree.svg>)

For creating data ingestion pipelines usually Azure Databricks and Azure Synapse Analytics are used. Azure ML is also an option to create data ingestion pipeline but they do not scale well.

### Chapter 2. Design a machine learning model training solution

Some common Machine Learning tasks are:

- Classification
- Regression
- Clustering
- Time-series Forecasting
- Computer Vision
- NLP

In Azure there are sevaral services which is used to train machine learning models.

1. Azure Machine Learning
2. Azure Synapse Analytics
3. Azure Databricks
4. Azure AI Services
5. Azure Cognitive Services

Which service is to be used for what kind of project depends on the requirements. Some general guidelines are below:

- Use Azure AI Services for pre-built models that fit your requirements to save time and effort.
- Use Azure Synapse Analytics or Azure Databricks if you need a unified experience for data-related tasks (data engineering, data science).
- Use distributed compute in Azure Synapse Analytics or Azure Databricks when working with large datasets to keep the process efficient. You'll work with PySpark.
- Use Azure Machine Learning or Azure Databricks for full control over the machine learning lifecycle.
- Choose Azure Machine Learning if Python is your preferred programming language.
- Choose Azure Machine Learning for an intuitive user interface to manage your machine learning lifecycle.

To find which virtual machine/compute best fits again there is general guidelines:

1. CPU/GPU - GPU is costly than CPU. CPU is preferred with small to medium size tabular data while GPU is preferred with large datasets or unstructured datasets.
2. VM sizes - It can be a. general puspose where CPU to memory ratio is balanced, usually good for testing and development or b. memory optimized where memory is high.
3. Spark - Synapse and databricks both have Spark compute option. Spark compute/cluster can have same configuration but it distribute the workloads. Spark cluster has driver node and worker nodes. Initially code written in Scala, SQL, RSpark or PySpark  communicatw with the driver node which distribute the work between worker nodes and worker nodes execute workloads which reduces processing time. Finally work is summarised and driver node sends back the final result. If we use Python here, then only driver node will be used and worker nodes will be unused. During creation of SPark cluster we can select GPU or CPU and also size of the cluster.

### Chapter 3. Design a model deployment solution

We can deploy a model to an endpoint to generate predictions for either real-time or batch.  
An ednpoint is a web address that the app can call to get a message back.  
For real-time predictions, compute should be always available. For this Azure Container Instance (ACI) or Azure Kubernetes Service (AKS) is used.  
For batch predictions we can use computer cluster with multiple nodes. This is triggered when required and after use it is scaled down to 0 nodes.

### Chapter 4. Design a machine learning operations solution

MLOps help is scaling a model from proof of concept or pilot project to production. This is similar to DevOps. Here monitoring and Retraining is very important.  
Retraining is done either based on schedule or based on some metrics.  
MLOps automation is usually done using Azure Devops and Github Actions. For that scripts works best compared to notebooks. Also Azure CLI works better than Azure ML Python SDK.

## Section 2. Explore and configure the Azure Machine Learning workspace

### Chapter 5. Explore Azure Machine Learning workspace resources and assets

Azure ML workspace can be created by:

- Using Azure CLI with Azure ML CLI Extension
- Using Azure Portal
- Using Azure Resource Manager (ARM) template
- Using Azure ML Python SDK

To create using python SDK, below is the code:

```python
from azure.ai.ml.entities import Workspace

workspace_name = "mlw-example"

ws_basic = Workspace(
    name=workspace_name,
    location="eastus",
    display_name="Basic workspace-example",
    description="This example shows how to create a basic workspace",
)
ml_client.workspaces.begin_create(ws_basic)
```

When Azure ML serice is created some Azure resources are automatically created:

- A storage account
- A key vault
- A container registry (ACR) - which stores images for ML environments
- Application Insights - for metrics and logs to monitor services

To give access to workspace, `Access Control(IAM)` tab is used. Here RBAC(role-based access control) is created. Here we are assigning a user to the workspace. For users 3 roles are there Owner, Contributor and Reader. Contributor has all the access to resources except he can not grant access to others.  
There are some built-in roles as well (And custom roles can be created):

- AzureML Data Scientist
- AzureML Computer Operator

Loosely There are 3 types of resources in Azure ML workspace:

- `Workspace`: Top level resource where traning tracking, deploying happens. This will have logs, metrics, outputs, models, snapshots of code etc.
- `Compute` : Usually handled by administrator because they cost a lot and used for training or deploying a model. There are 5 types of compute:
  - a. `Compute instances` : Like VM, usually used for development in Jupyter notebooks
  - b. `Compute clusters` : On demnd cluster of CPU/GPU nodes. Used for production
  - c. `Kubernetes clusters` : Can create/attach to AKS cluster, mostly used for production
  - d. `Attached computers` : To attach other compute resources like Azure Databricks or Synapse Spark pools
  - e. `Serverless compute` : Fully managed on demand compute, used usually for training jobs
- `Datastore` : Workstore does not store any data but references to different data services. Connection information to these data services are stored in Azure Key Vault. When a workspace is created an Azure storage account is created and automatically connected to the workspace and 4 datastores are already added to the workspace:
  - `workspaceartifactstore`: stores compute and experiment logs when running jobs
  - `workspaceworkingdirectory` : connects to file share of azure storage and used by Notebooks.
  - `workspaceblobstore` : default datastore. connects to blob storage.
  - `workspacefilestore` : connects to file share.

We can also create datastores to connect to other Azure data services like Azure Data Lake Storage(Gen2).

Data scientist usually work with variaous assets which are listed below:

- `Models`: This is end product of training a model. These are stored as pickle format or MLModel format. Python code is used ot convert to pickle while MLflow is used to store in MLModel format. Models are saves with name and version information.
- `Environments`: Environment also have name and version and they specify software packages, environment variables, software settings to run scripts. It is saved as image in the Azure Container registry. When we run a script we specify environment and then compute uses this info to install everything and change settings accordingly which will make code reusable cross compute targets.
- `Data`: Data assests are specific file or folder. Using them we can access data every time without authentication eevry single time. He we specify the path to point to the file or folder and the name and version.
- `Components`: There are reusable parts of code that can be used in multiple projects. We can write components and specify name, version, code and environment needed to run the code. Usually used in pipelines.

To train model we can use:

- `Automated Machine Learning`: It can iterate through multiple algorithms paired with feature selections to find the best performing model for the data automatically.
- `Jupyter notebooks`: These are saved in dile share of Azure storage. They use compute instance and we can edit and run them in Notebook page of the ML studio. We can even work with them in VS code.
- `Scripts as job`: For production ready jobs, scripts are mostly used. There are many types of jobs like Command(for single script), Sweep(for single scripts with hyperparameters tuning) and Pipleine(For running pipeline with multiple scripts and components).

### Chapter 6. Explore developer tools for workspace interaction

Usually ML Studio is ideal for quick experimentation or exploring past jobs. But for repetitive work or to automate tasks Azure CL or Python SDK are more ideal. Any time a script or pipeline is run, it will create a job in Azure Machine Learning.
`1. Python SDK` :

```python
# to install the SDK
pip install azure-ai-ml
# We need subscription_id, resource_group, workspace_name to connect to the workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)
# to create job
from azure.ai.ml. import command
job = command(code="./src", command="python script.py", environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1", compute="cpu-cluster", experiment_name="my-experiment")
# to connect to and submit job
returned_job = ml_client.jobs.create_or_update(job)
```

`2. Azure CLI` :

```bash
# to install ML extension ml
az extension add -n ml -y
# to get help on extension
az ml -h
# to create resource group and workspace
az group create --name "rg-dp100-labs" --location "eastus"
az ml workspace create --name "mlw-dp100-labs" -g "rg-dp100-labs"
# to create a compute target
az ml compute create --name aml-cluster --size STANDARD_DS3_v2 --min-instances 0 --max-instances 5 --type AmlCompute --resource-group my-resource-group --workspace-name my-workspace
```

We can also use YAML format to define all the configuration of the compute target like this

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json 
name: aml-cluster
type: amlcompute
size: STANDARD_DS3_v2
min_instances: 0
max_instances: 5
```

Then to create resource we can use:

```bash
az ml compute create --file compute.yaml
```


### Chapter 7. Make data available in Azure Machine Learning

### Chapter 8. Work with compute targets in Azure Machine Learning

### Chapter 9. Work with environments in Azure Machine Learning

## Section 3. Experiment with Azure Machine Learning

### Chapter 10. Find the best classification model with Automated Machine Learning

### Chapter 11. Track model training in Jupyter notebooks with MLflow

## Section 4. Optimize model training with Azure Machine Learning

### Chapter 12. Run a training script as a command job in Azure Machine Learning

### Chapter 13. Track model training with MLflow in jobs

### Chapter 14. Perform hyperparameter tuning with Azure Machine Learning

### Chapter 15. Run pipelines in Azure Machine Learning

## Section 5. Manage and review models in Azure Machine Learning

### Chapter 16. Register an MLflow model in Azure Machine Learning

### Chapter 17. Create and explore the Responsible AI dashboard for a model in Azure Machine Learning

## Section 6. Deploy and consume models with Azure Machine Learning

### Chapter 18. Deploy a model to a managed online endpoint

### Chapter 19. Deploy a model to a batch endpoint

## Section 7. Other topics from challanges

### Chapter 20. Understand the Transformer architecture and explore large language models in Azure Machine Learning

### Chapter 21. Fine-tune a foundation model with Azure Machine Learning

### Chapter 22. Get started with prompt flow to develop Large Language Model (LLM) apps

### Chapter 23. Train a model and debug it with Responsible AI dashboard
