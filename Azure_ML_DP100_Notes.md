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

Uniform Resource Identifiers(URIs) are used to access the data in Azure ML. URIs references location of the data. There are 3 protocols used for connecting:

- `http(s)` : used for data in azure blob or to get public available http(s) location. http is also used for private container but then authentication is required.
- `abfs(s)` : used for data stored in Azure Data lake gen2
- `azureml` : for data stored in datastore, data is somewhere else but connection and authentication information is saved here so authentication is not required every time.

Datastore gives abstraction for cloud data resources. They provide easy to use URIs to the actual data, securely stores connection information without exposing keys or secrets. There are two method to authenticate a storage account to a datastore:

1. Credential based:
   - `storage_account_key` : uses account key based authentication
   - `storage_account_sas` : uses sas based authentication
2. Identity based: authentication usually using Microsoft Entra identity

There are 4 built in datastores where two connect with azure blob an two with azure fileshares. But usually we work with datastore of our own. Datstore can be created using GUI, SDK or CLI.

```python
# To create a datastore to connect to blob storage using account key:
blob_datastore = AzureBlobDatastore(
       name = "blob_example",
       description = "Datastore pointing to a blob container",
       account_name = "mytestblobstore",
       container_name = "data-container",
       credentials = AccountKeyCredentials(
           account_key="XXXxxxXXXxXXXXxxXXX"
       ),
)
ml_client.create_or_update(blob_datastore)

blob_datastore = AzureBlobDatastore(
    name="blob_sas_example",
    description="Datastore pointing to a blob container",
    account_name="mytestblobstore",
    container_name="data-container",
    credentials=SasTokenCredentials(
        sas_token="?xx=XXXX-XX-XX&xx=xxxx&xxx=xxx&xx=xxxxxxxxxxx&xx=XXXX-XX-XXXXX:XX:XXX&xx=XXXX-XX-XXXXX:XX:XXX&xxx=xxxxx&xxx=XXxXXXxxxxxXXXXXXXxXxxxXXXXXxxXXXXXxXXXXxXXXxXXxXX"
    ),
)
ml_client.create_or_update(blob_datastore)
```

We can create data assets to get access to data in datastores, Azure storage services, public URLs, or data stored on your local device. The advantages are that we can version the metadata of the data asset, can share and reuse and do not have to care about connection string and data paths.

There are 3 types of data assets:

1. `URI file`: which points to a file.

    ```python
    # To create a URI file data asset:
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    my_path = '<supported-path>'
    my_data = Data(
        path=my_path,
        type=AssetTypes.URI_FILE,
        description="<description>",
        name="<name>",
        version="<version>"
    )
    ml_client.data.create_or_update(my_data)

    # to read data:
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.input_data)
    ```

2. `URI folder`: which points to a folder. Python SDK code is also quite similar.
3. `MLTable`: which can point to folder or file and includes a schema to read as tabular data. So even if data schema changes it will not affect when we are reading since MLTable already has a schema and it will use same. To define the schema we include a MLTable file in the same folder as the data we want to read.

    ```yml
    type: mltable

    paths:
    - pattern: ./*.txt
    transformations:
    - read_delimited:
        delimiter: ','
        encoding: ascii
        header: all_files_same_headers
    ````

### Chapter 8. Work with compute targets in Azure Machine Learning

Types of compute in Azure ML:

1. `Compute instance`: Behaves similarly to a virtual machine and is primarily used to run notebooks. It's ideal for experimentation. It can not handle parallel tasks and at one time can be used by one user only.
2. `Compute clusters`: Multi-node clusters of virtual machines that automatically scale up or down to meet demand. A cost-effective way to run scripts that need to process large volumes of data. Clusters also allow you to use parallel processing to distribute the workload and reduce the time it takes to run a script.
3. `Kubernetes clusters`: Cluster based on Kubernetes technology, giving you more control over how the compute is configured and managed. You can attach your self-managed Azure Kubernetes (AKS) cluster for cloud compute, or an Arc Kubernetes cluster for on-premises workloads.
4. `Attached compute`: Allows you to attach existing compute like Azure virtual machines or Azure Databricks clusters to your workspace.
5. `Serverless compute`: A fully managed, on-demand compute you can use for training jobs.

```python
    # To create a compute instance:
    from azure.ai.ml.entities import ComputeInstance

    ci_basic_name = "basic-ci-12345"
    ci_basic = ComputeInstance(
        name=ci_basic_name, 
        size="STANDARD_DS3_v2"
    )
    ml_client.begin_create_or_update(ci_basic).result()

    # To create a compute cluster:
    from azure.ai.ml.entities import AmlCompute

    cluster_basic = AmlCompute(
        name="cpu-cluster",
        type="amlcompute",
        size="STANDARD_DS3_v2",
        location="westus",
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120,
        tier="low_priority",
    )
    ml_client.begin_create_or_update(cluster_basic).result()
```

### Chapter 9. Work with environments in Azure Machine Learning

A lot of curated environment are already available in our workspaces. They use Prefix AzureML.

```python
    # To check all the environments available:
    envs = ml_client.environments.list()
    for env in envs:
        print(env.name)

    # To check details of a specific environment:
    env = ml_client.environments.get(name="<name>", version="<version>")
    print(env)
```

When running a job we can specify name of environment to be used.  
We can define an environment from a Docker image, a Docker build context, and a conda specification with Docker image.

## Section 3. Experiment with Azure Machine Learning

### Chapter 10. Find the best classification model with Automated Machine Learning

```python
# To use MLTable datasset as input for Automated Machine Learning
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml import Input

    my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")
```

We can also configure optional featurization which can include:

- Missing value imputation
- Categorical encoding
- Numerical normalization
- Feature selection or Dropping high-cardinality features
- Feature engineering (like extracting date from datetime features)

We can turn them off if we want. After AutoML experiment we can also check what all scaling or normalization was applied and if AutoML detected any issue with the data. Usually there are three status Passed, Done (where AutoML made some changes, we should review that) and Alerted (where AutoML detected some issue with the data but could not fix it, we should review).


AutoML requires MLTable data asset as input.

```python
    from azure.ai.ml import automl

    # configure the classification job
    classification_job = automl.classification(
        compute="aml-cluster",
        experiment_name="auto-ml-class-dev",
        training_data=my_training_data_input,
        target_column_name="Diabetic",
        primary_metric="accuracy",
        n_cross_validations=5,
        enable_model_explainability=True
    )
```

By default AutoML runs all the algorithms for the particular type of ML for example classification. But we can customize and restrict this based on our needs. Also we need to provide the primary metric based on which AutoML will choose the best model. All the metrics with details are listed [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml?view=azureml-api-2). Using SDK to check all the metrics available under classification, we can use:

```python
    from azure.ai.ml.automl import ClassificationPrimaryMetrics
    list(ClassificationPrimaryMetrics)
```

To set limits on cost and time we can use :

```python
    classification_job.set_limits(
        timeout_minutes=60, 
        trial_timeout_minutes=20, 
        max_trials=5,
        enable_early_termination=True,
    )
```

We can also set max_concurrent_trials which should be equal to or less than total number of nodes in compute cluster.  
To monitor job:

```python
    # submit the AutoML job
    returned_job = ml_client.jobs.create_or_update(
        classification_job
    ) 
    aml_url = returned_job.studio_url
    print("Monitor your job at", aml_url)
```

### Chapter 11. Track model training in Jupyter notebooks with MLflow

MLflow is an open-source library for tracking and managing your machine learning experiments. In particular, MLflow Tracking is a component of MLflow that logs everything about the model you're training, such as parameters, metrics, and artifacts. The azureml-mlflow package contains the integration code of Azure Machine Learning with MLflow. To use these:

```python
    pip show mlflow
    pip show azureml-mlflow
```

To group model training results, we use experiments. To track model metrics with MLflow when training a model in a notebook, we can use MLflow's logging capabilities. MLFlow supports both autologging and custom logging.

```python
    # To create an experiment
    import mlflow
    mlflow.set_experiment(experiment_name="heart-condition-classifier")

    # To enable autolog for xgboost
    from xgboost import XGBClassifier
    with mlflow.start_run():
        mlflow.xgboost.autolog()
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # For custom logging
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    with mlflow.start_run():
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy) # We can use log_param, log_metric, log_artifact also
```

## Section 4. Optimize model training with Azure Machine Learning

### Chapter 12. Run a training script as a command job in Azure Machine Learning

MLflow is an open-source platform, designed to manage the complete machine learning lifecycle.  
There are two options to track machine learning jobs with MLflow:

- Enable autologging using `mlflow.autolog`
- Use logging functions to track custom metrics using `mlflow.log_*`

To setup the invironment which includes MLflow, `mlflow` and `azureml-flow` packages needs to be installed.

```yml
    name: mlflow-env
    channels:
    - conda-forge
    dependencies:
    - python=3.8
    - pip
    - pip:
        - numpy
        - pandas
        - scikit-learn
        - matplotlib
        - mlflow
        - azureml-mlflow
```

Now for aulogging we can use `mlflow.autolog`. Autologging is supported for most popular machine learning libraries.
For custom metric we can use `mlflow.log_*`. For example - `mlflow.log_metric("accuracy", accuracy)`, mlflow.log_param("regularization rate", reg_rate) or mlflow.log_artifact("model.pkl")

When we run jobs, all the metrics and artifacts will be tracked. We can use ML studio or we can retrieve metrics related to job run using Mlflow.

```python
    # Using MLflow getting run metrics
    # To get details of active experiments
    experiments = mlflow.search_experiments(max_results=2)
    for exp in experiments:
        print(exp.name)
    # If we want even archived experiments
    from mlflow.entities import ViewType
    experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    for exp in experiments:
        print(exp.name)
    # To retrive a specific experiment
    exp = mlflow.get_experiment_by_name(experiment_name)
    print(exp)
    # To know metrics of a specifc run
    mlflow.search_runs(exp.experiment_id)
    # To show only the last two results by start time
    mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=2)
    # We can also filter the results by hyperparameters
    mlflow.search_runs(
        exp.experiment_id, filter_string="params.num_boost_round='100'", max_results=2
    )
```

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
