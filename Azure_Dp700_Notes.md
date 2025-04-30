# Microsoft Fabric Notes

## Table of Contents

1. [Introduction to end-to-end analytics using Microsoft Fabric](#introduction-to-end-to-end-analytics-using-microsoft-fabric)
2. [Get started with lakehouses in Microsoft Fabric](#get-started-with-lakehouses-in-microsoft-fabric)
3. [Organize a Fabric lakehouse using medallion architecture design](#organize-a-fabric-lakehouse-using-medallion-architecture-design)
4. [Use Apache Spark in Microsoft Fabric](#use-apache-spark-in-microsoft-fabric)
5. [Work with Delta Lake tables in Microsoft Fabric](#work-with-delta-lake-tables-in-microsoft-fabric)
6. [Ingest Data with Dataflows Gen2 in Microsoft Fabric](#ingest-data-with-dataflows-gen2-in-microsoft-fabric)
7. [Orchestrate processes and data movement with Microsoft Fabric](#orchestrate-processes-and-data-movement-with-microsoft-fabric)
8. [Get started with data warehouses in Microsoft Fabric](#get-started-with-data-warehouses-in-microsoft-fabric)
9. [Load data into a Microsoft Fabric data warehouse](#load-data-into-a-microsoft-fabric-data-warehouse)
10. [Query a data warehouse in Microsoft Fabric](#query-a-data-warehouse-in-microsoft-fabric)
11. [Monitor a Microsoft Fabric data warehouse](#monitor-a-microsoft-fabric-data-warehouse)
12. [Secure a Microsoft Fabric data warehouse](#secure-a-microsoft-fabric-data-warehouse)
13. [Get started with data science in Microsoft Fabric](#get-started-with-data-science-in-microsoft-fabric)
14. [Get started with Real-Time Intelligence in Microsoft Fabric](#get-started-with-real-time-intelligence-in-microsoft-fabric)
15. [Use real-time eventstreams in Microsoft Fabric](#use-real-time-eventstreams-in-microsoft-fabric)
16. [Work with real-time data in a Microsoft Fabric eventhouse](#work-with-real-time-data-in-a-microsoft-fabric-eventhouse)
17. [Create Real-Time Dashboards with Microsoft Fabric](#create-real-time-dashboards-with-microsoft-fabric)
18. [Monitor activities in Microsoft Fabric](#monitor-activities-in-microsoft-fabric)
19. [Secure data access in Microsoft Fabric](#secure-data-access-in-microsoft-fabric)
20. [Implement continuous integration and continuous delivery (CI/CD) in Microsoft Fabric](#implement-continuous-integration-and-continuous-delivery-cicd-in-microsoft-fabric)
21. [Administer a Microsoft Fabric environment](#administer-a-microsoft-fabric-environment)
22. [PySpark/SparkSQL Cheatsheet](#pysparksparksql-cheatsheet)

---

## Introduction to end-to-end analytics using Microsoft Fabric

### Overview of Microsoft Fabric
- **Fabric** is a unified software-as-a-service (SaaS) platform where all data is stored in a single open format in **OneLake**.
- **OneLake** ensures scalability, cost-effectiveness, and accessibility from anywhere with an internet connection.

### OneLake: Centralized Data Storage
- **OneLake** is Fabric's centralized data storage architecture that eliminates the need to move or copy data between systems.
- It unifies data across regions and clouds into a single logical lake without moving or duplicating data.
- Built on **Azure Data Lake Storage (ADLS)**, it supports various formats, including:
  - **Delta**
  - **Parquet**
  - **CSV**
  - **JSON**
- All compute engines in Fabric automatically store their data in OneLake, making it directly accessible without movement or duplication.
- For tabular data, analytical engines write data in **delta-parquet format**, ensuring seamless interaction across engines.

#### Shortcuts in OneLake
- Shortcuts are references to files or storage locations external to OneLake.
- They allow access to existing cloud data without copying it, ensuring data consistency and synchronization with the source.

### Workspaces in Microsoft Fabric
- **Workspaces** serve as logical containers to organize and manage data, reports, and other assets.
- They provide clear separation of resources, making it easier to control access and maintain security.
- Key features of workspaces:
  - Each workspace has its own set of permissions to ensure authorized access.
  - Manage compute resources and integrate with **Git** for version control.
  - Support for **data lineage view**, providing a visual representation of data flow and dependencies.

#### Workspace Settings
- Configure the following in workspace settings:
  - **License type** to use Fabric features.
  - **OneDrive access** for the workspace.
  - **Azure Data Lake Gen2 Storage** connection.
  - **Git integration** for version control.
  - **Spark workload settings** for performance optimization.
- Manage workspace access through four roles:
  - **Admin**
  - **Contributor**
  - **Member**
  - **Viewer**
- For granular access control, use **item-level permissions** based on business needs.

#### OneLake Catalog
- The **OneLake catalog** helps analyze, monitor, and maintain data governance by providing:
  - Guidance on sensitivity labels.
  - Item metadata.
  - Data refresh status.
- The **OneLake catalog** helps users find and access various data sources within their organization.
- Users can explore and connect to data sources, ensuring they have the right data for their needs.
- Offers insights into governance status and actions for improvement.
- Users only see items shared with them.

### Administration and Governance
- Fabric's **OneLake** is centrally governed and open for collaboration.
- Administration is centralized in the **Admin portal**, where you can:
  - Manage groups and permissions.
  - Configure data sources and gateways.
  - Monitor usage and performance.
  - Access Fabric admin APIs and SDKs to automate tasks and integrate with other systems.

#### Admin Roles
- **Fabric admin (formerly Power BI admin):** Manages Fabric settings and configurations.
- **Power Platform admin:** Oversees Power Platform services, including Fabric.
- **Microsoft 365 admin:** Manages organization-wide Microsoft services, including Fabric.
- Admins can enable Fabric in the **Admin portal > Tenant settings** in the Power BI service.
- Fabric can be enabled for The entire organization or Specific Microsoft 365 or Microsoft Entra security groups.
- Admins can delegate this ability to other users at the capacity level.



## Get started with lakehouses in Microsoft Fabric

### Overview of Lakehouses
- A **lakehouse** is the foundation of Microsoft Fabric, built on top of the **OneLake** scalable storage layer.
- Combines the flexibility and scalability of a **data lake** with the query and analytical capabilities of a **data warehouse**.
- Built using **Delta format tables**, lakehouses unify:
  - SQL-based analytical capabilities of a relational data warehouse.
  - Flexibility and scalability of a data lake.

### Key Features of Lakehouses
- **Unified Platform:**
  - Supports all data formats and integrates with various analytics tools and programming languages.
  - Provides a single location for data engineers, data scientists, and analysts to access and use data.
- **Scalability and Availability:**
  - Cloud-based solutions that scale automatically.
  - High availability and disaster recovery.
- **Data Processing Engines:**
  - Uses **Apache Spark** and **SQL compute engines** for big data processing.
  - Supports machine learning and predictive modeling analytics.
- **Schema-on-Read:**
  - Data is organized in a schema-on-read format, allowing schema definition as needed.
- **ACID Transactions:**
  - Ensures data consistency and integrity through **Delta Lake** formatted tables.

### Benefits of Lakehouses
- Centralized data storage and processing.
- Seamless integration with Microsoft Fabric tools.
- Enhanced data consistency and governance.

### Data Ingestion and Transformation
- **ETL Process:**
  - Follow the **Extract, Transform, Load (ETL)** process to ingest and transform data before loading it into the lakehouse.
- **Data Ingestion Methods:**
  - **Upload:** Upload local files directly.
  - **Dataflows Gen2:** Import and transform data using Power Query.
  - **Notebooks:** Use Apache Spark to ingest, transform, and load data.
  - **Data Factory Pipelines:** Use the Copy Data activity to orchestrate ETL activities.
- **Shortcuts:**
  - Create shortcuts to access external data sources like **Azure Data Lake Store Gen2** or **OneLake** without copying data.

### Working with Lakehouses
- **Lakehouse Explorer:**
  - Browse files, folders, shortcuts, and tables within the Fabric platform.
  - View and manage contents easily.
- **Data Transformation:**
  - Use **Apache Spark notebooks** or **Dataflows Gen2** for data transformation.
  - **Dataflows Gen2** provides a visual representation of transformations using Power Query.
  - Use **Data Factory Pipelines** for orchestrating complex ETL processes.

### Components of a Lakehouse
- **Lakehouse:**
  - Contains shortcuts, folders, files, and tables.
- **Semantic Model:**
  - Provides an easy data source for Power BI report developers.
- **SQL Analytics Endpoint:**
  - Allows read-only access to query data with SQL.

### Modes of Interaction
- **Lakehouse Mode:**
  - Add and interact with tables, files, and folders.
- **SQL Analytics Endpoint Mode:**
  - Use SQL to query tables and manage the relational semantic model.

### Tools for Data Engineers
- **Notebooks:**
  - Ideal for engineers familiar with programming languages like **PySpark**, **SQL**, and **Scala**.
- **Dataflows Gen2:**
  - Suitable for developers familiar with Power BI or Excel, leveraging the Power Query interface.
- **Pipelines:**
  - Provide a visual interface for performing and orchestrating ETL processes, ranging from simple to complex workflows.

## Organize a Fabric lakehouse using medallion architecture design

### Overview of Medallion Architecture
- **Medallion architecture** is a recommended data design pattern for organizing data in a lakehouse logically. Built on the **Delta Lake format**, it natively supports **ACID (Atomicity, Consistency, Isolation, Durability)** transactions. It aims to improve data quality as it moves through different layers.
- Typically consists of three layers: **Bronze (Raw)**, **Silver (Validated)**, and **Gold (Enriched)**. Also referred to as a "multi-hop" architecture, allowing data to move between layers as needed.
- Flexible design: Use the same lakehouse for multiple medallion architectures or different lakehouses across workspaces based on use cases.

### Bronze Layer: Raw Data
- The **bronze layer** is the landing zone for all data, whether structured, semi-structured, or unstructured. Data is stored in its original format without any modifications.
- Tools used: **Pipelines**, **Dataflows**, **Notebooks**.

### Silver Layer: Validated Data
- The **silver layer** is where data is validated and refined. Activities include combining and merging data, enforcing data validation rules (e.g., removing nulls, deduplication).
- Acts as a central repository for consistent data accessible by multiple teams.
- Tools used: **Dataflows**, **Notebooks**.

### Gold Layer: Enriched Data
- The **gold layer** is where data is further refined to meet specific business and analytics needs. Activities include aggregating data to specific granularities (e.g., daily, hourly) and enriching data with external information.
- Data in the gold layer is ready for use by downstream teams, including analytics, data science, and MLOps. Recommended to model the gold layer in a **star schema**.
- Tools used: **SQL Analytics Endpoint**, **Semantic Model**.

### Customizing the Medallion Architecture
- The medallion architecture is flexible and can be tailored to meet specific organizational needs. Examples of customization:
  - Adding a "raw" layer for landing data in a specific format before transforming it into the bronze layer.
  - Adding a "platinum" layer for further refined and enriched data for specific use cases.
- Regardless of the number of layers, the architecture can adapt to your organization's requirements.

### Querying Data in the Medallion Architecture
- Use either the **SQL Analytics Endpoint** or **Power BI Semantic Model** in **Direct Lake Mode** to query data.
- **SQL Analytics Endpoint** works in read-only mode.
- **Semantic Model** is created automatically when a lakehouse is created. It provides metrics on top of lakehouse data and connects to data using **Direct Lake Mode**, which caches frequently used data, refreshes data as required, and combines the speed of a semantic model with up-to-date lakehouse data.

## Use Apache Spark in Microsoft Fabric

### Overview of Apache Spark
- **Apache Spark** is a distributed data processing framework that enables large-scale data analytics by coordinating work across multiple processing nodes in a cluster, known in Microsoft Fabric as a **Spark pool**.
- Spark supports multiple programming languages, including:
  - **Java**
  - **Scala** (a Java-based scripting language)
  - **Spark R**
  - **Spark SQL**
  - **PySpark** (a Spark-specific variant of Python).

### Spark Pool Architecture
- A **Spark pool** contains two types of nodes:
  - **Head Node:** Coordinates distributed processes through a driver program.
  - **Worker Nodes:** Perform actual data processing tasks through executor processes.
- Microsoft Fabric provides a **starter pool** in each workspace, enabling quick setup and execution of Spark jobs with minimal configuration.

### Configuring Spark Pools
- Key configuration settings for Spark pools include:
  - **Node Family:** Type of virtual machines used for the Spark cluster nodes (e.g., memory-optimized nodes for better performance).
  - **Autoscale:** Automatically provisions nodes as needed, with options to set initial and maximum node counts.
  - **Dynamic Allocation:** Dynamically allocates executor processes on worker nodes based on data volumes.
- Multiple pools can be created, and one can be configured as the default.

### Spark Runtime and Libraries
- The **Spark runtime** determines the versions of Apache Spark, Delta Lake, Python, and other core components installed.
- The current default runtime version is **1.3**.
- Within a runtime, you can install and use a wide selection of libraries for common or specialized tasks.
- Custom environments can be created in a Fabric workspace to use specific Spark runtimes, libraries, and configurations for different operations.

### Native Execution Engine
- The **native execution engine** in Microsoft Fabric is a vectorized processing engine that runs Spark operations directly on lakehouse infrastructure.
- Benefits of the native execution engine:
  - Significantly improves query performance for large datasets in **Parquet** or **Delta** file formats.
- To enable the native execution engine, set the following Spark properties:
  - `spark.native.enabled: true`
  - `spark.shuffle.manager: org.apache.spark.shuffle.sort.ColumnarShuffleManager`

### Optimizing Spark Sessions
- When running Spark code in Microsoft Fabric, a **Spark session** is initiated.
- **High Concurrency Mode:**
  - Shares Spark sessions across multiple concurrent users or processes.
  - Ensures isolation of code to avoid variable conflicts between notebooks.
  - Can also be enabled for Spark jobs to optimize non-interactive script execution.

### Machine Learning with MLFlow
- **MLFlow** is an open-source library used to manage machine learning training and model deployment.
- Key capabilities of MLFlow in Microsoft Fabric:
  - Implicitly logs machine learning experiment activity without requiring explicit code.
  - This functionality can be disabled in workspace settings if needed.

### Schema Management
- While Spark can infer schemas, explicitly providing schemas improves performance.

## Work with Delta Lake tables in Microsoft Fabric

## Ingest Data with Dataflows Gen2 in Microsoft Fabric

## Orchestrate processes and data movement with Microsoft Fabric

## Get started with data warehouses in Microsoft Fabric

## Load data into a Microsoft Fabric data warehouse

## Query a data warehouse in Microsoft Fabric

## Monitor a Microsoft Fabric data warehouse

## Secure a Microsoft Fabric data warehouse

## Get started with data science in Microsoft Fabric

## Get started with Real-Time Intelligence in Microsoft Fabric

## Use real-time eventstreams in Microsoft Fabric

## Work with real-time data in a Microsoft Fabric eventhouse

## Create Real-Time Dashboards with Microsoft Fabric

## Monitor activities in Microsoft Fabric

## Secure data access in Microsoft Fabric

## Implement continuous integration and continuous delivery (CI/CD) in Microsoft Fabric

## Administer a Microsoft Fabric environment

## PySpark/SparkSQL Cheatsheet

### PySpark Basics
- **Create a SparkSession:**
  ```python
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName("AppName").getOrCreate()
  ```
- **Read a DataFrame:**
  ```python
  df = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)
  ```
- **Show DataFrame:**
  ```python
  df.show()
  ```
- **Write a DataFrame:**
  ```python
  df.write.csv("path/to/output.csv", header=True)
  ```

### Common DataFrame Operations
- **Select Columns:**
  ```python
  df.select("column1", "column2").show()
  ```
- **Filter Rows:**
  ```python
  df.filter(df["column"] > 10).show()
  ```
- **Group By and Aggregate:**
  ```python
  df.groupBy("column").agg({"column2": "sum"}).show()
  ```
- **Add a New Column:**
  ```python
  from pyspark.sql.functions import col
  df = df.withColumn("new_column", col("existing_column") * 2)
  ```
- **Drop a Column:**
  ```python
  df = df.drop("column_to_drop")
  ```

### SparkSQL Basics
- **Create a Temporary View:**
  ```python
  df.createOrReplaceTempView("table_name")
  ```
- **Run SQL Queries:**
  ```python
  result = spark.sql("SELECT * FROM table_name WHERE column > 10")
  result.show()
  ```

### Advanced PySpark Operations
- **Join DataFrames:**
  ```python
  df1.join(df2, df1["key"] == df2["key"], "inner").show()
  ```
- **Window Functions:**
  ```python
  from pyspark.sql.window import Window
  from pyspark.sql.functions import row_number
  windowSpec = Window.partitionBy("column").orderBy("column2")
  df = df.withColumn("row_number", row_number().over(windowSpec))
  ```
- **Pivot Table:**
  ```python
  df.groupBy("column1").pivot("column2").sum("value_column").show()
  ```

### Performance Tips
- **Cache DataFrame:**
  ```python
  df.cache()
  ```
- **Repartition DataFrame:**
  ```python
  df = df.repartition(4)
  ```
- **Broadcast Join:**
  ```python
  from pyspark.sql.functions import broadcast
  df1.join(broadcast(df2), "key").show()
  ```


