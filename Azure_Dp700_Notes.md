# Microsoft Fabric Notes

## Table of Contents

1. [Introduction to end-to-end analytics using Microsoft Fabric](#introduction-to-end-to-end-analytics-using-microsoft-fabric)
2. [Get started with lakehouses in Microsoft Fabric](#get-started-with-lakehouses-in-microsoft-fabric)
3. [Organize a Fabric lakehouse using medallion architecture design](#organize-a-fabric-lakehouse-using-medallion-architecture-design)
4. [Use Apache Spark in Microsoft Fabric](#use-apache-spark-in-microsoft-fabric)
5. [Work with Delta Lake tables in Microsoft Fabric](#work-with-delta-lake-tables-in-microsoft-fabric)
6. [Ingest Data with Dataflows Gen2 in Microsoft Fabric](#ingest-data-with-dataflows-gen2-in-microsoft-fabric)
7. [Govern data in Microsoft Fabric with Purview](#Govern-data-in-Microsoft-Fabric-with-Purview)
8. [Get started with data warehouses in Microsoft Fabric](#get-started-with-data-warehouses-in-microsoft-fabric)
9. [Get started with SQL Database in Microsoft Fabric](#get-started-with-sql-database-in-microsoft-fabric)
10. [Load data into a Microsoft Fabric data warehouse](#load-data-into-a-microsoft-fabric-data-warehouse)
11. [Query a data warehouse in Microsoft Fabric](#query-a-data-warehouse-in-microsoft-fabric)
12. [Monitor a Microsoft Fabric data warehouse](#monitor-a-microsoft-fabric-data-warehouse)
13. [Secure a Microsoft Fabric data warehouse](#secure-a-microsoft-fabric-data-warehouse)
14. [Get started with data science in Microsoft Fabric](#get-started-with-data-science-in-microsoft-fabric)
15. [Get started with Real-Time Intelligence in Microsoft Fabric](#get-started-with-real-time-intelligence-in-microsoft-fabric)
16. [Use real-time eventstreams in Microsoft Fabric](#use-real-time-eventstreams-in-microsoft-fabric)
17. [Work with real-time data in a Microsoft Fabric eventhouse](#work-with-real-time-data-in-a-microsoft-fabric-eventhouse)
18. [Create Real-Time Dashboards with Microsoft Fabric](#create-real-time-dashboards-with-microsoft-fabric)
19. [Monitor activities in Microsoft Fabric](#monitor-activities-in-microsoft-fabric)
20. [Secure data access in Microsoft Fabric](#secure-data-access-in-microsoft-fabric)
21. [Implement continuous integration and continuous delivery (CI/CD) in Microsoft Fabric](#implement-continuous-integration-and-continuous-delivery-cicd-in-microsoft-fabric)
22. [Govern data in Microsoft Fabric with Purview](#govern-data-in-microsoft-fabric-with-purview)
23. [Administer a Microsoft Fabric environment](#administer-a-microsoft-fabric-environment)
24. [PySpark/SparkSQL Cheatsheet](#pysparksparksql-cheatsheet)

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

### Overview of Delta Lake Tables
- Tables in a Microsoft Fabric lakehouse are based on the **Linux Foundation Delta Lake table format**, commonly used in Apache Spark.
- **Delta Lake** is an open-source storage layer that adds relational database semantics to Spark-based data lake processing.
- The underlying data for Delta tables is stored in **Parquet format**.

### Table Types in Delta Lake
- **Managed Table:** Fully managed by the lakehouse.
- **External Table:** Created by specifying a path to external storage.
- **Metadata-Only Table:** Created using `DeltaTableBuilder` to define table metadata without data.

### File Management and Optimization
- **Parquet Files:**
  - Parquet files are immutable, so updates create new files, leading to the **small file problem**, which impacts performance.
- **OptimizeWrite:**
  - Enabled by default in Fabric to reduce the number of files written.
  - Consolidates small files into fewer, larger files for better performance.
- **Optimize Feature:**
  - Can be run under maintenance in **Lakehouse Explorer** to further consolidate files.

### V-Order Optimization
- **V-Order** (enabled by default in Fabric):
  - Enables lightning-fast reads with in-memory-like data access times.
  - Improves cost efficiency by reducing network, disk, and CPU resources during reads.
  - Adds ~15% overhead during writes but significantly speeds up reads.
- **Microsoft Verti-Scan Technology:**
  - Used by Power BI and SQL engines to fully leverage V-Order optimization for faster reads.
  - Spark and other engines benefit from V-Order optimization with ~10% faster reads, sometimes up to 50%.
- **How V-Order Works:**
  - Applies special sorting, row group distribution, dictionary encoding, and compression on Parquet files.

### Maintenance with VACUUM
- The **VACUUM** command removes old Parquet data files but retains transaction logs.
- **Time Travel:**
  - Old Parquet files are retained to enable time travel.
  - Running VACUUM prevents time travel earlier than the retention period.
- Example Command:
  ```sql
  DESCRIBE HISTORY products
  ```
  - Shows the history of the `products` table.

### Streaming Data with Delta Lake
- **Spark Structured Streaming:**
  - Native support for streaming data through an API based on a boundless DataFrame.
  - Captures streaming data for processing efficiently.

## Ingest Data with Dataflows Gen2 in Microsoft Fabric

Dataflows are a type of cloud-based ETL (Extract, Transform, Load) tool for building and executing scalable data transformation processes. Dataflows Gen2 allow you to extract data from various sources, transform it using a wide range of transformation operations, and load it into a destination. Using Power Query Online also allows for a visual interface to perform these tasks.
In Microsoft Fabric, you can create a Dataflow Gen2 in the Data Factory workload or Power BI workspace, or directly in the lakehouse.
Dataflow Gen2 does not support row level security.

Data pipelines are a common concept in data engineering and offer a wide variety of activities to orchestrate. Some common activities include:

Copy data
Incorporate Dataflow
Add Notebook
Get metadata
Execute a script or stored procedure
Pipelines provide a visual way to complete activities in a specific order

Pipelines can also be scheduled or activated by a trigger to run your dataflow.

## Govern data in Microsoft Fabric with Purview

Microsoft Fabric brings together features from Power BI, Azure Synapse Analytics, and Azure Data Factory, along with new capabilities, to deliver a unified platform for data governance and management. Below is a structured overview of how governance is achieved in Fabric, with a focus on Microsoft Purview integration.

### Key Data Experiences in Fabric
- **Data Factory:** Ingest, prepare, and transform data from diverse sources using a wide range of connectors.
- **Synapse Data Engineering:** Use Spark for large-scale data collection, storage, processing, and analysis via jobs and notebooks.
- **Synapse Data Warehouse:** Separate compute from storage, scale independently, and store data in Data Lake format for high-performance SQL analytics.
- **Synapse Data Science:** Train, deploy, and use machine learning models.
- **Real-Time Intelligence:** Manage event-driven data such as telemetry, logs, and streaming data.
- **Power BI:** Visualize, analyze, and share insights to support decision-making.
- **Data Activator:** Automate actions (emails, workflows) triggered by data conditions—no code required.

### Unified Storage and Data Lake
- All data in Fabric is stored in **OneLake**, built on Azure Data Lake Storage (ADLS) Gen2.
- **OneLake** supports both structured and unstructured data, eliminating silos and reducing costs.
- Universal policies and security are applied across all data assets.

### Built-in Data Governance Tools
- **No extra subscription required:** Many governance features are available out-of-the-box in Fabric.
- **Fabric Admin Portal:** Centralized control for tenant settings, capacities, domains, and more (admin access required).
- **Tenants, Domains, and Workspaces:**
  - **Tenants:** Organization-wide boundary for governance.
  - **Domains:** Group data by business area or subject.
  - **Workspaces:** Organize Fabric items for teams or departments.
- **Capacities:** Limit compute resource usage for all workloads.
- **Metadata Scanning:** Extracts metadata (names, identities, sensitivities, endorsements, etc.) from data lakes to support governance and policy enforcement.

### Microsoft Purview Integration
- **Microsoft Purview** is a comprehensive data governance and compliance platform for managing and protecting data assets.
- **Key Capabilities:**
  - **Data Discovery, Classification, and Cataloging:** Automates identification of sensitive data and centralizes metadata.
  - **Compliance & Risk Management:** Monitors regulatory adherence and assesses vulnerabilities.
  - **Information Protection:**
    - Classify, label, and protect sensitive data with customizable sensitivity labels.
    - Policies define access controls and enforce encryption.
    - Labels persist with data across emails, documents, and cloud storage.
    - Integrates with Data Loss Prevention (DLP) to prevent unauthorized sharing.
  - **Data Loss Prevention (DLP):**
    - Automatically detect, monitor, and control sharing/movement of sensitive data.
    - Customizable rules to block, restrict, or alert on sensitive data transfers.
    - Currently, DLP policies are only supported for Power BI semantic models.
  - **Audit Logging:**
    - User activities (file creation, item access) are logged and available in the Purview audit log.

### Connecting Purview to Fabric
- Register your Fabric tenant in Purview to enable discovery and management of all Fabric items.
- Once connected, the **Microsoft Purview hub** appears in Fabric, providing rich analysis and reporting.
- **Cross-Tenant Considerations:**
  - If Fabric and Purview are in different tenants, all features are supported except live view for Fabric items.
  - Manual entry of tenant ID is required; managed identity authentication is not supported for cross-tenant connections (use service principal or delegated authentication).

### Microsoft Purview Hub in Fabric
- The main section is **Microsoft Fabric data**, with two displays: **Items** and **Sensitivity**.
- **Open full report** for detailed analysis, including:
  - **Overview Report:** Summary of endorsements, labels, and workspaces for a broad view of data usage.
  - **Endorsement Report:** Analyze item endorsements and user confidence in data quality.
  - **Sensitivity Report:** Review sensitivity label usage and ensure correct application for compliance.
  - **Inventory Report:** Drill down by date, item type, and other filters to identify specific items.
  - **Items Page:** Insights into item distribution and endorsement coverage across the organization.
  - **Sensitivity Page:** Analyze sensitivity labeling by department and geography to identify gaps or best practices.

### Best Practices for Data Governance in Fabric
- **Centralize governance** using the Fabric Admin portal and Purview integration.
- **Classify and label** sensitive data with Information Protection and sensitivity labels.
- **Monitor and audit** user activities and data movement with Purview audit logs.
- **Leverage reports** in the Purview hub to identify gaps in endorsements, labeling, and compliance.
- **Apply DLP policies** where supported to prevent data leaks and unauthorized sharing.

**Summary:**
Microsoft Fabric, with Microsoft Purview, provides a robust, unified approach to data governance—enabling organizations to discover, classify, protect, and monitor their data estate efficiently and securely.

## Get started with data warehouses in Microsoft Fabric

### Overview of Data Warehouses
- Fabric's data warehouse contains **tables** to store data for analytics, organized in schemas optimized for **multidimensional modeling**.
- Tables are structured into:
  - **Fact Tables:** Contain numerical data for analysis (e.g., total sales amount).
  - **Dimension Tables:** Contain descriptive information to provide context for fact tables (e.g., customer details).

### Keys in Dimension Tables
- **Surrogate Key:** A unique identifier for each row in the dimension table, often an auto-generated integer.
- **Alternate Key:** A natural or business key (e.g., product code) that identifies specific entities in the source system.
- Both keys are essential for maintaining consistency and traceability between the data warehouse and source systems.

### Special Types of Dimensions
- **Time Dimensions:** Provide temporal context (e.g., year, quarter, month) for aggregating data.
- **Slowly Changing Dimensions:** Track changes to attributes over time (e.g., customer address changes).

### Schema Design
- **Star Schema:** Fact tables are directly related to dimension tables, enabling grouping and aggregation.
- **Snowflake Schema:** Used when there are many levels or shared information across dimensions.

### Transition from Lakehouse to Data Warehouse
- Fabric's **Lakehouse** acts as a database over a data lake, supporting big data processing with Spark and SQL engines.
- The **data warehouse experience** enables:
  - Modeling data using tables and views.
  - Running T-SQL queries across the data warehouse and Lakehouse.
  - Performing DML operations and serving reporting layers like Power BI.

### Data Ingestion Methods
- **Pipelines:** Automate data workflows.
- **Dataflows:** Import and transform data visually.
- **Cross-Database Querying:** Query data across databases.
- **COPY INTO Command:** Load data efficiently into tables.

### Table Clones
- **Zero-Copy Clones:** Create replicas of tables by copying metadata without duplicating data files.
- **Use Cases:**
  - Development and testing.
  - Data recovery.
  - Historical reporting.
- **Command Example:**
  ```sql
  CREATE TABLE new_table AS CLONE OF existing_table;
  ```

### Data Warehouse Load Process
1. **Ingest Data:** Load new data into a data lake with pre-load cleansing.
2. **Load Staging Tables:** Transfer data into staging tables in the warehouse.
3. **Load Dimension Tables:** Update or insert rows, generating surrogate keys as needed.
4. **Load Fact Tables:** Populate fact tables, looking up surrogate keys for dimensions.
5. **Post-Load Optimization:** Update indexes and table distribution statistics.

### Cross-Database Querying
- Query data in the Lakehouse directly from the data warehouse without copying it.
- **Query Tools:**
  - **Visual Query Editor:** Drag-and-drop interface for no-code queries.
  - **SQL Query Editor:** Write T-SQL queries for advanced operations.

### SQL Analytics Endpoint
- Connect to the data warehouse from any tool using the **SQL analytics endpoint**.

### Semantic Models in Power BI
- **Semantic Models:** Define relationships, calculations, and structure in data for meaningful analysis.
- **Default Model:** Auto-created with a Fabric data warehouse and updates with new tables.
- **Features:**
  - Create relationships between tables.
  - Hide fields to simplify views.
  - Use DAX for calculated measures.
- Enable easy creation and sharing of visual reports in Power BI.

### Dynamic Management Views (DMVs)
- **Available DMVs:**
  - `sys.dm_exec_connections`: Information about connections.
  - `sys.dm_exec_sessions`: Information about authenticated sessions.
  - `sys.dm_exec_requests`: Information about active requests.
- **KILL Command:** Terminate sessions with long-running queries (requires Workspace Admin role).

## Load data into a Microsoft Fabric data warehouse

### Overview of Microsoft Fabric Data Warehouse
- A **Microsoft Fabric Data Warehouse** is a unified platform for data, analytics, and AI, designed to store, organize, and manage large volumes of structured and semi-structured data.
- Powered by **Synapse Analytics**, it offers advanced query processing and supports full transactional T-SQL capabilities, similar to an enterprise data warehouse.
- Unlike dedicated SQL pools in Synapse Analytics, Fabric warehouses are built around a single data lake, with all data stored in **Parquet** file format in **Microsoft OneLake**.
- Data is automatically stored in **Delta Parquet format** for both warehouses and lakehouses, enabling seamless integration and analytics.

### Data Ingestion and Loading Concepts
- **Data Ingestion/Extract:** Moving raw data from various sources into a central repository (staging area).
- **Data Loading:** Transferring transformed or processed data into the final storage destination (warehouse tables) for analysis and reporting.
- **Staging Area:**
  - Acts as an abstraction layer to simplify and buffer the load operation.
  - Minimizes performance impact on the main warehouse during data loads.

### Keys and Dimension Management
- **Surrogate Key:**
  - System-generated unique identifier (usually integer or GUID) for each record in a warehouse table.
  - No business meaning; used for internal data integration and consistency.
- **Business Key (Natural Key):**
  - Identifier from the source system with business meaning (e.g., customer ID).
  - Used to uniquely identify records in the source and for data integration.
- Both key types are essential for effective data warehousing and integration.

### Slowly Changing Dimensions (SCD)
- **Slowly Changing Dimensions** track attribute changes over time. Common SCD types:
  - **Type 0:** Attributes never change.
  - **Type 1:** Overwrites existing data; no history kept.
  - **Type 2:** Adds new records for changes, keeping full history (active/inactive flags).
  - **Type 3:** Adds new columns for previous values (limited history).
  - **Type 4:** Uses a separate historical dimension table.
  - **Type 5:** Mini-dimension for large dimensions with changing attributes.
  - **Type 6:** Combines Type 2 and Type 3.
- SCD management is crucial for accurate historical analysis.

### Change Detection and Data Tracking
- Mechanisms for detecting changes in source systems include:
  - **Change Data Capture (CDC)**
  - **Change Tracking**
  - **Database Triggers**
- These features help identify inserts, updates, and deletes for accurate data synchronization.

### Data Pipelines and Integration
- **Data Pipelines:**
  - Cloud-based service for data integration, enabling creation of workflows for data movement and transformation at scale.
  - Supports complex ETL/ELT processes, scheduling, and orchestration.
  - Most pipeline functionality in Fabric is based on **Azure Data Factory**.
- **Pipeline Creation Options:**
  1. **Add pipeline activity:** Launches the editor to build custom pipelines.
  2. **Copy data:** Wizard to copy data from various sources; generates a preconfigured Copy Data task.
  3. **Choose a task to start:** Use predefined templates for common scenarios.
- **Scheduling:** Configure pipeline schedules in the pipeline editor's settings.
- **Recommendation:** Use pipelines for code-free or low-code data workflows, especially for scheduled or multi-source integration.

### Data Loading with the COPY Statement
- The **COPY** statement is the primary method for importing data into a Fabric warehouse.
- Key features:
  - Efficiently ingests data from external Azure storage accounts.
  - Supports specifying file format, error file location, skipping headers, and more.
  - Allows wildcards and multiple file paths for bulk loading (must be from the same storage account/container).
  - Rejected rows can be stored separately for data quality control (ERRORFILE applies to CSV only).
  - Supports authentication via Shared Access Signature (SAS) or Storage Account Key (SAK).
- **Example:**
  ```sql
  COPY INTO my_table
  FROM 'https://myaccount.blob.core.windows.net/myblobcontainer/folder0/*.csv, 
      https://myaccount.blob.core.windows.net/myblobcontainer/folder1/'
  WITH (
      FILE_TYPE = 'CSV',
      CREDENTIAL=(IDENTITY= 'Shared Access Signature', SECRET='<Your_SAS_Token>'),
      FIELDTERMINATOR = '|'
  )
  ```
- The option to use a different storage account for the error file location (REJECTED_ROW_LOCATION) improves error handling and debugging.

### Loading Data from Workspace Assets
- Data can be loaded from other warehouses and lakehouses within the same workspace.
- **Three-part naming** is used to reference data assets accurately.

### Dataflow Gen2 for Data Loading
- **Dataflow Gen2** provides a modern Power Query experience for importing and transforming data.
- Simplifies the process of creating dataflows and reduces the number of steps.
- Dataflows can be used in pipelines to ingest data into lakehouses or warehouses, or to define datasets for Power BI reports.

---

## Query a data warehouse in Microsoft Fabric

### Querying Tools
- **SQL Query Editor:** Write and execute T-SQL queries directly in the Fabric portal.
- **Visual Query Editor:** Design queries using a graphical interface; SQL code is generated automatically.
- **SQL Server Management Studio (SSMS):** Connect using Microsoft Entra ID user principals, user identities, or service principals. SQL authentication is not supported.
- **Other Tools:** Any tool that supports SQL connection strings using ODBC or OLE DB drivers with Microsoft Entra ID can be used to connect and query the data warehouse.

### Query Functions
- **APPROX_COUNT_DISTINCT:**
  - Uses the HyperLogLog algorithm to retrieve an approximate count of distinct values.
  - The result is guaranteed to have a maximum error rate of 2% with 97% probability.

### Additional Notes
- Visual Query Editor provides a no-code, drag-and-drop experience for building queries, making it accessible for users who prefer not to write SQL manually.
- SSMS connections require Microsoft Entra ID authentication; SQL authentication is not available for Fabric data warehouses.

## Monitor a Microsoft Fabric data warehouse

### Capacity and Cost Monitoring
- **Capacity Units (CUs):**
  - Microsoft Fabric resources are provisioned based on the license purchased, which determines the available capacity (pool of resources).
  - The cost of using Fabric is based on **capacity units (CUs)**. Every action in a Fabric resource (such as data reads/writes, queries, and file operations) consumes CUs.
  - Monitoring CU consumption is essential for cost management and planning, especially for data warehouse workloads where queries and file operations to OneLake are significant cost drivers.

- **Fabric Capacity Metrics App:**
  - An administrator can install the **Microsoft Fabric Capacity Metrics app** to monitor capacity utilization in the Fabric environment.
  - The app provides:
    - Trends in capacity usage.
    - Insights into which processes are consuming CUs.
    - Detection of throttling events (when resource limits are reached).
  - Use these insights to optimize workloads and manage costs effectively.

### Monitoring Data Warehouse Activity
- **Dynamic Management Views (DMVs):**
  - DMVs provide real-time information about the current state of the data warehouse.
  - Common DMVs include:
    - `sys.dm_exec_connections`: Returns information about data warehouse connections.
    - `sys.dm_exec_sessions`: Returns information about authenticated sessions.
    - `sys.dm_exec_requests`: Returns information about active requests.
  - Use DMVs to monitor connections, sessions, and running queries for troubleshooting and performance analysis.

### Query Insights and Performance Tuning
- **Query Insights Feature:**
  - Microsoft Fabric data warehouses include a **query insights** feature that provides historical and aggregated information about executed queries.
  - Enables identification of:
    - Frequently used queries.
    - Long-running queries.
    - Query performance trends for tuning and optimization.

- **Query Insights Views:**
  - `queryinsights.exec_requests_history`: Details of each completed SQL query.
  - `queryinsights.long_running_queries`: Details of query execution times for long-running queries.
  - `queryinsights.frequently_run_queries`: Details of frequently executed queries.
  - Use these views to analyze workload patterns and optimize query performance.

---

## Secure a Microsoft Fabric data warehouse

### Role-Based Access Control (RBAC) and Permissions
- **Roles:**
  - Security in Microsoft Fabric data warehouses is managed through built-in roles: **Admin**, **Member**, **Contributor**, and **Viewer**.
  - Each role has specific permissions for managing, editing, or viewing data and resources.
- **Item-Level Permissions:**
  - Granular access can be set at the item level (e.g., specific tables, views, or reports) to restrict or allow access based on business needs.
- **Other Data Protection:**
  - Additional security features include workspace-level settings, integration with Microsoft Entra ID (Azure AD), and audit logging.

### Dynamic Data Masking (DDM)
- **Purpose:**
  - DDM limits data exposure by masking sensitive information in query results for nonprivileged users, without altering the underlying data.
- **Implementation:**
  - Masking is applied at the column level and is easy to configure—no complex coding required.
  - Masking rules are enforced in real time when queries are executed.
- **Masking Types:**
  | Masking Type | Description | Use Case | Limitations | Masking Rule |
  |--------------|-------------|----------|-------------|--------------|
  | Default      | Full masking based on data type | Completely hide data | No information visible | `default()` |
  | Email        | Shows first letter and '.com' | Indicate email field without revealing address | Only for email fields | `email()` |
  | Custom Text  | Shows first/last n chars, pads middle | Partially hide data | Not for numeric/date/time | `partial(prefix, pad, suffix)` |
  | Random       | Replaces numeric/binary with random value | Hide numeric/binary data | Only for numeric/binary | `random(low, high)` |
- **Example:**
  ```sql
  ALTER TABLE Customers
  ALTER COLUMN CreditCardNumber ADD MASKED WITH (FUNCTION = 'partial(0,"XXXX-XXXX-XXXX-",4)');
  ```
- **Benefits:**
  - Data remains intact and secure; only query results are masked for nonprivileged users.

### Row-Level Security (RLS)
- **Purpose:**
  - RLS restricts access to rows in a table based on user identity, group membership, or execution context.
- **How It Works:**
  - A security predicate (function) is associated with a table. It returns true/false to determine row visibility.
  - Enforced automatically by SQL Server, transparent to users.
- **Implementation Steps:**
  1. **Filter Predicate:** Inline table-valued function filters rows for SELECT, UPDATE, DELETE.
  2. **Security Policy:** Associates the predicate with the table.
  - Example: Disabling a policy with `WITH (STATE = OFF)` exposes all rows.
- **Security Considerations:**
  - Be aware of side-channel attacks (e.g., using divide-by-zero errors in WHERE clauses to infer data).
- **Best Practices:**
  - Use a separate schema for predicate functions and policies.
  - Avoid type conversions and excessive joins/recursion in predicates for performance.

### Column-Level Security (CLS) and Views
- **Column-Level Security (CLS):**
  - Restricts access to specific columns, providing granular protection for sensitive data.
  - Permissions are tied to columns and adapt automatically to table changes.
- **Views:**
  - Restrict access by exposing only selected columns or rows.
  - Can provide both column- and row-level security, and transform data (e.g., calculated columns).
- **Comparison Table:**
  | Aspect | Column-Level Security | Views |
  |--------|----------------------|-------|
  | Granularity | Fine-grained, per column | Requires separate views per permission set |
  | Maintenance | Adapts to table changes | Views must be updated if table changes |
  | Performance | Efficient, direct on table | May add overhead for complex/large tables |
  | Transparency | Transparent to user | User queries a different object |
  | Flexibility | Less flexible | Highly flexible, supports transformations |

### Permissions and Dynamic SQL
- **Permissions:**
  - Use `GRANT`, `DENY`, `ALTER`, and `GRANT` statements to manage access to tables, columns, functions, and stored procedures.
- **Dynamic SQL:**
  - Allows programmatic construction and execution of SQL statements within stored procedures.
  - Example:
    ```sql
    CREATE PROCEDURE sp_TopTenRows @tableName NVARCHAR(128)
    AS
    BEGIN
        DECLARE @query NVARCHAR(MAX);
        SET @query = N'SELECT TOP 10 * FROM ' + QUOTENAME(@tableName);
        EXEC sp_executesql @query;
    END;
    ```
  - Use `QUOTENAME` to prevent SQL injection and `sp_executesql` to execute dynamic queries securely.

---

## Get started with data science in Microsoft Fabric

### Data Exploration with Data Wrangler

Microsoft Fabric provides the **Data Wrangler**—an intuitive tool to help you explore and transform your data efficiently.  
- **Descriptive Overview:** Instantly view summary statistics and visualizations of your dataset.
- **Data Quality:** Quickly identify issues such as missing values or outliers.
- **Transformation:** Apply common data cleaning and transformation steps with a user-friendly interface.

### Experiment Tracking

When training machine learning models in notebooks, you can **track your work using experiments** in Microsoft Fabric.
- **Experiment:** A logical container for tracking related model training runs.
- **Run:** Each execution (e.g., training a model with a specific dataset or parameters) is recorded as a separate run within an experiment.
- **Comparison:** Easily compare runs to evaluate which model or configuration performs best.

**Example:**  
If you're building a sales forecasting model, you might try different algorithms or datasets. Each attempt is a new run under the same experiment, allowing you to compare results side by side.

### Model Management and Versioning

After training, you typically want to use your model for predictions (scoring).  
- **MLflow Integration:** When you track models with MLflow, all relevant artifacts (model files, metadata) are stored with the experiment run.
- **Model Registration:** Save your trained model as a registered model in Microsoft Fabric for easy management.
- **Versioning:** Each time you save a model with the same name, a new version is created, enabling robust model lifecycle management.

### Generating Predictions

To use a trained model for batch predictions:
- **PREDICT Function:** Microsoft Fabric provides the `PREDICT` function, which seamlessly integrates with MLflow models.
- **Batch Scoring:** Apply your registered model to new data to generate predictions or insights at scale.

**Key Benefits:**
- Centralized tracking and comparison of experiments.
- Simple model registration and versioning.
- Streamlined batch prediction workflows.

## Get started with Real-Time Intelligence in Microsoft Fabric
Real-Time Intelligence in Microsoft Fabric enables organizations to capture, analyze, and act on streaming data as it arrives. This capability is essential for scenarios where immediate insights or automated actions are required.

### Goals of Real-Time Analytics
- **Continuous Analysis:** Monitor data streams to detect issues or trends as they emerge.
- **Behavioral Insights:** Understand system or component behavior under various conditions to inform future enhancements.
- **Automated Actions:** Trigger alerts or specific actions when events occur or thresholds are exceeded.

### Characteristics of Stream Processing Solutions
- **Unbounded Data Streams:** Data is continuously added, with no defined end.
- **Temporal Data:** Each record typically includes a timestamp indicating when the event occurred.
- **Windowed Aggregation:** Data is often aggregated over time windows (e.g., per minute, per hour) to support real-time metrics.
- **Dual Use:** Results can drive real-time automation/visualization or be persisted for historical analysis—often both.

### Core Components in Microsoft Fabric Real-Time Intelligence

#### 1. Eventstreams
- **Purpose:** Capture, transform, and ingest real-time data from a wide range of streaming sources.
- **Supported Sources:**
  - External services (Azure Event Hubs, IoT Hubs, Kafka, CDC feeds, etc.)
  - Fabric events (workspace changes, OneLake data changes, job events)
  - Sample data for exploration and learning
- **Destinations:**
  - **Eventhouse:** Store and analyze real-time data using KQL.
  - **Lakehouse:** Transform and store real-time events in Delta Lake format for further analytics.
  - **Derived Stream:** Redirect processed data to another eventstream for further transformation.
  - **Fabric Activator:** Automate actions based on stream values.
  - **Custom Endpoint:** Route data to external systems or custom applications.

#### 2. Eventhouses
- **Role:** Store and organize real-time data for analysis.
- **Components:**
  - **KQL Databases:** Optimized for real-time data, hosting tables, functions, materialized views, and shortcuts.
  - **KQL Querysets:** Collections of KQL (Kusto Query Language) queries for analyzing data.
- **KQL:** A powerful, time-based query language used across Azure Data Explorer, Log Analytics, Sentinel, and Fabric.

#### 3. Real-Time Dashboards
- **Purpose:** Visualize real-time insights at a glance by pinning KQL query results to dashboard tiles.
- **Features:**
  - Each tile displays live data from eventhouse tables.
  - Dashboards can be created in a workspace or directly from a KQL queryset.
  - Requires tenant-level feature enablement by an administrator.

#### 4. Activator
- **Function:** Automate actions in response to streaming data events.
- **Core Concepts:**
  - **Events:** Each record in a stream, representing an occurrence at a specific time.
  - **Objects:** Business entities represented by event data (e.g., sensor, sales order).
  - **Properties:** Fields in event data mapped to object attributes (e.g., temperature, total_amount).
  - **Rules:** Define conditions that trigger actions (e.g., send alert if temperature exceeds threshold).
- **Use Cases:** Send notifications, trigger workflows, or execute Fabric jobs based on real-time data.

### Real-Time Hub in Fabric
- **Capabilities:**
  - Discover and connect to real-time data sources.
  - Create and manage eventstreams and Activator alerts.
  - Preview and manage real-time data connections and eventhouses.
  - Build and share real-time dashboards.
  - Endorse and share real-time data resources across the organization.

### Best Practices
- **Combine Real-Time and Historical Analytics:** Persist streaming data for later analysis alongside historical data.
- **Leverage KQL for Advanced Analysis:** Use KQL querysets to extract deep insights from real-time data.
- **Automate Responsively:** Use Activator to trigger timely actions and alerts, reducing manual intervention.
- **Visualize for Impact:** Build dashboards to surface key metrics and trends for rapid decision-making.

**Summary:**
Microsoft Fabric Real-Time Intelligence empowers organizations to act on data as it happens—enabling continuous monitoring, rapid response, and data-driven automation across a wide range of business scenarios.

## Use real-time eventstreams in Microsoft Fabric

Eventstreams in Microsoft Fabric provide a **no-code, visual way to capture, process, and route real-time events** from a variety of sources to multiple destinations. They enable you to design streaming data pipelines that can filter, aggregate, group, and enrich data before it reaches its destination.

### Key Concepts

- **Eventstream Pipeline:** Think of an eventstream as a **conveyor belt** that moves data from sources to destinations, with the ability to transform data along the way.
- **Visual Editor:** Use the **drag-and-drop eventstream editor** to design your pipeline, add sources, destinations, and transformations, and monitor data flow in real time.

---

### Supported Event Sources

Microsoft Fabric eventstreams can ingest data from a wide range of internal and external sources, including:

- **Azure Event Hub**
- **Azure IoT Hub**
- **Azure SQL CDC (Change Data Capture)**
- **PostgreSQL CDC**
- **MySQL CDC**
- **Azure Cosmos DB CDC**
- **Google Cloud Pub/Sub**
- **Amazon Kinesis Data Streams**
- **Confluent Cloud Kafka**
- **Fabric workspace events**
- **Azure Blob Storage events**
- **Custom endpoints**

---

### Eventstream Destinations

You can route processed events to several types of destinations:

- **Eventhouse:** Store real-time event data in a **KQL database** for advanced analysis using Kusto Query Language (KQL). This enables rich reporting and dashboarding.
- **Lakehouse:** Preprocess and store events in **Delta Lake format** within lakehouse tables, supporting data warehousing and analytics.
- **Custom Endpoint:** Send real-time data to external or proprietary applications for immediate consumption or integration with other systems.
- **Derived Stream:** Create a new stream after applying transformations (like filter or manage fields) to the original eventstream. This processed stream can be routed to any supported destination and monitored in the Real-Time hub.
- **Fabric Activator:** Trigger **automated actions** (such as alerts or workflows) based on streaming data values.

---

### Transformations and Windowing

**Transformations** allow you to shape and enrich your streaming data before it reaches its destination. Common transformations include:

- **Filter**
- **Manage Fields**
- **Aggregate**
- **Group By**
- **Union**
- **Expand**
- **Join**

#### Windowing Functions

**Windowing functions** enable operations over time-based segments of streaming data, which is essential for analyzing trends and patterns in real time. They are especially useful for scenarios like sensor monitoring, web analytics, and transaction processing.

- **Group By Transformation Parameters:**
  - **Window Type:** Defines how events are grouped (see below).
  - **Window Duration:** The length of each window.
  - **Window Offset:** (Optional) Shifts the start/end time of the window.
  - **Grouping Key:** One or more columns to group by.

**Types of Windows:**

- **Tumbling Windows:** Fixed, non-overlapping intervals (e.g., every 5 minutes).
- **Sliding Windows:** Fixed, overlapping intervals (e.g., every 5 minutes, sliding every 1 minute).
- **Session Windows:** Variable, non-overlapping intervals based on periods of activity separated by inactivity.
- **Hopping Windows:** Overlapping windows that "hop" forward by a set interval, allowing events to appear in multiple windows.
- **Snapshot Windows:** Group events with the same timestamp. Use `System.Timestamp()` in the `GROUP BY` clause.

---

### Best Practices

- **Design for Flexibility:** Use transformations and windowing to adapt to changing data patterns and business needs.
- **Monitor in Real Time:** Leverage the visual editor to observe data flow and troubleshoot issues as they occur.
- **Automate Actions:** Integrate with Fabric Activator to trigger alerts or workflows based on streaming data.

---

**Summary:**  
Microsoft Fabric eventstreams empower you to build robust, real-time data pipelines—enabling immediate insights, automation, and integration across your analytics ecosystem.

## Work with real-time data in a Microsoft Fabric eventhouse

An **eventhouse** in Microsoft Fabric is a specialized data store designed for efficiently handling large volumes of **time-based event data**. It is optimized for real-time analytics scenarios, enabling organizations to ingest, store, query, and analyze streaming data in near real-time.

### Key Features of an Eventhouse

- **Optimized for Real-Time Data:** Built to manage and analyze continuous streams of events, such as telemetry, logs, or IoT sensor data.
- **KQL Databases:** Each eventhouse contains one or more **KQL (Kusto Query Language) databases** that support:
  - **Tables** for storing event data
  - **Materialized Views** for pre-aggregated, summarized data
  - **Stored Functions** for reusable query logic
  - **Stored Procedures** and other advanced objects
- **OneLake Integration:** You can enable the **OneLake** option for a database or for individual tables, making eventhouse data available across the Microsoft Fabric ecosystem.

---

### Working with KQL Databases

After creating an eventhouse, you can use the default KQL database or create new ones as needed. KQL databases provide a rich set of features for managing and analyzing real-time data.

#### Materialized Views

- **Purpose:** Materialized views provide a **precomputed summary** of data from a source table or another materialized view, improving query performance for common aggregations.
- **Types:**
  - **For New Data Ingestion:** Only summarizes new incoming data.
  - **With Backfill:** Summarizes both historical and new data.

**Example: Create a Materialized View for New Data**
```kql
.create materialized-view TripsByVendor on table Automotive
{
    Automotive
    | summarize trips = count() by vendor_id, pickup_date = format_datetime(pickup_datetime, "yyyy-MM-dd")
}
```

**Example: Create a Materialized View with Backfill**
```kql
.create async materialized-view with (backfill=true)
TripsByVendor on table Automotive
{
    Automotive
    | summarize trips = count() by vendor_id, pickup_date = format_datetime(pickup_datetime, "yyyy-MM-dd")
}
```

#### Stored Functions

- **Purpose:** Stored functions encapsulate reusable query logic, making it easy to apply complex filters or calculations across multiple queries.

**Example: Create a KQL Stored Function**
```kql
.create-or-alter function trips_by_min_passenger_count(num_passengers:long)
{
    Automotive
    | where passenger_count >= num_passengers 
    | project trip_id, pickup_datetime
}
```

- **Usage:** Both materialized views and stored functions can be queried just like regular tables, enabling modular and maintainable analytics workflows.

---

### Best Practices

- **Leverage Materialized Views:** Use materialized views to accelerate queries on large, frequently accessed datasets.
- **Encapsulate Logic in Functions:** Store complex or commonly used logic in KQL functions for reusability and clarity.
- **Enable OneLake Integration:** Make your eventhouse data broadly accessible by enabling OneLake for relevant tables or databases.
- **Monitor and Optimize:** Regularly review query performance and optimize materialized views and functions as data patterns evolve.

---

**Summary:**  
A Microsoft Fabric eventhouse, powered by KQL databases, provides a robust platform for real-time analytics—enabling efficient storage, transformation, and querying of streaming event data at scale.

## Create Real-Time Dashboards with Microsoft Fabric

Real-Time Dashboards in Microsoft Fabric allow you to visualize and monitor streaming data as it arrives, providing actionable insights for rapid decision-making. Below are key concepts, configuration options, and best practices for implementing effective real-time dashboards.

### Authorization Schemes

When connecting a dashboard to its data source, you can specify one of two **authorization schemes**:

- **Pass-through Identity:** The dashboard accesses data using the identity of the user viewing the dashboard. This ensures data security and compliance with user-level permissions.
- **Dashboard Editor's Identity:** The dashboard accesses data using the identity of the user who created (edited) the dashboard. This can simplify access but may expose more data than intended if not managed carefully.

### Dashboard Structure

- A dashboard consists of one or more **tiles**, each displaying the results of a **KQL query**.
- Tiles can be configured for interactivity, filtering, and real-time updates.

### Best Practices for Real-Time Dashboards

#### 1. Clarity and Simplicity
- **Keep dashboards simple and uncluttered.**
- Use clear, descriptive labels for tiles and visuals.
- Organize content using multiple pages for different subject areas or navigation when necessary.

#### 2. Relevance
- **Display only data that is relevant** to the dashboard's purpose and the audience's needs.
- Regularly review dashboard content to ensure continued alignment with business goals.

#### 3. Refresh Rate
- **Set an appropriate refresh rate** to keep data up to date without overloading the system.
- Consult with users to determine acceptable refresh intervals.

#### 4. Accessibility
- **Design dashboards for all users,** including those with viewer permissions.
- Ensure visuals and navigation are accessible and intuitive.

#### 5. Interactivity
- Include features such as **filters, drill-downs, and parameters** to allow users to explore data.
- Elicit regular feedback to ensure dashboards remain valuable and user-friendly.
- As users become more familiar, introduce new features to enhance productivity.
- **Leverage Copilot** where possible to increase productivity and automate insights.

#### 6. Performance
- **Optimize queries and visuals** for fast loading and smooth user experience.
- Use parameters to filter data at the query level, reducing unnecessary data transfer.
- Avoid querying more data than is needed for the visualization.

#### 7. Security
- **Implement robust security measures:**
  - Protect sensitive data in dashboards and underlying data sources.
  - Manage authentication (who can access the system) and authorization (what they can access) carefully.
  - Remember, Fabric is a **Software as a Service (SaaS)** solution—properly manage user roles and permissions.

#### 8. Testing
- **Regularly test dashboards** for functionality and performance.
- Include user-acceptance testing and feedback loops to ensure dashboards meet user needs and expectations.

---

**Summary:**  
A well-designed real-time dashboard in Microsoft Fabric delivers timely, relevant, and actionable insights—empowering users to make informed decisions and respond quickly to changing business conditions.

## Monitor activities in Microsoft Fabric

Monitoring activities in Microsoft Fabric is essential to ensure data delivery, reliability, and performance across your analytics solutions. Below are key activities and tools to monitor, along with best practices and important concepts.

### 1. Data Pipeline Activity
- **Data pipelines** are collections of activities that perform data ingestion, transformation, and loading (ETL) as a unified process.
- **Monitoring Tips:**
  - Track the **success or failure** of jobs and individual pipeline activities.
  - Review **job history** to compare current and past performance, helping to identify when errors were introduced.
  - Investigate errors and failures promptly to maintain data quality and reliability.

### 2. Dataflows
- **Dataflows** provide a low-code interface for ingesting, loading, and transforming data.
- Can be run **manually, on a schedule, or as part of pipeline orchestration**.
- **Monitoring Tips:**
  - Monitor **start/end times, status, duration, and table load activities**.
  - Drill down into specific activities to view error details and troubleshoot issues.

### 3. Semantic Model Refreshes
- A **semantic model** is a visual, ready-for-reporting data model containing transformations, calculations, and relationships.
- **Refreshes** are required when the underlying data or model changes.
- **Monitoring Tips:**
  - Track **refresh status, retries, and failures** to identify transient or persistent issues.
  - Use pipeline-triggered refresh activities for automation and monitoring.

### 4. Spark Jobs, Notebooks, and Lakehouses
- **Notebooks** are used to develop and execute Apache Spark jobs for data loading and transformation in lakehouses.
- **Monitoring Tips:**
  - Monitor **job progress, task execution, resource usage, and Spark logs**.
  - Review logs for errors, bottlenecks, and performance optimization opportunities.

### 5. Microsoft Fabric Eventstreams
- **Eventstreams** ingest and process real-time or streaming data for analytics and routing to destinations.
- **Monitoring Tips:**
  - Track **streaming data ingestion status and performance**.
  - Monitor for data lags, dropped events, or ingestion bottlenecks.

### 6. Monitor Hub
- The **Monitor hub** is the central visualization tool for monitoring activities in Microsoft Fabric.
- **Key Features:**
  - Aggregates data from multiple Fabric items and processes into a single interface.
  - Allows you to view the **status, start/end time, duration, and statistics** for each activity.
  - Supports actions such as **opening activities, retrying, viewing details, and historical runs**.
- **Best Practice:** Use Monitor hub to gain a holistic view of your data integration, transformation, movement, and analysis activities.

### 7. Real-Time Intelligence and Activator
- **Activator** is a tool for triggering actions on streaming data in real time.
- **How It Works:**
  - Define **rules** based on event properties, thresholds, patterns, or Kusto Query Language (KQL) queries.
  - When a rule condition is met, Activator can **alert users, execute Fabric jobs (like pipelines), or trigger Power Automate workflows**.
- **Core Concepts:**
  - **Events:** Each record in a data stream, representing an occurrence at a specific time.
  - **Objects:** Business entities represented by event data (e.g., sales order, sensor).
  - **Properties:** Fields in event data mapped to object attributes (e.g., temperature, total_amount).
  - **Rules:** Conditions that trigger actions based on property values (e.g., send alert if temperature exceeds threshold).

**Summary:**
- Proactive monitoring of these activities ensures data reliability, timely delivery, and rapid troubleshooting in Microsoft Fabric.
- Leverage the Monitor hub and built-in monitoring features to maintain operational excellence across your analytics environment.

## Secure data access in Microsoft Fabric

Securing data access in Microsoft Fabric involves multiple layers and granular controls to ensure only authorized users can access sensitive data. Below is a structured overview of the security model and best practices.

### Security Levels in Fabric
Microsoft Fabric evaluates access through **three sequential security levels**:

1. **Microsoft Entra ID Authentication**
   - Verifies if the user can authenticate with Azure's identity and access management service (**Microsoft Entra ID**).
2. **Fabric Access**
   - Checks if the user has access to the Fabric environment itself.
3. **Data Security**
   - Determines if the user can perform the requested action (e.g., read, write) on a specific table or file.

> **All three levels must be satisfied for access to be granted.**

### Data Security Building Blocks
The third level, **data security**, is highly configurable and consists of several building blocks that can be used individually or together:

#### 1. Workspace Roles
- Assign users to roles (Admin, Member, Contributor, Viewer) that define their access within a workspace.
- Controls broad access to all items within the workspace.

#### 2. Item Permissions
- Permissions can be inherited from workspace roles or set individually by sharing specific items (lakehouses, warehouses, semantic models, etc.).
- Use item permissions to restrict access when workspace roles are too permissive.

#### 3. Compute or Granular Permissions
- Apply fine-grained permissions within a specific compute engine (e.g., SQL Endpoint, semantic model).
- Examples: **Read**, **ReadData**, **ReadAll** permissions on tables or views.

#### 4. OneLake Data Access Controls (Preview)
- Restrict access to specific files or folders in OneLake using **role-based access control (RBAC)**.
- Enables secure, granular access to data at the storage layer.

### Applying Granular Security
When workspace roles or item permissions are not sufficient, use granular security controls such as:
- **Table and Row-Level Security:** Enforce access at the table or row level using SQL analytics endpoints or warehouse features.
- **File and Folder Access:** Restrict access to specific files/folders in OneLake using data access roles (preview).
- **Semantic Model Security:** Apply security at the semantic model level for Power BI and analytics scenarios.

### Summary Table: Access Control Mechanisms
| Security Layer         | Example Controls                | Where Applied                |
|-----------------------|---------------------------------|------------------------------|
| Workspace Roles       | Admin, Member, Contributor      | Workspace                    |
| Item Permissions      | Share, View, Edit               | Individual Fabric items      |
| Granular Permissions  | Read, ReadData, Row-level       | SQL Endpoint, Warehouse      |
| OneLake Data Access   | RBAC for files/folders (preview)| OneLake storage              |

### Best Practices
- **Principle of Least Privilege:** Always grant users the minimum permissions required for their role.
- **Review Inheritance:** Be aware of inherited permissions from workspace roles and override with item permissions as needed.
- **Combine Controls:** Use a combination of workspace, item, and granular permissions to meet complex security requirements.
- **Monitor and Audit:** Regularly review access logs and permissions to ensure compliance and detect unauthorized access.

**In summary:** Microsoft Fabric provides a layered, flexible approach to securing data access, allowing organizations to tailor security to their specific needs—from broad workspace roles to granular file, table, and row-level controls.

## Implement continuous integration and continuous delivery (CI/CD) in Microsoft Fabric

Implementing **CI/CD** in Microsoft Fabric ensures that your analytics solutions are robust, reliable, and delivered efficiently. Below is a structured overview of CI/CD concepts and their application in Fabric.

### What is CI/CD?
- **Continuous Integration (CI):**
  - Developers frequently commit code to a shared branch in a version control system (e.g., Git).
  - Each commit triggers automated builds and tests, catching bugs and integration issues early.
  - **Benefits:** Early conflict detection, faster bug resolution, and improved code quality.
- **Continuous Delivery (CD):**
  - After CI, code is automatically deployed to a staging environment for further automated testing.
  - Ensures that code is always in a deployable state.
- **Continuous Deployment:**
  - Extends CD by automatically releasing updates to production after passing all tests.
  - Enables rapid, reliable delivery of new features and fixes.

### CI/CD in Microsoft Fabric
Managing the lifecycle of Fabric items with CI/CD involves two main parts:

#### 1. Integration (Source Control with Git)
- **Git Integration:**
  - Fabric supports integration with **GitHub** and **Azure DevOps** at the workspace level.
  - Teams collaborate using branches, manage incremental changes, and track code history.
  - When a workspace is connected to Git, you can sync items with a repository branch and view their sync status directly in Fabric.

#### 2. Deployment (Deployment Pipelines)
- **Deployment Pipelines:**
  - Automate the promotion of content through **development**, **test**, and **production** environments.
  - Pipelines ensure content is updated, tested, and regularly refreshed.
  - Can be used in conjunction with Git branches to promote content between environments, even when each environment uses different repositories or branches.

#### 3. Automation (Fabric REST APIs)
- **Fabric REST APIs:**
  - Enable programmatic management and automation of CI/CD processes.
  - APIs are available for both deployment pipelines and Git integration.
  - Use these APIs to automate repetitive tasks, trigger deployments, or integrate with external systems.

### Supported Version Control Systems
- **GitHub**
- **Azure DevOps**

> **Note:** Version control integration is at the workspace level. You can version and manage all items developed within a workspace.

### Best Practices for CI/CD in Fabric
- **Use branches** for feature development and bug fixes to isolate changes.
- **Automate builds and tests** to catch issues early and ensure code quality.
- **Promote content** through deployment pipelines to maintain environment consistency.
- **Monitor sync status** in Fabric to ensure your workspace is up-to-date with the remote repository.
- **Leverage REST APIs** for advanced automation and integration with other tools.

**Summary:**
Microsoft Fabric's CI/CD capabilities—powered by Git integration, deployment pipelines, and REST APIs—enable teams to deliver high-quality analytics solutions efficiently, with confidence in code quality and deployment reliability.

## Administer a Microsoft Fabric environment

### OneLake and Fabric Hierarchy

- **OneLake** is the unified storage layer for all data in Fabric, built on Azure Data Lake Storage (ADLS) Gen2.
  - **Single OneLake per tenant:** Provides a global, hierarchical namespace across users, regions, and clouds.
  - **Tenant:** The top-level organizational boundary, mapped to Microsoft Entra ID, and aligns with the root of OneLake.
  - **Capacity:** Dedicated compute resources assigned to a tenant. Multiple capacities can be associated with a tenant, and are managed via Fabric SKUs or trials.
  - **Domain:** Logical grouping of workspaces (e.g., Sales, Marketing, Finance) for easier management and access control.
  - **Workspace:** A container for Fabric items (data warehouses, pipelines, datasets, reports, dashboards) and access management.
  - **Items:** The core objects you create and manage in Fabric.

### Administrative Roles

- **Fabric Admin (formerly Power BI Admin):** Central role for managing Fabric settings and configurations.
- **Microsoft 365 Admin, Power Platform Admin, Capacity Admin:** Also play key roles in administration.
- **Role-based access:** Controls who can manage, configure, and monitor Fabric resources.

### Admin Portal

- **Web-based portal** for centralized management of Fabric.
  - Manage tenant and capacity settings.
  - Control user, admin, and group access.
  - Access audit logs and monitor usage/performance.
  - **Fabric On/Off Switch:** Enable or disable Fabric at the tenant or capacity level.

### Automation and APIs

- **PowerShell Cmdlets:** Automate common admin tasks (e.g., group management, data source configuration, monitoring).
- **Admin APIs & SDKs:** Programmatically interact with Fabric for automation and integration with other systems.
  - **API:** Set of protocols for communication between applications.
  - **SDK:** Tools and libraries for building integrations and automations.

### Monitoring and Insights

- **Admin Monitoring Workspace:** Provides access to usage and performance insights.
  - Share workspace or specific items with others.
  - Includes the Feature Usage and Adoption dataset/report for analytics.

### License Management

- **User licenses** determine access and functionality within Fabric.
  - Admins allocate and monitor licenses via the Microsoft 365 admin center.
  - Efficient license management helps control costs and ensures compliance.

### Content Sharing and Security

- **Workspace Apps:** Preferred method for distributing content (read-only access).
- **Workspace Access:** For collaboration and development.
- **Least Privilege Principle:** Grant only necessary permissions to users.

### Governance and Endorsement

- **Endorsement:** Mark items as Promoted (workspace-level) or Certified (org-wide, after review).
  - **Promoted:** Visible badge, can be set by workspace contributors/admins.
  - **Certified:** Requires formal review, managed by admins.
- **Scanner API:** Scan Fabric items (warehouses, pipelines, datasets, reports, dashboards) for sensitive data.
- **Metadata Scanning:** Catalog and report on all metadata for governance.
- **Data Lineage:** Track data flow and transformations for impact analysis and compliance.
  - **Lineage View:** Visualize data movement and dependencies in workspaces.
  - **Microsoft Purview Hub (preview):** Centralized governance for Fabric data estate.

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


