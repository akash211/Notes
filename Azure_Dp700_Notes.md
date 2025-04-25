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

## Organize a Fabric lakehouse using medallion architecture design

## Use Apache Spark in Microsoft Fabric

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