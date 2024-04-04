# NOTES FOR AZURE CosmosDB Service focusing specially on DP420 Certification

Most of the notes is taken from [Microsoft learn](https://learn.microsoft.com/en-us/training/courses/dp-420t00).

## Section 1. Get started with Azure Cosmos DB for NoSQL

### Chapter 1. Introduction to Azure Cosmos DB for NoSQL

NoSQL databases are not defined by a specific formal definition, rather they share common characteristics

- A nonrelational data store.
- Being designed to scale out.
- Not enforcing a specific schema.

Generally, NoSQL databases don't enforce relational constraints or put locks on data, making writes fast. Also, they're often designed to horizontally scale via sharding or partitioning, which allows them to maintain high-performance regardless of size.

There are 4 broad categories of NoSQL databses:  
a. Documents  
b. Key-Value  
c. Column-Family  
d. Graph

Some points regarding CosmosDB:

- This database provides single digit millisecond response times and 99.999% availability.
- A partition key has two components: partition key path and the partition key value.
- The partition key should only have strings. numbers should be converted into string.
- The partition key should have high cardinality.
- Max size supported for partition values is 2048 bytes if large partition keys are enabled, other max size can be 101 bytes. During creation of the container, we get this option in the advanced tab.
- Once you select your partition key, it isn't possible to change it in-place. If you need to change your partition key, you should move your data to a new container with your new desired partition key.

### Chapter 2. Try Azure Cosmos DB for NoSQL

**JSON**:

- JavaScript Object Notation (JSON) is an open standard file format, and data interchange format, that uses human-readable text to store and transmit data objects consisting of attribute–value pairs and array data types (or any other serializable value).
- JSON is a language-independent data format with well-defined data types and near universal support across a diverse range of services and programing languages.  
- JSON, a lightweight data format, is highly compatible with the object notation of JavaScript.
- Azure Cosmos DB supports JSON natively, ensuring fast and predictable performance due to atomic write operations on JSON documents.
  
Below is an example of JSON document.  

  ```JSON
          {
      "id": "0012D555-C7DE",
      "type": "customer",
      "fullName": "Franklin Ye",
      "title": null,
      "emailAddress": "fye@cosmic.works",
      "creationDate": "2014-02-05",
      "addresses": [
        {
          "addressLine": "1796 Westbury Drive",
          "cityStateZip": "Melton, VIC 3337 AU"
        },
        {
          "addressLine": "9505 Hargate Court",
          "cityStateZip": "Bellflower, CA 90706 US"
        }
      ],
      "password": {
        "hash": "GQF7qjEgMk=",
        "salt": "12C0F5A5"
      },
      "salesOrderCount": 2
    }
  ```

An Azure Cosmos DB for NoSQL account is composed of a basic hierarchy of resources that include:  

a. **An account**:

- Each tenant of the Azure Cosmos DB service is created by provisioning a database account.
- Accounts are the fundamental units of distribution and high availability.
- At the account level, you can configure the region[s] for your data in Azure Cosmos DB for NoSQL.
- Accounts also contain the globally unique DNS name used for API requests.

b. **One or more databases**:

- A database is a logical unit of management for containers in Azure Cosmos DB for NoSQL.
- An Azure Cosmos DB database manages users, permissions, and containers.
- Within the database, you can find one or more containers.
- You can also elect to provision throughput for your data here at the database level.

c. **One or more containers**:

- Containers are the fundamental unit of scalability in Azure Cosmos DB for NoSQL.
- Typically, you provision throughput at the container level.
- Azure Cosmos DB for NoSQL will automatically and transparently partition the data in a container.
- You can also optionally configure an indexing policy or a default time-to-live value at the container level.
- Containers can also store JavaScript-based stored procedures, triggers, and user-defined functions (UDFs).

d. **Many items**:

- An Azure Cosmos DB for NoSQL resource container is a schema-agnostic container of arbitrary user-generated JSON items.
- The NoSQL API for Azure Cosmos DB stores individual documents in JSON format as items within the container.

When creating a new account in the Azure portal, you must first select an API for your workload.  
The API selection cannot be changed after the account is created.  
Below is the list of APIs available:  
  a. Azure CosmosDB for NoSQL  
  b. Azure CosmosDB for MongoDB  
  c. Azure CosmosDB for Apache Cassandra  
  d. Azure CosmosDB for Table  
  e. Azure CosmosDB for Apache Gremlin  
  f. Azure CosmosDB for PostGreSQL

  **We can see RUs used in data explorer in query stat tab beside query result.**

## Section 2. Plan and implement Azure Cosmos DB for NoSQL

### Chapter 3. Plan Resource Requirements

Each container is a unit of scalability for both throughput and storage.

#### Throughput and Storage

- When configuring Azure Cosmos DB, you can provision throughput at either or both the database and container levels.
- Throuput measure is done using Rus (Request Units) which is combinaiton of memory, CPU and IOPS (Input Output Per second).
- There can be mixedithroughput provisioning where some containers have dedicated RUs and rest containers in the database have shared RUs. once a container is defined as shared it cannot be changed to dedicated and vice versa.
- Each request including Reads, Writes, Queries, Stored procedures take some RUs.
- When you create a database or container in Azure Cosmos DB, you can provision request units in an increment of request units per second (or RU/s for short). You cannot provision less than 400 RU/s, and they are provisioned in increments of 100.
- Storage is also billed per GB per month.
- Azure Cosmos DB only charges for storage you directly consume in real time, and you don't have to pre-reserve storage in advance. In high-write scenarios, TTL values can be used to save on data storage costs in Azure Cosmos DB.

#### TTL

- Azure Cosmos DB allows you to set the length of time documents live in the database before being automatically purged.
- It is called Time to Live and measured in seconds.
- It is et at the container level and can be overriden on a per item basis.
- The maximum TTL value is 2147483647.
- TTL expiration is a background activity and consumes some RUs.
- TTL for container is configured using `DefaultTimeToLive` property of container's JSON object.
- If Item TTL is defined but container TTL is not defined then item will not expire.
- Container TTL can be defined as -1, which means by default items will not expire. Now setting item TTL will work.
  
### Chapter 4. Configure Azure Cosmos DB for NoSQL database and containers

### Chapter 5. Move data into and out of Azure Cosmos DB for NoSQL

## Section 3. Connect to Azure Cosmos DB for NoSQL with the SDK

### Chapter 6. Use the Azure Cosmos DB for NoSQL SDK

### Chapter 7. Configure the Azure Cosmos DB for NoSQL SDK

## Section 4. Access and manage data with the Azure Cosmos DB for NoSQL SDKs

### Chapter 8. Implement Azure Cosmos DB for NoSQL point operations

### Chapter 9. Perform cross-document transactional operations with the Azure Cosmos DB for NoSQL

### Chapter 10. Process bulk data in Azure Cosmos DB for NoSQL

## Section 5. Execute queries in Azure Cosmos DB for NoSQL

### Chapter 11. Query the Azure Cosmos DB for NoSQL

### Chapter 12. Author complex queries with the Azure Cosmos DB for NoSQL

## Section 6. Define and implement an indexing strategy for Azure Cosmos DB for NoSQL

### Chapter 13. Define indexes in Azure Cosmos DB for NoSQL

### Chapter 14. Customize indexes in Azure Cosmos DB for NoSQL

## Section 7. Integrate Azure Cosmos DB for NoSQL with Azure services

### Chapter 15. Consume an Azure Cosmos DB for NoSQL change feed using the SDK

### Chapter 16. Handle events with Azure Functions and Azure Cosmos DB for NoSQL change feed

### Chapter 17. Search Azure Cosmos DB for NoSQL data with Azure Cognitive Search

## Section 8. Implement a data modeling and partitioning strategy for Azure Cosmos DB for NoSQL

### Chapter 18. Implement a non-relational data model

### Chapter 19. Design a data partitioning strategy

## Section 9. Design and implement a replication strategy for Azure Cosmos DB for NoSQL

### Chapter 20. Configure replication and manage failovers in Azure Cosmos DB

### Chapter 21. Use consistency models in Azure Cosmos DB for NoSQL

### Chapter 22. Configure multi-region write in Azure Cosmos DB for NoSQL

## Section 10. Optimize query and operation performance in Azure Cosmos DB for NoSQL

### Chapter 23. Customize an indexing policy in Azure Cosmos DB for NoSQL

### Chapter 24. Measure index performance in Azure Cosmos DB for NoSQL

### Chapter 25. Implement integrated cache in Azure Cosmos DB for NoSQL

## Section 11. Monitor and troubleshoot an Azure Cosmos DB for NoSQL solution

### Chapter 26. Measure performance in Azure Cosmos DB for NoSQL

### Chapter 27. Monitor responses and events in Azure Cosmos DB for NoSQL

### Chapter 28. Implement backup and restore for Azure Cosmos DB for NoSQL

### Chapter 29. Implement security in Azure Cosmos DB for NoSQL

## Section 12. Manage an Azure Cosmos DB for NoSQL solution using DevOps practices

### Chapter 30. Write management scripts for Azure Cosmos DB for NoSQL

### Chapter 31. Create resource template for Azure Cosmos DB for NoSQL

## Section 13. Create server-side programming constructs in Azure Cosmos DB for NoSQL

### Chapter 32. Build multi-item transactions with the Azure Cosmos DB for NoSQL

### Chapter 33. Expand query and transaction functionality in Azure Cosmos DB for NoSQL
