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

#### Serverless vs. Provisioned Throughput

**Provisioned Throughput**: Can be distributed to an unlimited number of Azure regions.  
Allows for practically unlimited storage capacity.  
**Serverless**: Limited to running in a single region.  
Allows only up to 50 GB in a container.

#### Autoscale vs. Standard (Manual) Throughput

**Standard (Manual) Throughput**: Has a set provision of Request Units (RUs).  
After reaching the set RUs, responses will be rate limited, causing delays.
**Autoscale**: Will scale up to the maximum RUs and then reach rate limiting response.  
Only requires setting the maximum RUs, with the minimum billed being 10% of the maximum when there are zero requests.  
If the maximum throughput is less than 66% of hours per month, autoscale is more efficient.  
Autoscale is better when usage cannot be predicted.  
Has a minimum acceptable performance and maximum allowed spend.  
Allows for migrating existing containers to and from autoscale, with the ability to change the assigned RUs value later during migration.

### Chapter 5. Move data into and out of Azure Cosmos DB for NoSQL

- **Azure Data Factory (ADF):**
- ADF is a native service to extract data, transform it, and load it across sinks and stores in an entirely serverless fashion. From a data integration perspective, this means you can marshal data from one datastore to another, regardless of the nuances of each, if you can reasonably transform the data between each data paradigm.
- This database is available as a linked service within ADF both as a source of data ingest and as a target(sink) of data output.
- This can be configured using Azure portal or using a Json object. Below is Json format:

 ```json
 {
     "name": "<example-name-of-linked-service>",
     "properties": {
         "type": "CosmosDb",
         "typeProperties": {
             "connectionString": "AccountEndpoint=<cosmos-endpoint>;AccountKey=<cosmos-key>;Database=<cosmos-database>"
         }
     }
 }
 ```

- The database can be configured using service principals and managed identities as well with ADF.
- When reading we must configure database as source and during writing as sink.
- During reading we must give SQL query to get the data and during writing we must give write behaviour.
- **Source JSON configuration:**

  ```json
  {
   "source": {
    "type": "CosmosDbSqlApiSource",
    "query": "SELECT id, categoryId, price, quantity, name FROM products WHERE price > 500",
    "preferredRegions": [
     "East US",
     "West US"
    ]        
   }
  }
  ```

- **Sink JSON configuration:**

  ```json
  "sink": {
   "type": "CosmosDbSqlApiSink",
   "writeBehavior": "upsert"
  }
  ```
  
- **Kafka connector:**
- Apache Kafka is an open-source platform used to stream events in a distributed manner.
- Kafka connect is used to stream data between Kafka and other data systems like Azure Cosmos DB.
- The database can be used as a source or sink.
- connection has 4 properties - Endpoint, MasterKey, DatabaseName, Containers.topicmap (This is using csv format a mapping between containers and Kafka topics)
- Each container should be mapped to a topic. So, if the products container is to be mapped to the prodlistener topic and the customers container to the custlistener topic. In that case, CSV mapping string will be: prodlistener#products,custlistener#customers.
- Write to the database commands:

 ```shell
 kafka-topics --create --zookeeper localhost:2181 --topic prodlistener --replication-factor 1 --partitions 1 
 kafka-console-producer --broker-list localhost:9092 --topic prodlistener
 {"id": "0ac8b014-c3f4-4db0-8a1f-434bab460938", "name": "handlebar", "categoryId": "78148556-4e84-44be-abae-9755dde9c9e3"}
 {"id": "54ba00da-50cf-44d8-b122-1d18bd1db400", "name": "handlebar", "categoryId": "eb642a5e-0c6f-4c83-b96b-bb2903b85e59"}
 {"id": "381dde84-e6c2-4583-b66c-e4a4116f7d6e", "name": "handlebar", "categoryId": "cf8ae707-6d74-4563-831a-06e15a70a0dc"}
 ```

- Read from the database, using below JSON it will be configured and then published to Kafka topic:

 ```json
 {
   "name": "cosmosdb-source-connector",
   "config": {
     "connector.class": "com.azure.cosmos.kafka.connect.source.CosmosDBSourceConnector",
     "tasks.max": "1",
     "key.converter": "org.apache.kafka.connect.json.JsonConverter",
     "value.converter": "org.apache.kafka.connect.json.JsonConverter",
     "connect.cosmos.task.poll.interval": "100",
     "connect.cosmos.connection.endpoint": "<cosmos-endpoint>",
     "connect.cosmos.master.key": "<cosmos-key>",
     "connect.cosmos.databasename": "<cosmos-database>",
     "connect.cosmos.containers.topicmap": "<kafka-topic>#<cosmos-container>",
     "connect.cosmos.offset.useLatest": false,
     "value.converter.schemas.enable": "false",
     "key.converter.schemas.enable": "false"
   }
 }
 ```

- **Using Azure Stream Analytics:**
- Azure Stream Analytics is a real-time event-processing engine designed to process fast streaming data from multiple sources simultaneously. It can aggregate, analyse, transform, and even move data around to other data stores for more profound and further analysis.
- Write now only supports Azure cosmos dB NoSQL.
- Is configured as sink using Account Id(Endpoint), Account Key, Database, Container, Output Alias.
- Container should be already existing.
- To write to cosmos DB, Query results will be processed as JSON output, based on id field.
- If id already exists, then data is updated else inserted.

- **Using Spark Connector:**
- With Azure Synapse Analytics and Azure Synapse Link for Azure Cosmos DB, you can create a cloud-native hybrid transactional and analytical processing (HTAP) to run analytics over your data in Azure Cosmos DB for NoSQL. This connection enables integration over your data pipeline on both ends of your data world, Azure Cosmos DB and Azure Synapse Analytics.
- Using below command Synapse Link can be enabled on account level (or using Portal):

 ```shell
 az cosmosdb create --name <name> --resource-group <resource-group> --enable-analytical-storage true
 ```

- And then on container level also analytical storage should be enabled:

 ```shell
 az cosmosdb sql container create --resource-group <resource-group> --account <account> --database <database> --name <name> --partition-key-path <partition-key-path> --throughput <throughput> --analytical-storage-ttl -1
 ```

- These can be enabled using developer SDKs as well.
- The commands work inside Azure Synapse Analytics workspace.
- For reading from database: Python code can be used or Spark table pointing to database directly can be created.

## Section 3. Connect to Azure Cosmos DB for NoSQL with the SDK

### Chapter 6. Use the Azure Cosmos DB for NoSQL SDK

- Primary library is `Microsoft.Azure.Cosmos` for .NET and hosted at nuget.
- It is open source.
- `dotnet add package Microsoft.Azure.Cosmos` imports latest stable version while, `dotnet add package Microsoft.Azure.Cosmos --version 3.22.1` imports specific version.
- Primary classes are:
  1. Microsoft.Azure.Cosmos.CosmosClient
  2. Microsoft.Azure.Cosmos.Database
  3. Microsoft.Azure.Cosmos.Container  
- First, we must initiate dotnet project using dotnet new console.
- To run C# files using coderunner in VS Code, we must change executerMap of C# from scriptcs to dotnet run.
- We must run dotnet build command.
- To import we use `using Microsoft.Azure.Cosmos`;
- To create CosmosClient class instance, we can use connection string or endpoint+key.
  
  ```C#
  string connectionString = "AccountEndpoint=https­://dp420.documents.azure.com:443/;AccountKey=fDR2ci9QgkdkvERTQ==";
  CosmosClient client = new (connectionString);
  ```

  ```C#
  string endpoint = "https­://dp420.documents.azure.com:443/";
  string key = "fDR2ci9QgkdkvERTQ==";
  CosmosClient client = new (endpoint, key);
  ```

- Each instance of the CosmosClient class is thread-safe, efficiently manage connections, cache address when working in direct mode. So, it is best practice to use single instance for the entire lifecycle of the application otherwise new instance loses the benefit of caching and connection management. So, we must use async/await paradigm. Also, singleton instance is best wayt to manage connections.
- To read properties of account we can use below:
  
  ```C#
  AccountProperties account = await client.ReadAccountAsync();
  ```

  Now from account instance we can take Id(unique name of the account), ReadableRegions, WritableRegions, Consistency and others properties of account.

- Some more code in C#

```C#
  // To connect to existing database
  Database database = client.GetDatabase("databaseName");
  // To create a  new database
  Database database = await client.CreateDatabaseAsync("databaseName");
  // To create new database only if it does not exist already
  Database database = await client.CreateDatabaseIfNotExistsAsync("databaseName");

  // Same for container
  Container container = await database.CreateContainerIfNotExistsAsync("containerName", "/partitionKeyPath", throughput: 400);
```

- To configure the options for the client we can use `CosmosClientOptions` class. If we do not use this then default options will be used like:  
  a. Connect to primary region  
  b. Will use default consistency level  
  c. Will use default throughput  
  d. Will use default max retry count  
  e. will connect to data nodes for requests

  We can also set ConnectionMode propertyy using CosmosClientOptions class. There are two connection modes: Direct and Gateway. By default it is set to Direct.

  ```C#
  CosmosClientOptions options = new CosmosClientOptions(); 
  CosmosClient client = new (connectionString, options); // using connection string
  CosmosClient client = new (endpoint, key, options); // using endpoint and key

  // to connect in Gateway mode
  CosmosClientOptions options = new ()
  {
      ConnectionMode = ConnectionMode.Gateway
  };

  // to change the consistency level
  CosmosClientOptions options = new ()
  {
      ConsistencyLevel = ConsistencyLevel.Eventual
  };

  // to change the preferred regions 
  CosmosClientOptions options = new ()
  {
      ApplicationPreferredRegions  = new List<string> {"East US", "West US"}
  };
 
  ```

### Chapter 7. Configure the Azure Cosmos DB for NoSQL SDK

- We can use Emulator for offline testing. It is available for Windows, Linux and docker.
- Emulator endpoint and key are static.
- We can install emulator directly in Windows and must start the service.
- The SDK has built-in logic to handle transient errors for read requests. For write requests, it does not retry itself but can be written in applications.
- Transient errors are those which have an underlying reason and can resolve themselves.
- Some common Transient errors:
  - 429 – Too many requests
  - 449 – Concurrency error
  - 500 – Unexpected service error
  - 503 – Service Unavailable
- Some client-side errors should not be retried and fixed in apps:
  - 400 – bad request
  - 401 – not authorized
  - 403 – forbidden
  - 404 - not found
- For better performance, built-in iterators work better than LINQ lists.
- For configuration of query – `QueryRequestOptions` is used as below:

```csharp
QueryRequestOptions options = new ()
{
    MaxItemCount = 500,
    MaxConcurrency = 5,
    MaxBufferedItemCount = 5000
};
```

- Default value for `MaxItemCount` is 100 and it means per page how many items will be returned.
- `MaxConcurrency` default is 1 means serially executed. If it is -1 then SDK will manage this and we can specify the number based on physical partition.
- `Microsoft.Azure.Cosmos.Fluent.CosmosClientBuilder` is a builder class that fluently configures a new client instance for injecting custom handlers.
- It can be used for logging using a new class that inherits from `RequestHandler`.

```csharp
using Microsoft.Azure.Cosmos.Fluent;

public class LogHandler : RequestHandler 
{    
    public override async Task<ResponseMessage> SendAsync(RequestMessage request, CancellationToken cancellationToken)
    {
        Console.WriteLine($"[{request.Method.Method}]\t{request.RequestUri}");

        ResponseMessage response = await base.SendAsync(request, cancellationToken);
        
        Console.WriteLine($"[{Convert.ToInt32(response.StatusCode)}]\t{response.StatusCode}");
        
        return response;
    }
}

CosmosClientBuilder builder = new (endpoint, key);
builder.AddCustomHandlers(new LogHandler());
CosmosClient client = builder.Build();
```

## Section 4. Access and manage data with the Azure Cosmos DB for NoSQL SDKs

### Chapter 8. Implement Azure Cosmos DB for NoSQL point operations

- We can use the `container` class to create and manage items in the database.
- to define a product class:

```csharp
public class Product
{
    public string id { get; set; }
    public string name { get; set; }
    public string categoryId { get; set; }
    public double price { get; set; }
    public string[] tags { get; set; }
}
```

- We can also use a different name in C# class and still get correct JSON using below:

```csharp
[JsonProperty(PropertyName = "id")]
public string InternalId { get; set; }
```

- We can create an item using:

```csharp
await container.CreateItemAsync<Product>(saddle);
```

Or, we can use the following to get metadata of operations like request units used:

```csharp
ItemResponse<Product> response = await container.CreateItemAsync<Product>(saddle);
HttpStatusCode status = response.StatusCode;
double requestUnits = response.RequestCharge;
Product item = response.Resource;
```

| Code | Title | Reason |
|------|-------|--------|
| 400  | Bad Request | Something was wrong with the item in the body of the request |
| 403  | Forbidden | Container was likely full |
| 409  | Conflict | Item in the container likely already had a matching ID |
| 413  | RequestEntityTooLarge | Item exceeds the max entity size |
| 429  | TooManyRequests | Current request exceeds the maximum RU/s provisioned for the container |

To catch an error:

```csharp
try
{
    await container.CreateItemAsync<Product>(saddle);
}
catch (CosmosException ex) when (ex.StatusCode == HttpStatusCode.Conflict)
{
    // Add logic to handle conflicting IDs
}
catch (CosmosException ex)
{
    // Add general exception handling logic
}
```

- To read a product:

```csharp
string id = "027D0B9A-F9D9-4C96-8213-C8546C4AAE71";
string categoryId = "26C74104-40BC-4541-8EF5-9892F7F03D72";
PartitionKey partitionKey = new (categoryId);
Product saddle = await container.ReadItemAsync<Product>(id, partitionKey);
string formattedName = $"New Product [${saddle}]";
Console.WriteLine(formattedName);
```

- To update an item:

```csharp
saddle.price = 35.00d;
await container.UpsertItemAsync<Product>(saddle);
saddle.tags = new string[] { "brown", "new", "crisp" };
await container.UpsertItemAsync<Product>(saddle);
```

- To configure TTL for an individual item (only works if TTL is configured at the container level, TTL at the container level should not be NULL):

```csharp
[JsonProperty(PropertyName = "ttl", NullValueHandling = NullValueHandling.Ignore)]
public int? ttl { get; set; }
saddle.ttl = 1000;
await container.UpsertItemAsync<Product>(saddle);
```

- To delete an item:

```csharp
await container.DeleteItemAsync<Product>(id, partitionKey);
```

So, all CRUD operations are happening on the container class.

### Chapter 9. Perform cross-document transactional operations with the Azure Cosmos DB for NoSQL

### Notes on TransactionalBatch in Container Class

The `CreateTransactionalBatch` method in the container class is used to create a `TransactionBatch` instance for batch operations. The following code snippet demonstrates adding two items:

```csharp
Product saddle = new Product("0120", "Worn Saddle", "accessories-used");
Product handlebar = new Product("012A", "Rusty Handlebar", "accessories-used");
PartitionKey partitionKey = new PartitionKey("accessories-used");
TransactionalBatch batch = container.CreateTransactionalBatch(partitionKey)
    .CreateItem<Product>(saddle)
    .CreateItem<Product>(handlebar);
using TransactionalBatchResponse response = await batch.ExecuteAsync();
```

- All items must have the same `partitionKey`; an error will be thrown if the partition key is different.
- The response includes `StatusCode` and `IsSuccessStatusCode` (bool) properties if needed.

### Handling Delay and Concurrent Writes

There may be delays in reading and updating the database, especially when multiple users are writing to it. To address this, each item has an `ETag` (Entity Tag) value that updates whenever the item changes. The code snippet below illustrates this:

```csharp
string categoryId = "9603ca6c-9e28-4a02-9194-51cdb7fea816";
PartitionKey partitionKey = new PartitionKey(categoryId);
ItemResponse<Product> response = await container.ReadItemAsync<Product>("01AC0", partitionKey);
Product product = response.Resource;
string eTag = response.ETag;
product.Price = 50d;
ItemRequestOptions options = new ItemRequestOptions { IfMatchEtag = eTag };
await container.UpsertItemAsync<Product>(product, partitionKey, requestOptions: options);
```

In this scenario, the price will only be updated if the value has not changed.

### Chapter 10. Process bulk data in Azure Cosmos DB for NoSQL

- To enable bulk execution, the following code snippet is used:

```csharp
CosmosClientOptions options = new () 
{ 
    AllowBulkExecution = true 
};
```

- For adding bulk products, the code below can be utilized:
  
```csharp
List<Product> productsToInsert = GetOurProductsFromSomeWhere();

List<Task> concurrentTasks = new List<Task>();

foreach(Product product in productsToInsert)
{
    concurrentTasks.Add(
        container.CreateItemAsync<Product>(
            product, 
            new PartitionKey(product.partitionKeyValue))
    );
}
// Actual operations happen after this line
Task.WhenAll(concurrentTasks);
```

- During bulk operations, more Request Units (RUs) will be consumed, resulting in latency. This method works better for smaller items due to the SDK's automatic creation of batches for optimization, with a maximum of 2 MB (or 100 operations). Larger documents have an inverse effect.

- While some bulk operations can occur without providing a partition key, there will be additional overhead. To minimize this, it is advisable to provide the partition key.

- It is recommended to avoid serialization and deserialization to reduce overhead. Instead, consider using stream variants of common item operations.

## Section 5. Execute queries in Azure Cosmos DB for NoSQL

### Chapter 11. Query the Azure Cosmos DB for NoSQL

- In Cosmos DB SQL, we can use any name as a source and even use an alias. For example, `SELECT * FROM anything`.
- The following query demonstrates how the result JSON can be converted as per the requirement:

    ```sql
    SELECT
        p.name, 
        p.categoryName AS category,
        { "price": p.price } AS scannerData
    FROM 
        products p
    WHERE
        p.price >= 50 AND
        p.price <= 100
    ```

- The keywords `WHERE`, `DISTINCT`, `AS`, `LOWER` work as expected. `CONCAT` works with the pipe (`|`).
- To flatten JSON result into a list, the `VALUE` keyword is used like this: `SELECT VALUE p.name FROM c`.
- To check if a particular property exists for the document or not, we can use `IS_DEFINED`. To check if the property is an array or not, we can use `IS_ARRAY`. To check if the property is NULL or not, we can use `IS_NULL`. Similarly, for `IS_STRING`, `IS_NUMBER`, `IS_BOOLEAN`, `IS_OBJECT`.
- `GetCurrentDateTime()` returns the current datetime and can be used in the query.
- To run a query in C#:

```csharp
QueryDefinition query = new QueryDefinition("SELECT * FROM products p");

public class Product
{
    public string id { get; set; }
    public string name { get; set; }
    public string price { get; set; }
}

using (FeedIterator<Product> feedIterator = this.Container.GetItemQueryIterator<Product>(
    query,
    null,
    new QueryRequestOptions() { }))
{
    while (feedIterator.HasMoreResults)
    {
        foreach(var item in await feedIterator.ReadNextAsync())
        {
            Console.WriteLine($"[{item.id}]\t{item.name,35}\t{item.price,15:C}");
        }
    }
}
```

### Chapter 12. Author complex queries with the Azure Cosmos DB for NoSQL

- Join in this database joins different sections of the same item, unlike relational databases. It is called cross-product queries.

For the given JSON:

```json
{
    "id": "80D3630F-B661-4FD6-A296-CD03BB7A4A0C",
    "categoryId": "629A8F3C-CFB0-4347-8DCC-505A4789876B",
    "categoryName": "Clothing, Vests",
    "sku": "VE-C304-L",
    "name": "Classic Vest, L",
    "description": "A worn brown classic vest that was a trade-in apparel item",
    "price": 32.4,
    "tags": [
        {
            "id": "2CE9DADE-DCAC-436C-9D69-B7C886A01B77",
            "name": "apparel",
            "class": "group"
        },
        {
            "id": "CA170AAD-A5F6-42FF-B115-146FADD87298",
            "name": "worn",
            "class": "trade-in"
        },
        {
            "id": "CA170AAD-A5F6-42FF-B115-146FADD87298",
            "name": "no-damaged",
            "class": "trade-in"
        }
    ]
}
```

We can use:

```sql
SELECT 
    p.id,
    p.name,
    t.name AS tag
FROM 
    products p
JOIN
    t IN p.tags
```

to get:

```json
[
    {
        "id": "80D3630F-B661-4FD6-A296-CD03BB7A4A0C",
        "name": "Classic Vest, L",
        "tag": "apparel"
    },
    {
        "id": "80D3630F-B661-4FD6-A296-CD03BB7A4A0C",
        "name": "Classic Vest, L",
        "tag": "worn"
    },
    {
        "id": "80D3630F-B661-4FD6-A296-CD03BB7A4A0C",
        "name": "Classic Vest, L",
        "tag": "no-damaged"
    }
]
```

- We can use a subquery to filter in the same way as we do in relational databases.
- Variables can also be used in queries:

```csharp
string sql = "SELECT p.name, t.name AS tag FROM products p JOIN t IN p.tags WHERE p.price >= @lower AND p.price <= @upper"
QueryDefinition query = new QueryDefinition(sql)
    .WithParameter("@lower", 500)
    .WithParameter("@upper", 1000);
```

- We can paginate the results using a feed generator. Below is a sample code:

```csharp
string sql = "SELECT * FROM products WHERE p.price > 500";
QueryDefinition query = new QueryDefinition(sql);
QueryRequestOptions options = new QueryRequestOptions
{
    MaxItemCount = 100
};

FeedIterator<Product> iterator = container.GetItemQueryIterator<Product>(query, requestOptions: options);
while(iterator.HasMoreResults)
{
    foreach(Product product in await iterator.ReadNextAsync())
    {
        // Handle individual items
    }
}
```

## Section 6. Define and implement an indexing strategy for Azure Cosmos DB for NoSQL

### Chapter 13. Define indexes in Azure Cosmos DB for NoSQL

- By default, Cosmos DB automatically indexes all paths of items stored using the NoSQL API.

- **Default Indexing Policy Settings:**
    1. The inverted index is updated for all create, update, or delete operations.
    2. All properties for every item are indexed.
    3. Range indexes are used for all strings or numbers.

- **Indexing Policy Components: Default Values**
    1. Indexing Mode: Consistent
    2. Automatic: Enabled
    3. Included Paths: All (*)
    4. Excluded Paths: _etag property path

- **JSON Representation:**

    ```json
    {
      "indexingMode": "consistent",
      "automatic": true,
      "includedPaths": [
        {
          "path": "/*"
        }
      ],
      "excludedPaths": [
        {
          "path": "/\"_etag\"/?"
        }
      ]
    }
    ```

- **Indexing Mode:**
  - Consistent or None
    - None disables indexing completely, useful for bulk operations. Indexing can be disabled for bulk operations and re-enabled later.

- **Path Patterns:**
  - *: Any property after
  - []: Includes array
  - ?: Scalar values
  - Examples:
    - /*: All properties
    - ?name/?: Scalar value of the name property
    - /category/*: All properties under the category property
    - /tags/[]/name/?: All scalar names inside tags

- **Path Conflict Resolution:**
  - Most precise path takes precedence
  - /* is the root path and must be part of included or excluded paths

### Chapter 14. Customize indexes in Azure Cosmos DB for NoSQL

```csharp
// Customize Indexing Policy for a Container
IndexingPolicy policy = new IndexingPolicy()
{
    IndexingMode = IndexingMode.Consistent,
    Automatic = true
};
policy.ExcludedPaths.Add(
    new ExcludedPath { Path = "/*" }
);
policy.IncludedPaths.Add(
    new IncludedPath { Path = "/name/?" }
);
policy.IncludedPaths.Add(
    new IncludedPath { Path = "/categoryName/?" }
);
ContainerProperties options = new ContainerProperties()
{
    Id = "products",
    PartitionKeyPath = "/categoryId",
    IndexingPolicy = policy
};
Container container = await database.CreateContainerIfNotExistsAsync(options, throughput: 400);
```

- Index policy is set for containers.

- If a query contains multiple filters, a composite index can be created to make it faster. This is beneficial for queries that order the results using multiple properties. The properties in the composite index should have the same order as the filter or order in the query. Multiple composite indexes can be defined.

Below JSON adds "name" in ascending order and "price" in descending order:

```json
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [
    {
      "path": "/*"
    }
  ],
  "excludedPaths": [
    {
      "path": "/_etag/?"
    }
  ],
  "compositeIndexes": [
    [
      {
        "path": "/name",
        "order": "ascending"
      },
      {
        "path": "/price",
        "order": "descending"
      }
    ]
  ]
}
```

- Usually, it is a better practice to include all paths and only exclude specific paths.

## Section 7. Integrate Azure Cosmos DB for NoSQL with Azure services

### Chapter 15. Consume an Azure Cosmos DB for NoSQL change feed using the SDK

- The .NET SDK includes a change feed processor with 4 components:
    1. Monitored Container
    2. Lease Container
    3. Host (client app)
    4. Delegate (Code in client app that implements business logic for changes)

- **Delegate** is a variable that references a method with a specific parameter list and return type. It is of type ChangesHandler and includes two parameters: a read-only list of changes and a cancellation token.

    ```csharpl̥
    static async Task HandleChangesAsync(
        IReadOnlyCollection<Product> changes,
        CancellationToken cancellationToken
    ) 
    {
        // Do something with the batch of changes
    }
    ChangesHandler<Product> changeHandlerDelegate = HandleChangesAsync;
    ```

    Using a loop and an anonymous function, it can be written as:

    ```csharp
    ChangesHandler<Product> changeHandlerDelegate = async (
        IReadOnlyCollection<Product> changes,
        CancellationToken cancellationToken
    ) => {
        foreach(Product product in changes)
        {
            await Console.Out.WriteLineAsync($"Detected Operation:\t[{product.id}]\t{product.name}");
            // Do something with each change
        }
    };
    ```

- The change feed processor is created by:
    1. Creating a processor builder from the monitored container variable.
    2. Using the builder to build out the processor by specifying the delegate, processor name, lease container, and host instance name.
    3. Starting the processor.

- To create the source container and lease container instances, use the `ChangeFeedProcessorBuilder` method from the container instance:

    ```csharp
    Container sourceContainer = client.GetContainer("cosmicworks", "products");

    Container leaseContainer = client.GetContainer("cosmicworks", "productslease");
    var builder = sourceContainer.GetChangeFeedProcessorBuilder<Product>(
        processorName: "productItemProcessor",
        onChangesDelegate: changeHandlerDelegate
    );
    ChangeFeedProcessor processor = builder
        .WithInstanceName("desktopApplication")
        .WithLeaseContainer(leaseContainer)
        .Build();
    await processor.StartAsync();

    // Wait while processor handles items

    await processor.StopAsync();
    ```

    The `ChangeFeedProcessor` has many methods like `WithInstanceName`, `WithStartTime`, `WithLeaseContainer`, `WithErrorNotification`, `WithMaxItems`, `WithPollInterval`.

- The **Change Feed Processor** functions as a time-based pointer. It moves forward in time across the change feed and sends batches of changes to the delegate.

- The **Change Feed Processor** can be constrained by the resources of the host container. In such cases, scaling out can be done across multiple hosts, all reading from the change feed concurrently. Identifying scaling out requires an estimator. For the estimator, a delegate is required using `ChangesEstimationHandler`.

    ```csharp
    ChangesEstimationHandler changeEstimationDelegate = async (
        long estimation, 
        CancellationToken cancellationToken
    ) => {
        // Do something with the estimation
    };
    ChangeFeedProcessor estimator = sourceContainer.GetChangeFeedEstimatorBuilder(
        processorName: "productItemEstimator",
        estimationDelegate: changeEstimationDelegate)
        .WithLeaseContainer(leaseContainer)
        .Build();
    ```

### Chapter 16. Handle events with Azure Functions and Azure Cosmos DB for NoSQL change feed

- **Cosmos DB** supports Trigger, Input Binding, and Output Binding with Azure Functions.
- To bind functions with Cosmos DB, first, an app setting is created in the function instance with the Cosmos DB connection string.

## Trigger for Cosmos DB

- Trigger for Cosmos DB requires the following code snippet:

```json
{
  "type": "cosmosDBTrigger",
  "name": "changes",
  "direction": "in",
  "connectionStringSetting": "cosmosdbsqlconnstr",
  "databaseName": "cosmicworks",
  "collectionName": "products",
  "leaseCollectionName": "productslease",
  "createLeaseCollectionIfNotExists": false
}
```

- Input binding can be of two types: point and SQL query binding. Below are the code snippets for both types:
- Point Binding

```json
{
  "type": "cosmosDB",
  "name": "item",
  "direction": "in",
  "connectionStringSetting": "cosmosdbsqlconnstr",
  "databaseName": "cosmicworks",
  "collectionName": "products",
  "id": "91AA100C-D092-4190-92A7-7C02410F04EA",
  "partitionKey": "F3FBB167-11D8-41E4-84B4-5AAA92B1E737"
}
```

- SQL Query Binding

```json
{
  "type": "cosmosDB",
  "name": "items",
  "direction": "in",
  "connectionStringSetting": "cosmosdbsqlconnstr",
  "databaseName": "cosmicworks",
  "collectionName": "products",
  "sqlQuery": "SELECT p.id, p.name, p.categoryId FROM products p WHERE p.price > 500"
}
```

- Output binding code is as follows:

```json
{
  "type": "cosmosDB",
  "name": "output",
  "direction": "out",
  "connectionStringSetting": "cosmosdbsqlconnstr",
  "databaseName": "cosmicworks",
  "collectionName": "products"
}
```

### Chapter 17. Search Azure Cosmos DB for NoSQL data with Azure Cognitive Search

- Azure Cognitive Search instance has **Indexes**, **Indexers**, and **Data Sources**. Indexes contain searchable JSON documents, Indexers crawl data from various sources and insert them into indexes.
- Cosmos DB container can be configured as a data source. An SQL query can be used as an Indexer to get data.
- Index contains a name, key, and optional features. Indexer contains a name and schedule.
- Two policies, one for change detection and one for deletion, must be created.
- Customizing Indexes:
  - **Retrievable**: Configures the field to be projected in search result sets
  - **Filterable**: Accepts OData-style filtering on the field
  - **Sortable**: Enables sorting using the field
  - **Facetable**: Allows the field to be dynamically aggregated and grouped
  - **Searchable**: Allows search queries to match terms in the field
- Stored procedures are scoped to a single partition.
- All operations within a stored procedure should be completed within the server request timeout duration.
- A helper Boolean value is returned to indicate if all operations were completed. If not, a pointer can be returned to start the next stored procedure from that point.
- Example of a function:

```javascript
function createProduct(item) {
    var context = getContext();
    var container = context.getCollection();
    var accepted = container.createDocument(
        container.getSelfLink(),
        item,
        (error, newItem) => {
            if (error) throw error;
            context.getResponse().setBody(newItem);
        }
    );
    if (!accepted) return;
}
```

- In SDK, the **Scripts** class has all tools to create User-Defined Functions (UDFs), Triggers, and stored procedures.
- Pre-triggers run before an operation and cannot have any input parameters. They are usually used for validating or filling missing properties.
- Post-triggers run after an operation has completed and can have input parameters.
- All items within a post-trigger should have the same logical partition key as the original item that was the source of the trigger.
- Triggers are not automatically triggered but need to be specified for each operation.
- **Consistency Models**: Strong, Bounded Staleness, Session, Consistent Prefix, Eventual
  - **Session**: Within the SDK instance and uses a session token. Outside the instance, it changes to Consistent Prefix or Eventual.
  - **Bounded Staleness**: Read can be some version or some time interval behind writes. In a region, it gives strong consistency.
  - **Consistent Prefix**: There will be a delay, but the reads will be in write order.
  - **Eventual**: Lowest write latency, highest availability, and highest read scalability.
- Each Cosmos DB account has a default consistency of Session.
- Using SDK, for each operation, consistency can be relaxed but cannot be made stronger.
- Session token can be pulled from a client and used on another to preserve the session between multiple clients.
- In a multi-region write case, strong consistency is not supported.
- Last write wins in multi-region write Cosmos DB. It uses the `_ts` property, but it can be configured to use something else. This conflict resolution policy is set on new containers only.
- To create a custom stored procedure for conflict resolution, we need:

```javascript
function <function-name>(incomingItem, existingItem, isTombstone, conflictingItems)
```

- Conflicts can be written to the conflicts feed if no stored procedure is created for conflict resolution.
- By default, automatic failover to a different region is not enabled.

## Section 8. Implement a data modeling and partitioning strategy for Azure Cosmos DB for NoSQL

### Chapter 18. Implement a non-relational data model

- **Scalability**:
  - Cosmos DB is horizontally scalable, adding more servers or nodes as required. These nodes are known as physical partitions in Cosmos DB.
  - Data is routed and accessed using a partition key, which is a virtual partition. This allows the database to grow to an unlimited size without impacting response time.
  - As partitions are added, more compute resources are also added to maintain performance.

- **NoSQL Database Characteristics**:
  - NoSQL databases lack foreign keys, constraints, and enforced relationships. Relational databases enforce ACID using locks and latches, impacting concurrency, latency, and availability.
  - Today, with low storage and memory costs, optimization for storage efficiency is less critical. NoSQL databases are compute-efficient, denormalizing data and violating traditional relational database rules.

- **Data Modeling**:
  - **Embedding Data**:
    - Combining related entity tables into a single table (container).
    - Criteria for embedding data include related data that is frequently read or updated together and has a 1:1 or 1:Few relationship.

  - **Referencing Data Separately**:
    - Criteria for separate documents:
      1. Data with properties accessed and updated at different frequencies.
      2. Unbounded 1:Many or Many:Many relationships where document size exceeds 2 MB.

- **Partitioning**:
  - Cosmos DB uses partition keys to route data to physical partitions, with a maximum size of 50 GB and throughput of 10,000 RU/s per partition key.
  - Logical partitions have a maximum size of 20 GB, with multiple logical partitions in a physical partition.
  - To avoid hot partitions, select a partition key that evenly distributes data, especially in large containers with multiple physical partitions.

- **Partition Key Selection**:
  - For write-heavy databases, choose a partition key with high cardinality (many unique values).
  - For read-heavy databases, use where filters to access only one or a few partitions.

- **Cross-Partition Queries**:
  - Queries using non-partition key filters scan all physical partitions, resulting in slower and costlier queries for large containers.

- **Scalability Considerations**:
  - Millions of logical partitions are manageable within Azure Cosmos DB.

Note: Ensure proper partition key selection, data modeling, and partitioning strategies to optimize performance and scalability in Azure Cosmos DB.

### Chapter 19. Design a data partitioning strategy

- **Denormalization**:
  - Denormalizing data involves duplicating information in multiple containers to optimize query performance.
  - For example, having category information duplicated in both the category and product containers reduces the Request Units (RU) required for queries that involve category names with products.

- **Referential Integrity**:
  - To maintain referential integrity in denormalized data, use the Change Feed feature.
  - Change Feed is an API available in every container that streams data changes. It triggers events for changed data, allowing developers to respond accordingly.
  - Each container requiring referential integrity will have its own Change Feed, ensuring data consistency across related entities.

- **Data Storage in the Same Container**:
  - Data from different entities can be stored in the same container if they share the same partition key.
  - Use an additional 'type' property to differentiate between the types of data stored in the container.

- **De-normalized Aggregates**:
  - De-normalized aggregates involve storing aggregated data in the same container with the raw data.
  - This optimization is beneficial for frequently accessed aggregates that are derived from multiple entities.
  - Aggregates can be updated using stored procedures or transactional batches, ensuring consistency and efficiency.

- **Limitations**:
  - Joins do not work across different documents like categories and orders in Azure Cosmos DB.
  - To address this limitation, consider de-normalizing aggregates and storing related data together based on the partition key.

Note: By following these denormalization and data modeling best practices, developers can optimize query performance and maintain data consistency in Azure Cosmos DB.

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
