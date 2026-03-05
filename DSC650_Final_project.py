from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase
import time

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("SupplyChainMLPrediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load supply chain data from Hive
df = spark.sql("""
SELECT 
SKU,
`Product type`,
Price,
Availability,
`Stock levels`,
`Order quantities`,
`Number of products sold`,
`Revenue generated`
FROM supply_chain_data
""")

# Step 3: Handle null values
df = df.na.drop()

# Step 4: Assemble ML features
assembler = VectorAssembler(
    inputCols=[
        "Price",
        "Availability",
        "Stock levels",
        "Order quantities",
        "Number of products sold"
    ],
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(df).select(
    "features",
    "`Revenue generated`",
    "SKU",
    "`Product type`",
    "Price",
    "Availability",
    "`Stock levels`",
    "`Number of products sold`"
)

assembled_df = assembled_df.withColumnRenamed("Revenue generated", "label")

# Step 5: Train/Test split
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Train Linear Regression model
lr = LinearRegression(labelCol="label")
model = lr.fit(train_data)

# Step 7: Evaluate model
results = model.evaluate(test_data)

rmse = results.rootMeanSquaredError
r2 = results.r2

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# ----------------------------------------
# Prepare data to store in HBase
# ----------------------------------------

row_key = "model_run_" + str(int(time.time()))

data = [
    (row_key, "metrics:rmse", str(rmse)),
    (row_key, "metrics:r2", str(r2))
]

# Take a few supply chain rows to store
sample_rows = df.limit(5).collect()

for row in sample_rows:
    key = row["SKU"]

    data.extend([
        (key, "product_info:product_type", str(row["Product type"])),
        (key, "product_info:price", str(row["Price"])),

        (key, "inventory:availability", str(row["Availability"])),
        (key, "inventory:stock_levels", str(row["Stock levels"])),

        (key, "sales:products_sold", str(row["Number of products sold"])),
        (key, "sales:revenue_generated", str(row["Revenue generated"]))
    ])

# ----------------------------------------
# Function to write to HBase
# ----------------------------------------

def write_to_hbase_partition(partition):

    connection = happybase.Connection('master')
    connection.open()

    table = connection.table('supply_chain_ml_data')

    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})

    connection.close()

# Parallelize data
rdd = spark.sparkContext.parallelize(data)

# Write to HBase
rdd.foreachPartition(write_to_hbase_partition)

# Stop Spark
spark.stop()