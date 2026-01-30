# ckd_prediction.py
# Predicting Chronic Kidney Disease using PySpark Decision Tree

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when, length
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------------------
# 1. Create Spark Session
# ---------------------------
spark = SparkSession.builder \
    .appName("CKD Prediction") \
    .getOrCreate()

# ---------------------------
# 2. Load Dataset
# ---------------------------
data = spark.read.csv(
    "data/kidney_disease_final_clean.csv",
    header=True,
    inferSchema=True
)

# ---------------------------
# 3. Data Cleaning
# ---------------------------
data = data.drop("id")

data = data.withColumn("pcv", regexp_replace(col("pcv"), "[^0-9]", ""))
data = data.withColumn("wc", regexp_replace(col("wc"), "[^0-9]", ""))
data = data.withColumn("rc", regexp_replace(col("rc"), "[^0-9.]", ""))

data = data.withColumn("pcv", when(length(col("pcv")) == 0, None).otherwise(col("pcv")))
data = data.withColumn("wc", when(length(col("wc")) == 0, None).otherwise(col("wc")))
data = data.withColumn("rc", when(length(col("rc")) == 0, None).otherwise(col("rc")))

data = data.na.drop(subset=["pcv", "wc", "rc"])

data = data.withColumn("pcv", col("pcv").cast("int"))
data = data.withColumn("wc", col("wc").cast("int"))
data = data.withColumn("rc", col("rc").cast("float"))

# ---------------------------
# 4. Feature Engineering
# ---------------------------
categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
label_col = "classification"

numeric_cols = [c for c in data.columns if c not in categorical_cols + [label_col]]

indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

label_indexer = StringIndexer(
    inputCol=label_col,
    outputCol="label",
    handleInvalid="keep"
)

assembler_inputs = [c + "_idx" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# ---------------------------
# 5. Model Training
# ---------------------------
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=indexers + [label_indexer, assembler, dt])

train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# ---------------------------
# 6. Model Evaluation
# ---------------------------
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_eval.evaluate(predictions)
f1_score = f1_eval.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# ---------------------------
# 7. Stop Spark Session
# ---------------------------
spark.stop()
