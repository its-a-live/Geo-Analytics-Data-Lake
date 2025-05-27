import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import sin, cos, asin, sqrt, col, pow, lit, row_number

avg_radius = 6371


def get_city_dict(city_dict_path: str, spark: SparkSession) -> pyspark.sql.DataFrame:
    city_df = spark.read.csv(city_dict_path, sep=';', header=True)

    cities = city_df \
        .withColumnRenamed("id", "city_id") \
        .withColumnRenamed("lat", "city_lat") \
        .withColumnRenamed("lng", "city_lon") \
        .withColumnRenamed("timezone", "city_timezone")

    return cities


def get_sampled_events(events_input_path: str, sample_fraction: float, spark: SparkSession) -> pyspark.sql.DataFrame:
    if sample_fraction > 1:
        sample_fraction = 1

    events_sample = spark.read.parquet(events_input_path) \
        .sample(sample_fraction) \
        .withColumn('event_id', F.monotonically_increasing_id())

    return events_sample


def get_sphere_points_distance():
    distance = 2 * F.lit(avg_radius) * F.asin(
        F.sqrt(
            F.pow(F.sin((F.radians(F.col("lat")) - F.radians(F.col("city_lat"))) / 2), 2) +
            F.cos(F.radians(F.col("lat"))) * F.cos(F.radians(F.col("city_lat"))) *
            F.pow(F.sin((F.radians(F.col("lon")) - F.radians(F.col("city_lon"))) / 2), 2)
        )
    )

    return distance