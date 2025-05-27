import os
import sys

import findspark

findspark.init()
findspark.find()

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import row_number, when

import geo

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME'] = '/usr'
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYTHONPATH'] = '/usr/local/lib/python3.8'

sample_source = '/storage/data/sample/mart_1'


def calculate_mart(events_input_path: str, city_dict: str, spark: SparkSession) -> pyspark.sql.DataFrame:
    cities = geo.get_city_dict(city_dict, spark).persist()
    events = geo.get_sampled_events(events_input_path, 0.03, spark) \
        .where(("event_type == 'message' AND event.message_ts is not null AND lat is not null and lon is not null")) \
        .withColumn("user_id", F.col("event.message_from")) \
        .withColumn("message_ts", F.to_date("event.message_ts")) \
        .select("user_id", "event_id", "message_ts", "date", "lat", "lon")

    events.write.mode("overwrite").parquet(f'{sample_source}')

    messages = spark.read.parquet(sample_source)

    distance_window = Window().partitionBy('event_id').orderBy(F.col("distance").asc())

    message_map_to_city = (messages.crossJoin(cities)
                           .withColumn("distance", geo.get_sphere_points_distance())
                           .withColumn("rn", F.row_number().over(distance_window))
                           .filter(F.col("rn") == 1)
                           .drop("rn", "lat", "lon", "city_lat", "city_lon", "distance")
                           .persist()
                           )

    actual_city_window = Window().partitionBy('user_id').orderBy(F.col("message_ts").desc())
    user_actual_city = (message_map_to_city
                        .withColumn("rn", F.row_number().over(actual_city_window))
                        .filter(F.col("rn") == 1)
                        .withColumn("local_time", F.from_utc_timestamp(F.col("message_ts"), F.col('city_timezone')))
                        .select('user_id', 'city', 'local_time')
                        .drop("rn")
                        )


    unique_date_message = (message_map_to_city
                           .groupBy("user_id", "city", "date")
                           .count()
                           )

    home_city_window = Window().partitionBy("user_id", "city").orderBy(F.col("date").asc())

    user_home_city = (unique_date_message
                      .withColumn("rn", F.row_number().over(home_city_window))
                      .withColumn("common_date", F.expr("date_add(date, -rn)"))
                      .groupBy("user_id", "city", "common_date")
                      .count()
                      .filter(F.col("count") >= 27)
                      .withColumn("rn2",
                                  F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc("common_date"))))
                      .filter(F.col("rn2") < 2)
                      .select("user_id", "city", "count")
                      )

    travel_city_window = Window.partitionBy('user_id').orderBy('date')

    user_travel_city = (unique_date_message
                        .withColumn("previous_city", F.lag("city").over(travel_city_window))
                        .withColumn("common_flag", when(F.col("previous_city").isNull(), 0)
                                    .when(F.col("previous_city") == F.col('city'), 0)
                                    .otherwise(1))
                        .where(F.col("common_flag") == 1)
                        .groupBy("user_id")
                        .agg(F.collect_list('city').alias('travel_array'), F.count("*").alias('travel_count'))
                        )

    result = (user_actual_city
    .alias('a')
    .join(user_home_city.alias('b'), F.col('a.user_id') == F.col('b.user_id'), how='left')
    .join(user_travel_city.alias('c'), F.col('a.user_id') == F.col('c.user_id'), how='left')
    .select(
        F.col('a.user_id'),
        F.col('a.city').alias('act_city'),
        F.col('b.city').alias('home_city'),
        F.col('c.travel_count'),
        F.col('c.travel_array'),
        F.col('a.local_time'))
    )

    return result


def main():
    spark = SparkSession \
        .builder \
        .master("yarn") \
        .config("spark.driver.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .appName("project_s7_step_2") \
        .getOrCreate()

    events_input_path = sys.argv[1]
    city_dict = sys.argv[2]
    output_path = sys.argv[3]

    mart_df = calculate_mart(events_input_path, city_dict, spark)

    mart_df.write.mode("overwrite").parquet(f'{output_path}')


if __name__ == '__main__':
    main()