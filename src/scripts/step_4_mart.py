import os
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

sample_source = '/storage/data/sample/mart_3'


def calculate_mart(events_input_path: str, city_dict: str, spark: SparkSession) -> pyspark.sql.DataFrame:
    cities = geo.get_city_dict(city_dict, spark).persist()
    events = geo.get_sampled_events('/source/data/geo/events', 0.03, spark) \
        .where(("lat is not null and lon is not null"))

    events.write.mode("overwrite").parquet(f'{sample_source}')

    sampled_events = spark.read.parquet(sample_source)

    distance_window = Window().partitionBy("event_id").orderBy(F.col("distance").asc())

    message_map_to_city = (sampled_events.crossJoin(cities)
                           .where(
        "event_type == 'message' and event.message_from is not null and event.message_to is not null")
                           .withColumn("distance", geo.get_sphere_points_distance())
                           .withColumn("rn", F.row_number().over(distance_window))
                           .filter(F.col("rn") == 1)
                           .drop("rn", "city_lat", "city_lon", "distance")
                           .selectExpr(
        "event.datetime",
        "event.message_from as message_from",
        "event.message_to as message_to",
        "event.message_from as user_id",
        "event.message_ts as message_ts",
        "lat",
        "lon",
        "city_id",
        "city_timezone")
                           .persist()
                           )

    users_contacts = (message_map_to_city
                      .selectExpr("message_from", "message_to")
                      .union(message_map_to_city
                             .selectExpr("message_to", "message_from"))
                      .distinct()
                      )

    all_subscribers = (sampled_events
                       .where("event_type == 'subscription'")
                       .selectExpr("event.datetime", "event.user as user_id", "event.subscription_channel")
                       )

    users_withih_same_channel = (all_subscribers
                                 .selectExpr("user_id as user_left", "subscription_channel as channel").distinct()
                                 .join(all_subscribers
                                       .selectExpr("user_id as user_left",
                                                   "subscription_channel as channel").distinct(),
                                       "channel",
                                       "full"
                                       )
                                 .where(F.col("user_left") < F.col("user_right"))
                                 .distinct()
                                 )

    users_same_ch_no_contacts = users_withih_same_channel.exceptAll(users_contacts)

    last_message_window = Window().partitionBy("user_id").orderBy(F.col("message_ts").desc())
    user_last_message = (message_map_to_city
                         .withColumn("rn", F.row_number().over(last_message_window))
                         .filter(F.col("rn") == 1)
                         .withColumn("local_time", F.from_utc_timestamp(F.col("message_ts"), F.col('city_timezone')))
                         .select("user_id", "lon", "lat", "city_id", "local_time")
                         .drop("rn")
                         )

    distance_window = Window().partitionBy('event_id').orderBy(F.col("distance").asc())

    result = (users_same_ch_no_contacts
              .join(user_last_message
                    .selectExpr("city_id", "local_time", "user_id as user_left", "lat", "lon"), "user_left", "inner")
              .join(user_last_message
                    .selectExpr("user_id as user_right", "lat as city_lat", "lon as city_lon"), "user_right", "inner")
              .withColumn("distance", geo.get_sphere_points_distance())
              .withColumn("rn", F.row_number().over(distance_window))
              .filter(F.col("rn") == 1)
              .selectExpr("user_left", "user_right", "processed_dttm", "city_id as zone_id", "local_time")
              )

    return result


def main():
    spark = SparkSession \
        .builder \
        .master("yarn") \
        .config("spark.driver.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .appName("project_s7_step_4") \
        .getOrCreate()

    events_input_path = sys.argv[1]
    city_dict = sys.argv[2]
    output_path = sys.argv[3]

    mart_df = calculate_mart(events_input_path, city_dict, spark)

    mart_df.write.mode("overwrite").parquet(f'{output_path}')


if __name__ == '__main__':
    main()