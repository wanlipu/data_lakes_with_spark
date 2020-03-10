import configparser
from datetime import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str
from pyspark.sql.types import IntegerType as Int, LongType as Long, DateType as Date, TimestampType as Tst


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['Credentials']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['Credentials']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.0') \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    load song data in json format from S3 bucket and process these data by extracting 
    songs table and artists table, and save these tables back to S3 bucket
    
    :param spark: spark session
    :param input_data: data location for input data
    :param output_data: data location for output data
    :return: no return value
    '''
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    # create songs schema
    songSchema = R([
        Fld('artist_id',Str()),
        Fld('artist_latitude',Dbl()),
        Fld('artist_location',Str()),
        Fld('artist_longitude',Dbl()),
        Fld('artist_name',Str()),
        Fld('duration',Dbl()),
        Fld('num_songs',Int()),
        Fld('title',Str()),
        Fld('year',Int()),
    ])
    
    # load songs json files from S3
    df_songs = spark.read.json(song_data, schema=songSchema)
    
    # select columns for songs_table
    songs_attr = ['title', 'artist_id','year', 'duration']
    songs_table = df_songs.select(songs_attr)\
    .dropDuplicates()\
    .withColumn('song_id', monotonically_increasing_id())
    
    # write songs_table to S3
    songs_table.write.partitionBy('year', 'artist_id').parquet(output_data + 'songs/')
    
    # select artists columns
    artists_attr = ['artist_id', 'artist_name', 'artist_location', 
                   'artist_latitude', 'artist_longitude']
    artists_table = df_songs.select(artists_attr)\
    .dropDuplicates()
    
    artists_table = artists_table\
    .withColumnRenamed('artist_name','name')\
    .withColumnRenamed('artist_location','location')\
    .withColumnRenamed('artist_latitude','latitude')\
    .withColumnRenamed('artist_longitude','longitude')
    
    # write artists_table to S3
    artists_table.write.parquet(output_data + 'artists/')


def process_log_data(spark, input_data, output_data):
    '''
    load log data in json format from S3 bucket and process these data by extracting 
    users table, time table and songplays table, and save these tables back to S3 bucket
    
    :param spark: spark session
    :param input_data: data location for input data
    :param output_data: data location for output data
    :return: no return value
    '''
    
    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'
    # log_data = input_data + 'log_data/*.json' # for local files
    
    logdataSchema = R([
        Fld('artist', Str()),
        Fld('auth', Str()),
        Fld('firstName', Str()),
        Fld('gender', Str()),
        Fld('itemInSession', Long()),
        Fld('lastName', Str()),
        Fld('length', Dbl()),
        Fld('level', Str()),
        Fld('location', Str()),
        Fld('method', Str()),
        Fld('page', Str()),
        Fld('registration', Dbl()),
        Fld('sessionId', Long()),
        Fld('song', Str()),
        Fld('status', Long()),
        Fld('ts', Long()),
        Fld('userAgent', Str()),
        Fld('userId', Str()),
    ])

    # load json files from S3
    df_log = spark.read.json(log_data, schema=logdataSchema)
    df_log = df_log.filter(df_log.page == 'NextSong')
    
    # select users columns
    users_attr = ['userId', 'firstName', 'lastName', 'gender', 'level']
    users_table = df_log.select(users_attr)\
    .dropDuplicates()
    
    users_table = users_table\
    .withColumnRenamed('userId','user_id')\
    .withColumnRenamed('firstName','first_name')\
    .withColumnRenamed('lastName','last_name')
    
    # write users table to S3
    users_table.write.parquet(output_data + 'users/')
    
    # create time table
    tsFormat = 'yyyy-MM-dd HH:MM:ss z'
    time_table = df_log.withColumn('ts', 
                                   to_timestamp(date_format((df_log.ts/1000)\
                                                            .cast(dataType=Tst()), 
                                                            tsFormat), tsFormat))

    time_table = time_table.select(col('ts').alias('start_time'),
                                   hour(col('ts')).alias('hour'),
                                   dayofmonth(col('ts')).alias('day'), 
                                   weekofyear(col('ts')).alias('week'), 
                                   month(col('ts')).alias('month'),
                                   year(col('ts')).alias('year'))

    # write time table to S3
    time_table.write.partitionBy('year', 'month').parquet(output_data + 'time/')

    # load songs and artist tables from previous handling
    df_songs = spark.read.parquet(output_data + 'songs/*/*/*')
    df_artists = spark.read.parquet(output_data + 'artists/*')
    df_artists = df_artists.drop('location')

    # create songs_logs table
    songs_logs = df_log.join(df_songs, (df_log.song == df_songs.title))
    
    # create artists_songs_logs table
    artists_songs_logs = songs_logs.join(df_artists, 
                                         (songs_logs.artist == df_artists.name))
    
    artists_songs_logs = artists_songs_logs\
    .withColumn('ts', 
                to_timestamp(date_format((artists_songs_logs.ts/1000)\
                                         .cast(dataType=Tst()),tsFormat), tsFormat))
    
    # create songplays table
    songplays = artists_songs_logs.join(
        time_table,
        artists_songs_logs.ts == time_table.start_time, 'left')
    
    songplays_attr = ['start_time', 'userId', 'level', 'song_id', 'artist_id',
                      'sessionId', 'location', 'userAgent', 'year', 'month']
    
    songplays_table = songplays.select(songplays_attr)\
    .dropDuplicates()
    
    songplays_table = songplays_table\
    .withColumnRenamed('userId','user_id')\
    .withColumnRenamed('sessionId','session_id')\
    .withColumnRenamed('userAgent','user_agent')\
    .repartition('year', 'month')

    # write songplays table to S3
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + 'songplays/')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-dend/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
