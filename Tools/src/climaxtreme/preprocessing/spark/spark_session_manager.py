"""
Manages the Spark session for the climate data preprocessing pipeline.
"""

import logging
import os
from typing import Optional
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class SparkSessionManager:
    """
    Manages the lifecycle of a Spark session.
    """
    
    def __init__(self, app_name: str = "climaXtreme"):
        """
        Initializes the session manager.
        
        Args:
            app_name: The name of the Spark application.
        """
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None

    def get_spark_session(self) -> SparkSession:
        """
        Get or create a Spark session with an optimized configuration.
        
        Returns:
            A SparkSession instance.
        """
        if self.spark is None:
            shuffle_partitions = os.getenv("CLX_SHUFFLE_PARTITIONS", "200")
            default_parallelism = os.getenv("CLX_DEFAULT_PARALLELISM", "100")
            ui_port = os.getenv("CLX_SPARK_UI_PORT", "4040")

            self.spark = (
                SparkSession.builder
                .appName(self.app_name)
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.adaptive.skewJoin.enabled", "true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.shuffle.partitions", shuffle_partitions)
                .config("spark.default.parallelism", default_parallelism)
                .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128 MB
                .config("spark.ui.port", ui_port)
                .config("spark.sql.codegen.wholeStage", "false")
                .config("spark.sql.codegen.fallback", "true")
                .config("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem")
                .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
                .getOrCreate()
            )
            
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark session initialized with optimized settings for large datasets")
        
        return self.spark

    def stop_spark_session(self) -> None:
        """
        Stops the current Spark session.
        """
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")