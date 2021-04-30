package logistic.binary

import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{DataFrame, SparkSession}

object SampleDataUtils {
  val SPARK_HOME = "/usr/local/share/spark-3.1.1-bin-hadoop3.2"


  def getTrainingData(spark:SparkSession): DataFrame ={


    // Load training data
    val data_path = String.format("%s/data/mllib/sample_libsvm_data.txt",SPARK_HOME)
    println("load data from " + data_path)
    val training = spark.read.format("libsvm").load(data_path)
    training
  }

  def getMultiClassSampleData(spark:SparkSession): DataFrame ={
    val training = spark
      .read
      .format("libsvm")
      .load(s"${SPARK_HOME}/data/mllib/sample_multiclass_classification_data.txt")
    training
  }

}
