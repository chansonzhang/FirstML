package spark

import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession

object SparkSessionUtils {
  def getSession(): SparkSession ={
    val spark = SparkSession.builder().config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.master", "local")
      .getOrCreate()
    spark
  }

}
