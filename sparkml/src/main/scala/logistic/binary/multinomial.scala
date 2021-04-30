package logistic.binary

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import spark.SparkSessionUtils

object multinomial {

  def main(args: Array[String]): Unit = {
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")


    val spark = SparkSessionUtils.getSession()
    import spark.implicits._

    val training = SampleDataUtils.getTrainingData(spark)
    val mlrModel = mlr.fit(training)

    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")
  }


}
