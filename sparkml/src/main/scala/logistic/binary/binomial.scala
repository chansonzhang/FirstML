// scalastyle:off println
package logistic.binary


// $example on$
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.serializer.KryoSerializer
// $example off$
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max


object RunLR {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val training = SampleDataUtils.getTrainingData(spark)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Extract the summary from the returned LogisticRegressionModel instance trained earlier
    val trainingSummary = lrModel.binarySummary

    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = trainingSummary.roc
    roc.show()
    println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

    // Set the model threshold to maximize F-Measure
    val fMeasure = trainingSummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure"))
      .head()
      .getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold").head().getDouble(0)

    println(s"best threshold: ${bestThreshold}")
    lrModel.setThreshold(bestThreshold)

  }
}


