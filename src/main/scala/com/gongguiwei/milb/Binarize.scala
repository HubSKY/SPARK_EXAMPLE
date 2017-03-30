package com.gongguiwei.milb
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
object Binarize {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("DecisionTree").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)

    //var dataPath = "/datum/test/data/ml-1m/"

    // Create a SparContext with the given master URL
    val data = sc.textFile("E://sample_tree_data.csv")
    val parsedData = data.map { line =>
      val parts = line.split('\t').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    parsedData.foreach(println)
    val maxDepth = 5
    val model = DecisionTree.train(parsedData, Classification, Gini, maxDepth)

    val labelAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / parsedData.count
    println("Training Error = " + trainErr)
    sc.stop()

  }

}