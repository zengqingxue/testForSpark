package testForSparkML

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object TestForLR {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("TestForLR").setMaster("local[1]")
//    val conf = new SparkConf().setAppName("testForSpark").setMas("local[1]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    //    val sc = new SparkContext(conf)
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
//    sc.setCheckpointDir("./TestForLR")

    val trainDF: DataFrame = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")
    trainDF.show()
//    +-----+--------------+
//    |label|      features|
//    +-----+--------------+
//    |  1.0| [0.0,1.1,0.1]|
//    |  0.0|[2.0,1.0,-1.0]|
//    |  0.0| [2.0,1.3,1.0]|
//    |  1.0|[0.0,1.2,-0.5]|
//    +-----+--------------+
    val lr: LogisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
    val lrModel: LogisticRegressionModel = lr.fit(trainDF)
    val trainTransformDF: DataFrame = lrModel.transform(trainDF)
    trainTransformDF.show(false)

    println("===========================lr处理二分类: 训练集上的效果评估 ==============================")
/*    //1. 训练集上的效果评估
    println("Binomial model's numClasses: " +lrModel.numClasses) //2
    println("Binomial model's numFeatures: " +lrModel.numFeatures) //3
    println("Binomial model's coefficients: " +lrModel.coefficients)
//      [-3.1009356010205327,2.60821473832145,-0.3801791225430309]
    println("Binomial model's intercept: " +lrModel.intercept) //0.06817659473873576
    println("Binomial model's hasSummary: " +lrModel.hasSummary) //true*/

    val summary: LogisticRegressionTrainingSummary = lrModel.summary
  /*  summary.predictions.show(false)
    +-----+--------------+----------------------------------------+----------------------------------------+----------+
    |label|features      |rawPrediction                           |probability                             |prediction|
    +-----+--------------+----------------------------------------+----------------------------------------+----------+
    |1.0  |[0.0,1.1,0.1] |[-2.8991948946380277,2.8991948946380277]|[0.05219337666300758,0.9478066233369924]|1.0       |
    |0.0  |[2.0,1.0,-1.0]|[3.1453007464378486,-3.1453007464378486]|[0.9587231582899965,0.04127684171000351]|0.0       |
    |0.0  |[2.0,1.3,1.0] |[3.1231945700274752,-3.1231945700274752]|[0.9578394235295768,0.04216057647042307]|0.0       |
    |1.0  |[0.0,1.2,-0.5]|[-3.3881238419959914,3.3881238419959914]|[0.03266869266264558,0.9673313073373544]|1.0       |
    +-----+--------------+----------------------------------------+----------------------------------------+----------+
    println("Binomial summary's objectiveHistory:"+summary.objectiveHistory.mkString(","))
    0.6931471805599453,0.4274232924955569,0.12582029480507145,0.1144102682046959,
    0.10884647747685207,0.10840715948469526,0.10831890205457553,0.10826850363446151,
    0.10823620528587521,0.10822178855356387,0.10821879266352699
    println("Binomial summary's totalIteration:"+summary.totalIterations)  //11*/
    val binarySummary: BinaryLogisticRegressionSummary = summary.asInstanceOf[BinaryLogisticRegressionSummary]
    binarySummary.predictions.show(false)

//    binarySummary.precisionByThreshold.show()
    /*+-------------------+------------------+
    |          threshold|         precision|
    +-------------------+------------------+
    | 0.9673313073373544|               1.0|
    | 0.9478066233369924|               1.0|
    |0.04216057647042307|0.6666666666666666|
    |0.04127684171000351|               0.5|
    +-------------------+------------------+*/
//    binarySummary.recallByThreshold.show()
/*    +-------------------+------+
    |          threshold|recall|
    +-------------------+------+
    | 0.9673313073373544|   0.5|
    | 0.9478066233369924|   1.0|
    |0.04216057647042307|   1.0|
    |0.04127684171000351|   1.0|
    +-------------------+------+*/
//    binarySummary.fMeasureByThreshold.show()
  /*  +-------------------+------------------+
    |          threshold|         F-Measure|
    +-------------------+------------------+
    | 0.9673313073373544|0.6666666666666666|
    | 0.9478066233369924|               1.0|
    |0.04216057647042307|               0.8|
    |0.04127684171000351|0.6666666666666666|
    +-------------------+------------------+*/
//    binarySummary.pr.show()
/*    +------+------------------+
    |recall|         precision|
    +------+------------------+
    |   0.0|               1.0|
    |   0.5|               1.0|
    |   1.0|               1.0|
    |   1.0|0.6666666666666666|
    |   1.0|               0.5|
    +------+------------------+*/
//    binarySummary.roc.show()
    /*+---+---+
    |FPR|TPR|
    +---+---+
    |0.0|0.0|
    |0.0|0.5|
    |0.0|1.0|
    |0.5|1.0|
    |1.0|1.0|
    |1.0|1.0|
    +---+---+*/
    println("Binomial summmary's areaUnderROC: "+binarySummary.areaUnderROC) //1.0



    //2. 测试集上的效果评估
    println("===========================lr处理二分类: 测试集上的效果评估 ==============================")
    val testDF: DataFrame = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label","features")
//    testDF.show()
    /*    +-----+--------------+
    |label|      features|
    +-----+--------------+
    |  1.0|[-1.0,1.5,1.3]|
    |  0.0|[3.0,2.0,-0.1]|
    |  1.0|[0.0,2.2,-1.5]|
    +-----+--------------+*/

    val d: Double = lrModel.predict(testDF.head().getAs("features"))
    val testSummary: LogisticRegressionSummary = lrModel.evaluate(testDF)
    testSummary.predictions.show(false)
    testSummary.predictions.show()
    testSummary.accuracy
    testSummary.fMeasureByLabel
    testSummary.precisionByLabel
    testSummary.falsePositiveRateByLabel
    val testBinarySummary: BinaryLogisticRegressionSummary = testSummary.asBinary
    testBinarySummary.areaUnderROC
    testBinarySummary.pr.show(false)
    testBinarySummary.roc.show(false)
    testBinarySummary.fMeasureByThreshold.show(false)
    testBinarySummary.recallByThreshold.show(false)
    testBinarySummary.precisionByThreshold.show(false)
    testBinarySummary.predictions
    testBinarySummary.accuracy
    testBinarySummary.precisionByLabel
    testBinarySummary.fMeasureByLabel
    testBinarySummary.falsePositiveRateByLabel

/*    +-----+--------------+--------------------------------------+------------------------------------------+----------+
    |label|features      |rawPrediction                         |probability                               |prediction|
    +-----+--------------+--------------------------------------+------------------------------------------+----------+
    |1.0  |[-1.0,1.5,1.3]|[-6.587201443935503,6.587201443935503]|[0.0013759947069214356,0.9986240052930786]|1.0       |
    |0.0  |[3.0,2.0,-0.1]|[3.980182819425659,-3.980182819425659]|[0.9816604009374171,0.01833959906258293]  |0.0       |
    |1.0  |[0.0,2.2,-1.5]|[-6.376517702860472,6.376517702860472]|[0.0016981475578358176,0.9983018524421641]|1.0       |
    +-----+--------------+--------------------------------------+------------------------------------------+----------+*/
    val data: Array[linalg.Vector] = Array(
      (Vectors.dense(-1.0, 1.5, 1.3)),
      (Vectors.dense(3.0, 2.0, -0.1)),
      (Vectors.dense(0.0, 2.2, -1.5))
    )
    println(data)
    val featuresDF: DataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    featuresDF.show()
 /*   +--------------+
    |      features|
    +--------------+
    |[-1.0,1.5,1.3]|
    |[3.0,2.0,-0.1]|
    |[0.0,2.2,-1.5]|
    +--------------+*/
//    val frame: DataFrame = lrModel.transform(testDF)
    lrModel.transform(featuresDF).show(false)
    lrModel.transform(testDF).show(false)  //在原先测试集输入数据的基础上扩充了列
    /*+-----+--------------+--------------------------------------+------------------------------------------+----------+
    |label|features      |rawPrediction                         |probability                               |prediction|
    +-----+--------------+--------------------------------------+------------------------------------------+----------+
    |1.0  |[-1.0,1.5,1.3]|[-6.587201443935503,6.587201443935503]|[0.0013759947069214356,0.9986240052930786]|1.0       |
    |0.0  |[3.0,2.0,-0.1]|[3.980182819425659,-3.980182819425659]|[0.9816604009374171,0.01833959906258293]  |0.0       |
    |1.0  |[0.0,2.2,-1.5]|[-6.376517702860472,6.376517702860472]|[0.0016981475578358176,0.9983018524421641]|1.0       |
    +-----+--------------+--------------------------------------+------------------------------------------+----------+
*/




 /*   //3. lr处理多分类
    println("===========================lr处理多分类==============================")
    val multinomialTraining: DataFrame = spark.createDataFrame(Seq(
      (3.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")
    multinomialTraining.show(false)
/*    +-----+--------------+
    |label|features      |
    +-----+--------------+
    |3.0  |[0.0,1.1,0.1] |
    |0.0  |[2.0,1.0,-1.0]|
    |0.0  |[2.0,1.3,1.0] |
    |1.0  |[0.0,1.2,-0.5]|
    +-----+--------------+*/
    val mutiLRModel: LogisticRegressionModel = lr.fit(multinomialTraining)
    println("Multinomial model's numClasses: "+ mutiLRModel.numClasses)  //4
    println("Multinomial model's numFeatures: "+ mutiLRModel.numFeatures)  //3
    println("Multinomial model's coefficients : "+ mutiLRModel.coefficientMatrix)
//    2.3755319833863693    4.2888956935790485   0.33925717456625215  //为什么多分类的特征重要性有多个??
//    -1.1168211546663132   6.986561348601071    -1.1694540068639812
//    -0.19800783089303753  -18.265435960577697  0.04667029423568606
//    -1.0607029978270182   6.989978918397559    0.7835265380620421
    println("Multinomial model's intercept: "+ mutiLRModel.interceptVector)
//    [0.548550771807033,0.05028251887521078,-0.9026627343287892,0.3038294436465453]
    println("Multinomial model's hasSummary: "+ mutiLRModel.hasSummary) //True
    mutiLRModel.transform(multinomialTraining).show(false)
   /* +-----+--------------+----------------------------------------------------------------------------+-----------------------------------------------------------------------------------+----------+
    |label|features      |rawPrediction                                                               |probability                                                                        |prediction|
    +-----+--------------+----------------------------------------------------------------------------+-----------------------------------------------------------------------------------+----------+
    |3.0  |[0.0,1.1,0.1] |[5.300261752200613,7.618554601649991,-20.98997526154069,8.071158907690064]  |[0.03685783217150718,0.37441356957458194,1.408720793881421E-13,0.58872859825377]   |3.0       |
    |0.0  |[2.0,1.0,-1.0]|[9.249253257592567,5.972655565007637,-19.610784650928245,4.388875828328026] |[0.9564764415586482,0.03611320060230872,2.798459760120472E-13,0.007410357838763275]|0.0       |
    |0.0  |[2.0,1.3,1.0] |[11.214436314798789,5.729715955859994,-24.997074850630185,8.052922579971376]|[0.9555558881232505,0.003965265818210627,1.7938933190121827E-16,0.0404788460585386]|0.0       |
    |1.0  |[0.0,1.2,-0.5]|[5.5255970168187645,9.018883140628486,-22.84452103413987,8.300040876692595] |[0.02003061745720267,0.6588843985612369,9.56552071269138E-15,0.32108498398155083]  |1.0       |
    +-----+--------------+----------------------------------------------------------------------------+-----------------------------------------------------------------------------------+----------+
*/

*/


  }

}
