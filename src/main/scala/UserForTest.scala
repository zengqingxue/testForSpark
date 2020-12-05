import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSizeHint}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.{SparkConf, SparkContext}
import spire.implicits.eqOps

import scala.reflect.ClassManifestFactory.Null

object UserForTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("testForSpark").setMaster("local[1]")
//    val sc = new SparkContext(conf)  //通过SparkContext+conf来创建sc
//    val spark: SparkSession = new SparkSession(sc) ???
    val spark = SparkSession ////SparkSession+conf来创建ssc
      .builder()
      .config(conf)
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext  //通过sparkSession(ssc).sparkContext来创建spark(sc)
    sc.setLogLevel("WARN")
    sc.setCheckpointDir("./scalaForTest")
//    val df1 = spark.createDataFrame(
//        Seq((0, "a", "max"), (1, "b","mid"), (2, "c","min"), (3, "a","max"), (4, "a","max"), (5, "c","min"))
//      ).toDF("id", "category","rank")
    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),
        (1, 19, 2.0, Vectors.dense(1.0, 11.0, 1.5), 0.0),
        (2, 20, 3.0, Vectors.dense(2.0, 12.0), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
   dataset.show()

/*    dataset.show()
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")
//      .setHandleInvalid("keep") //到底是选skip，还是选择keep,取决于这样处理后会不会对后续的模型训练产生影响
      .setHandleInvalid("skip")*/

    val sizeHint = new VectorSizeHint()  //创建VectorSizeHint转换器,进行元数据的指定,保证向量列长度的确定
    val datasetWithSize = sizeHint
      .setInputCol("userFeatures")
      .setHandleInvalid("skip")
      .setSize(3)  //userFeatures特征列为vector列,其长度为3,指定好
      .transform(dataset) //使用VectorSizeHint转换器对数据集dataset进行transform
    datasetWithSize.show()
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")
      .setHandleInvalid("keep") //到底是选skip，还是选择keep,取决于这样处理后会不会对后续的模型训练产生影响

    val output = assembler.transform(datasetWithSize) //assembler.transform的是经过VectorSizeHint转换后得到的数据集
    output.show(false)




    //    import spark.implicits._
    //    import spark.sql
    //    val a= sql("""select "1" as k""")
    //    a.show
    //    val rdd=sc.parallelize(List(1,2,3,4,5,6)).map(_*3)
    //    rdd.filter(_>10).collect().foreach(println)
    //    println(rdd.reduce(_+_))
    //    println("hello world")

   /* val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
    indexer.fit(df1).transform(df1).show()

    val indexers = df1.columns.filter(_!="id").flatMap((column:String)=>{
      val indexer = new StringIndexer()
        .setInputCol(column)
        .setOutputCol(s"${column}Index")
      Array(indexer)
    })
    new Pipeline().setStages(indexers).fit(df1).transform(df1).show()*/


    spark.stop()

  }

}