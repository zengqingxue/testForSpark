package testForSparkML

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/*
这个程序是一个很好的测试sparkUI及spark的Application划分job,stage,task的spark程序
 */
object Spark_test {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\Documents\\Downloads\\Programs\\DevelopTools\\")
    //创建SparkConf配置文件
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("test")
    //创建SparkContext对象
//    val sc = new SparkContext(sparkConf)
    val sc = new SparkContext(sparkConf)

    //3. 创建RDD
    val dataRDD: RDD[Int] = sc.makeRDD(List(1,2,3,4,1,2),2)

    //3.1 聚合
    val resultRDD: RDD[(Int, Int)] = dataRDD.map((_,1)).reduceByKey(_+_)

    // Job：一个Action算子就会生成一个Job；
    //3.2 job1打印到控制台
    resultRDD.collect().foreach(println)

    //3.3 job2输出到磁盘
    resultRDD.saveAsTextFile("output")

    Thread.sleep(10000000)

    //释放资源/关闭连接
    sc.stop()
  }

}
