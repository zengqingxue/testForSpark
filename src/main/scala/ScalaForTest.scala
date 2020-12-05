object ScalaForTest {
  def main(args: Array[String]): Unit = {
    var map = Map[String, Int]("Alice"->10, "Bob"->3, "Cindy"->8)
    print(map)  //打印整个map

    map += ("Tom"->18)
    print(map)

  }
}
