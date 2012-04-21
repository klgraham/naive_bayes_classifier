package bayes

import java.io.File
import java.util.Scanner
import collection.mutable.{HashMap, HashSet, ArrayBuffer}
import collection.immutable.Map

class NaiveBayes {
  def bayes(p_a: Float, p_b: Float, p_b_a: Float) : Float = p_a * p_b_a / p_b

  def tokenize(sentence: String) : (Int, Array[String]) = {
    (sentence.split('.').size, 
    sentence.replaceAll("[.,]", "").split(" ").map {s => s.toLowerCase})
  }

  def buildDictionary(pathToFiles: String) : Seq[(String, Int)] = {
    val dictionary = new HashSet[String]
    val files = new File(pathToFiles).listFiles.iterator
    
    // read each file
    files.foreach {f => {
      val file = new Scanner(f)
      while (file.hasNext) 
        dictionary += file.next.toString
    }}

    var wordIndex: Int = -1
    dictionary.map {s => {
      wordIndex += 1
      (s, wordIndex)}
    }.toMap.toSeq.sortWith(_._2 > _._2)
  }

  def buildFeatures(pathToFiles: String, dictionary: Map[String, Int]) : Seq[(Int, Int, Int)] = {
    val files = new File(pathToFiles).listFiles.iterator
    val msgs = new ArrayBuffer[(Int, Int, Int)]

    // loop over each msg
    var msgId = 0
    files.foreach {f => {
      val msg = new Scanner(f)
      val features = new HashMap[String, Int]
      while (msg.hasNext){
        val word = msg.next
        // build hashmap of [wordId, wordCount]
        val count = features.getOrElseUpdate(word, 1)
        if (count > 1) features.put(word, count + 1)
      }
      msgs appendAll features.map(f => (msgId, dictionary.get(f._1).get, f._2)).toSeq
    }}
  
    msgs
  }

  // train

  // model

  // predict
}
