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

  /**
  * Counts word frequencies over entire dataset. More frequent terms
  * come first.
  * @return (word index, word, word frequency)
  */
  def buildDictionary(pathToFiles: String) : Seq[(Int, String, Int)] = {
    val dictionary = new HashMap[String, Int]
    val files = new File(pathToFiles).listFiles.iterator
    
    // read each file
    files.foreach {f => {
      val file = new Scanner(f)
      while (file.hasNext){
        val word = file.next.toString
        val count = dictionary.getOrElseUpdate(word, 1)
        if (count > 1) dictionary.put(word, count + 1)
      }
    }}

    var wordIndex: Int = -1
    dictionary.toSeq.sortWith(_._2 > _._2)
    .map(w => {
      wordIndex += 1
      (wordIndex, w._1, w._2)
    })
    .toSeq
  }

  /**
  *
  *
  */
  def buildFeatures(pathToFiles: String) : Seq[(Int, String, Int)] = {
    val files = new File(pathToFiles).listFiles.iterator
    val msgs = new ArrayBuffer[(Int, String, Int)]

    var msgId = 0
    files.foreach {f => {
      val msgDict = buildDictionary(f.toString)
      msgs appendAll msgDict.map(f => (msgId, f._2, f._3)).toSeq
      msgId += 1
    }}
  
    msgs
  }

  // train
  // compute frequency of each spam word, and freq of each non-spam word
  def train(spamPath: String, nonSpamPath: String) : (Map[String, Float], Map[String, Float]) = {
    val spamFreqs = buildDictionary(spamPath).map(w => (w._2, w._3))
    val N_spam = spamFreqs.size
    val nonSpamFreqs = buildDictionary(spamPath).map(w => (w._2, w._3))
    val N_notSpam = nonSpamFreqs.size
    (
      spamFreqs.map(w => (w._1, w._2.toFloat/N_spam.toFloat)).toMap, 
      nonSpamFreqs.map(w => (w._1, w._2.toFloat/N_notSpam.toFloat)).toMap
    )
  }

  // model

  // predict
}
