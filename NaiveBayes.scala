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
  * Word frequencies for each document
  * @return (docId, word, frequency)
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
  
  /**
  * Trains the model, i.e. it computes P(word|spam) and P(word|non-spam)
  * @return (log P(word|spam), log P(word|non-spam))
  */
  def train(spamTrainingPath: String, nonSpamTrainingPath: String, dictSize: Int) : (Map[String, Double], Map[String, Double]) = {
    // train probabilities for spam
    val spamFreqs = buildDictionary(spamTrainingPath).map(w => (w._2, w._3))
    val N_spam = spamFreqs.size
    val nonSpamFreqs = buildDictionary(nonSpamTrainingPath).map(w => (w._2, w._3))
    val N_notSpam = nonSpamFreqs.size
    (
      spamFreqs.map(w => (w._1, math.log((w._2 + dictSize*0.5)/(N_spam + dictSize).toDouble))).toMap, 
      nonSpamFreqs.map(w => (w._1, math.log((w._2 + dictSize*0.5)/(N_notSpam + dictSize).toDouble))).toMap
    )
  }

  /**
  * Computes the probability that a given message 
  */
  def predict(testSetPath: String, p_word_spam: Map[String, Double], p_word_notSpam: Map[String, Double]) = {
    var p_spam: Double = 0
    var p_notSpam: Double = 0
    val files = new File(testSetPath).listFiles.iterator 

    files.foreach(f => {
      val msg = buildDictionary(f.toString)
      msg.map(w => w._2)
      .toSeq.foreach(w => {
        p_spam += p_word_spam.getOrElse(w, 0.0)
        p_notSpam += p_word_notSpam.getOrElse(w, 0.0)
      })
      
      p_spam -= math.log(2.0)
      p_notSpam -= math.log(2.0)
      val prediction = if (p_spam > p_notSpam) "SPAM" else "not spam"
      println("P(spam) = " + p_spam + ", P(not spam) = " + p_notSpam + ", " + prediction)
    })
  }
}

object NaiveBayes {
  def main(args: Array[String]) = {
    val spam_train = "/Users/ken/Dropbox/projects/naive_bayes_classifier/spamData/spam-train"
    val spam_test = "/Users/ken/Dropbox/projects/naive_bayes_classifier/spamData/spam-test"
    val nonspam_train = "/Users/ken/Dropbox/projects/naive_bayes_classifier/spamData/nonspam-train"
    val nonspam_test = "/Users/ken/Dropbox/projects/naive_bayes_classifier/spamData/nonspam-test"
    val allData = "/Users/ken/Dropbox/projects/naive_bayes_classifier/spamData/all"
    val nb = new NaiveBayes
    
    val wordList = nb.buildDictionary(allData)
    val training = nb.train(spam_train, nonspam_train, wordList.size)
    
    println("*** Spam Test Set ***\n")
    nb.predict(spam_test, training._1, training._2)
    println("\n*** Non Spam Test Set ***\n")
    nb.predict(nonspam_test, training._1, training._2)
  }
}
