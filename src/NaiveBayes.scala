//package bayes

import java.io.File
import java.util.Scanner
import collection.mutable.{HashMap, HashSet, ArrayBuffer}
import collection.immutable.Map

class NaiveBayes(alpha: Double = 1.0) {
  var pUnknownWord_spam: Double = 0.0
  var pUnknownWord_nonSpam: Double = 0.0

  /*def tokenize(sentence: String) : (Int, Array[String]) = {
    (sentence.split('.').size, 
    sentence.replaceAll("[.,]", "").split(" ").map {s => s.toLowerCase})
  }*/

  /**
  * Counts word frequencies over files in the specifed input. More frequent terms
  * come first.
  * @return (word, word frequency)
  */
  def buildVocabulary(pathToFiles: String) : Map[String, Int] = {
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
    dictionary.toMap  
  }

  /**
  * Counts word frequencies over single data file. More frequent terms
  * come first.
  * @return (word, word frequency)
  */
  def buildFileDict(pathToFile: String) : Map[String, Int]= {
    val dictionary = new HashMap[String, Int]
    val msg = new File(pathToFile)

    val file = new Scanner(msg)
    while (file.hasNext){
      val word = file.next.toString
      val count = dictionary.getOrElseUpdate(word, 1)
      if (count > 1) dictionary.put(word, count + 1)
    }
    dictionary.toMap
  }

  /**
  * Word frequencies for each document
  * @return (docId, word, frequency)
  */
  /*def buildFeatures(pathToFiles: String) : Seq[(Int, String, Int)] = {
    val files = new File(pathToFiles).listFiles.iterator
    val msgs = new ArrayBuffer[(Int, String, Int)]

    var msgId = 0
    files.foreach {f => {
      val msgDict = buildVocabulary(f.toString)
      msgs appendAll msgDict.map(f => (msgId, f._2, f._3)).toSeq
      msgId += 1
    }}
  
    msgs
  }
  */
  
  /**
  * Trains the model, i.e. it computes conditional probabilities
  * P(word|spam) and P(word|non-spam)
  * @return (log P(word|spam), log P(word|non-spam))
  */
  def train(spamTrainingPath: String, nonSpamTrainingPath: String, dictSize: Int) : (Map[String, Double], Map[String, Double], Double, Double) = {
    val spamFreqs = buildVocabulary(spamTrainingPath)
    //spamFreqs.filter(w => w._2 > 1).foreach(w => println(w._1 + ": " + w._2))
    val spamWordCount = spamFreqs.size.toDouble
    val N_spam = new File(spamTrainingPath).listFiles.size.toDouble
    val nonSpamFreqs = buildVocabulary(nonSpamTrainingPath)
    val nonSpamWordCount = nonSpamFreqs.size.toDouble
    val N_notSpam = new File(nonSpamTrainingPath).listFiles.size.toDouble
    //nonSpamFreqs.filter(w => w._2 == 0).foreach(w => println(w._1 + ": " + w._2))

    val spamDenominator: Double = (spamWordCount + (dictSize + 1) * alpha).toDouble
    val nonSpamDenominator: Double = (nonSpamWordCount + (dictSize + 1) * alpha).toDouble
    val N: Double = N_spam + N_notSpam
    val p_spam = N_spam / N
    val p_notSpam = N_notSpam / N

    this.pUnknownWord_spam = math.log(alpha / spamDenominator)
    this.pUnknownWord_nonSpam = math.log(alpha / nonSpamDenominator) 

    (
      spamFreqs.map(w => 
        (w._1, math.log((w._2 + alpha) / spamDenominator))
      ).toMap, 
      nonSpamFreqs.map(w => 
        (w._1, math.log((w._2 + alpha) / nonSpamDenominator))
      ).toMap,
      math.log(p_spam),
      math.log(p_notSpam)
    )
  }

  // TODO fix bug here
  /**
  * Computes the probability that a given message is spam:
  * P(spam|word) = P(word|spam) * P(spam) / p(word)
  * P(~spam|word) = P(word|~spam) * P(~spam) / p(word)
  * P(s) = Sum_w P(s|w)
  */
  def predict(testSetPath: String, p_word_spam: Map[String, Double], p_word_notSpam: Map[String, Double], pSpam: Double, pNotSpam: Double) = {
    
    val files = new File(testSetPath).listFiles.iterator 

    files.foreach(f => {
      var p_spam: Double = 0
      var p_notSpam: Double = 0
      val prob_file: Double = 0
      val msg = buildFileDict(f.toString)

      //p_word_spam.foreach(w => println(w._1 + ": " + w._2))

      msg.foreach(w => {
        p_spam += p_word_spam.getOrElse(w._1, pUnknownWord_spam) * w._2.toDouble
        p_notSpam += p_word_notSpam.getOrElse(w._1, pUnknownWord_nonSpam) * w._2.toDouble
      })

      p_spam += pSpam
      p_notSpam += pNotSpam
      val prediction = if (p_spam > p_notSpam) "SPAM" else "not spam"
      println("P(spam) = " + p_spam + ", P(not spam) = " + p_notSpam + ", " + prediction)
    })
  }
}

object NaiveBayes {
  def main(args: Array[String]) : Unit = {
    val pathToData = args(0).toString
    val spam_train = pathToData + "/spam-train"
    val spam_test = pathToData + "/spam-test"
    val nonspam_train = pathToData + "/nonspam-train"
    val nonspam_test = pathToData + "/nonspam-test"
    val allTrainingData = pathToData + "/training-set"

    val nb = new NaiveBayes

    val wordList = nb.buildVocabulary(allTrainingData)
    //wordList.filter(w => w._2 > 1).foreach(w => println(w._1 + ": " + w._2))
    val training = nb.train(spam_train, nonspam_train, wordList.size)

    //println("pUnknownWord_spam = " + nb.pUnknownWord_spam)
    //println("pUnknownWord_nonSpam = " + nb.pUnknownWord_nonSpam)
    //println("Note: log of probabilities is used, rather than probability")
    println("*** Spam Test Set ***\n")
    nb.predict(spam_test, training._1, training._2, training._3, training._4)
    
    println("\n*** Non Spam Test Set ***\n")
    nb.predict(nonspam_test, training._1, training._2, training._3, training._4)
  }
}

