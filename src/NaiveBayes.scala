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
  * Counts word frequencies over files in the specifed input. 
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
  * Counts word frequencies over single data file.
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
  * Trains the model, i.e. it computes conditional probabilities
  * P(word|spam) and P(word|non-spam)
  * @return (log P(word|spam), log P(word|non-spam))
  */
  def train(spamTrainingPath: String, nonSpamTrainingPath: String, dictSize: Int) : (Map[String, Double], Map[String, Double], Double, Double) = {
    val spamFreqs = buildVocabulary(spamTrainingPath)
    val spamWordCount = spamFreqs.size.toDouble
    val N_spam = new File(spamTrainingPath).listFiles.size.toDouble
    val nonSpamFreqs = buildVocabulary(nonSpamTrainingPath)
    val nonSpamWordCount = nonSpamFreqs.size.toDouble
    val N_notSpam = new File(nonSpamTrainingPath).listFiles.size.toDouble

    val spamDenominator: Double = spamWordCount + (dictSize + 1).toDouble * alpha
    val nonSpamDenominator: Double = nonSpamWordCount + (dictSize + 1).toDouble * alpha
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

  /**
  * Computes the probability that a given message is spam:
  * P(spam|doc) \simeq P(doc|spam) * P(spam)
  * P(~spam|word) \simeq P(doc|~spam) * P(~spam)
  * P(doc|c) = \prod_w P(w|c), c = {spam, ~spam}, product over words in doc
  * here, P(x) = log P(x), P(x|y) = log P(x|y)
  */
  def predict(classification: String, testSetPath: String, p_word_spam: Map[String, Double], 
    p_word_notSpam: Map[String, Double], pSpam: Double, pNotSpam: Double): (Double, Double) = {
    
    val files = new File(testSetPath).listFiles.iterator 
    var hits: Int = 0
    var misses: Int = 0

    files.foreach(f => {
      var p_spam: Double = 0
      var p_notSpam: Double = 0
      val prob_file: Double = 0
      val msg = buildFileDict(f.toString)

      msg.foreach(w => {
        p_spam += p_word_spam.getOrElse(w._1, pUnknownWord_spam) * w._2.toDouble
        p_notSpam += p_word_notSpam.getOrElse(w._1, pUnknownWord_nonSpam) * w._2.toDouble
      })

      p_spam += pSpam
      p_notSpam += pNotSpam
      val prediction = if (p_spam > p_notSpam) "spam" else "not spam"
      if (prediction equals classification) hits += 1 else misses += 1      
    })

    //println(classification + ": hits: " + hits + ", misses: " + misses 
    //  + " => % Correct: " + hits.toDouble / (hits + misses).toDouble)
    (hits.toDouble, misses.toDouble)
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

    println("| Alpha | Total Correct | Spam Correct | Non-Spam Correct |")
    println("| ------- | ------- | ------- | ------- |")
    var alpha:Double = 1.0
    for(x <- 1 to 40){
      alpha *= 0.5
      val nb = new NaiveBayes(alpha)
      val wordList = nb.buildVocabulary(allTrainingData)
      val training = nb.train(spam_train, nonspam_train, wordList.size)

      val spam = nb.predict("spam", spam_test, training._1, training._2, training._3, training._4)
      val nonSpam = nb.predict("not spam", nonspam_test, training._1, training._2, training._3, training._4)
      val N = spam._1 + spam._2 + nonSpam._1 + nonSpam._2
      val spamCorrect = spam._1 / (spam._1 + spam._2)
      val nonSpamCorrect = nonSpam._1 / (nonSpam._1 + nonSpam._2)
      println("| " + alpha + " | " + ((spam._1 + nonSpam._1) / N) + " | " + spamCorrect + " | " + nonSpamCorrect + " |")
    }
  }
}

