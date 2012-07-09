naive_bayes_classifier
======================

This is simple Naive Bayes classifier, for labeling emails as spam or not spam. It's currently unfinished.

Right now, I'm building like this:

scalac -d classes src/NaiveBayes.scala

And running like this:

scala -classpath classes/ NaiveBayes /path/to/your/data/directory
