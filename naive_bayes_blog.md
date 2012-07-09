## Learning about and Implementing a Naive Bayes Classifier

I'm a physicist working with a bunch of computer scientists who specialize in machine learning. I don't even know enough machine learning to be dangerous.

First, I'm trying out basic spam detection. I wrote some simple Scala code and tested on the lemm-stop subset of the Lingspam Dataset. Punctuation, numbers, and the "Subject" header were deleted. Tab and newline characters were replaced by a single space. See http://csmining.org/index.php/ling-spam-datasets.html for more info.

Naive Bayes correctly labelled all the spam emails in the test set. But only labelled 16% of the non-spam emails in the test set. This simple algorithm would send a lot of valid emails into a junk mail folder.