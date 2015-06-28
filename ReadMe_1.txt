Name: Aditya Borde


************Mail Filter*********

Please run the program using following command on Command prompt:

1) Compiling the Java file: .\javac program.java
.\javac MailFilter.java

2) Running the Java Code: .\java program <training_set> <test_set> <stopWords_Text_File>
*above command contains Sequence of Command line arguments:


<training_set> : It contains the path of folder containing only SPAM and HAM folders Set of Training Data.
** It is necessaray that it should contain the SPAM and HAM folders but not the SPAM and HAM files

<test_set> : It contains the path of folder containing only SPAM and HAM folders Set of Test Data.
** It is again necessaray that it should contain the SPAM and HAM folders of Test Data, but not the SPAM and HAM files

<stopWords_Text_File> : Path of the text file containing Stop words.
In this file Word seperator is new line. So, every word must be on new line.
This file has been provided in the zip submission.


Sample command :
.\java MailFilter C:\\data\\Training C:\\data\\Test C:\\data\\stopWords.txt


**Please note : above "C:\\data\\Training" and "C:\\data\\Test" contains only
SPAM and HAM folders of respective sets. 
