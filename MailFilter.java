import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;


class WordData
{
	int spamOccurence =0,hamOccurence =0;
	double probGivenInHam =0,probGivenInSpam =0;
	double tempWeight =0 ,weight =0;
	static double weightZero = 0;
	static double weightZeroSkippedList = 0;
	HashMap<String,Integer> mapfileNameCount = new HashMap<String, Integer>();
	
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	public double getTempWeight() {
		return tempWeight;
	}
	public void setTempWeight(double tempWeight) {
		this.tempWeight = tempWeight;
	}
	public static double getWeightZero() {
		return weightZero;
	}
	public static void setWeightZero(double weightZero) {
		WordData.weightZero = weightZero;
	}
	public int getSpamOccurence() {
		return spamOccurence;
	}
	public void setSpamOccurence(int spamOccurence) {
		this.spamOccurence = spamOccurence;
	}
	
	public int getHamOccurence() {
		return hamOccurence;
	}
	
	public void setHamOccurence(int hamOccurence) {
		this.hamOccurence = hamOccurence;
	}
	public double getProbGivenInSpam() {
		return probGivenInSpam;
	}
	public void setProbGivenInSpam(double probGivenInSpam) {
		this.probGivenInSpam = probGivenInSpam;
	}
	public double getProbGivenInHam() {
		return probGivenInHam;
	}
	public void setProbGivenInHam(double probGivenInInHam) {
		this.probGivenInHam = probGivenInInHam;
	}
	
}


public class MailFilter {
	
	static int updateCount = 100;
	static double Eta = 0.0008,Lambda = 0.0003; 
	static double probNewWord_Spam,probNewWord_Ham;
	static ArrayList<String> stoppingWords = new ArrayList<String>();
	static String dir_training,dir_test,dir_Spam_Training,dir_Ham_Training,dir_Spam_Test,dir_Ham_Test;
	
	static HashMap<String,WordData> MapWordInformation = new HashMap<String,WordData>();
	static HashMap<String,WordData> MapSkippingStopWords = new HashMap<String,WordData>();
	
	static HashMap<String,Double> MapFileProbability = new HashMap<String, Double>();
	
	static int wordCount_Spam,wordCount_Ham,wordCount_Skipped_Spam,wordCount_Skipped_Ham;
	
	// Class count in training data
	static int spam_class = 0,ham_class = 0;
	
	// Probabilities of each classes
	static double probOfSpamClass = 0, probOfHamClass =0, totalAccuracy; 
	
	static int totalSpamTests,positiveSpam,totalHamTests,positiveHam;
	
	public static void main(String[] args) throws IOException {
		// Get the directories of SPAM and HAM
		dir_training = args[0];
		dir_test = args[1];
		String stopWordFile = args[2];
		
		FillStoppingWordsList(stopWordFile);
		
		// Get the directory Files:
		File file = new File(dir_training);
		File[] dirNames = file.listFiles();
		
		// Get the directory Names
		for(File dir: dirNames)
		{
			
			if(dir.getAbsolutePath().contains("spam"))
				dir_Spam_Training = dir.getAbsolutePath();
			else if(dir.getAbsolutePath().contains("ham"))
				dir_Ham_Training = dir.getAbsolutePath();
		}
		//System.out.println(dir_Ham_training +" : "+ dir_Spam_Training);
		
		// Fill the data For SPAM 
		FillMapWordInformation(true);
		
		// Fill the data for HAM
		FillMapWordInformation(false);
		
		// Calculate Probability of word for Given class in SPAM or HAM
		CalculateProbabilities();
		
		// Set SPAM , HAM probabilities
		SetClassProbabilities();
		
		
		// Get the directory Files:
		file = new File(dir_test);
		File[] dirTestNames = file.listFiles();
		
		// Get the directory Names
		for(File dir: dirTestNames)
		{
			if(dir.getAbsolutePath().contains("spam"))
				dir_Spam_Test = dir.getAbsolutePath();
			else if(dir.getAbsolutePath().contains("ham"))
				dir_Ham_Test = dir.getAbsolutePath();
		}
		
		// Calculate Accuracy using Naive Bayes:
		double spamAccuracy = CalculateAccuracyNB(MapWordInformation,dir_Spam_Test,true,true);
		double hamAccuracy = CalculateAccuracyNB(MapWordInformation,dir_Ham_Test,false,true);
		
		// Before Removing Stop Words
		System.out.println("\n*** Naive Bayes Accuracy Before Removing Stop Words ***");
		
		System.out.println(" Naive Bayes Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Naive Bayes Accuracy in Ham :" + hamAccuracy + "%");
		
		totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		
		System.out.println(" Naive Bayes Accuracy Overall :" + totalAccuracy + "%");
		
		
		System.out.println("\n*** Naive Bayes Accuracy After Removing Stop Words ***");
		
		// After Removing Stop Words
		// Set positive counters to Zero
		positiveSpam = 0; positiveHam =0;
		spamAccuracy = CalculateAccuracyNB(MapSkippingStopWords,dir_Spam_Test,true,false);
		hamAccuracy = CalculateAccuracyNB(MapSkippingStopWords,dir_Ham_Test,false,false);
		
		System.out.println(" Naive Bayes Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Naive Bayes Accuracy in Ham :" + hamAccuracy + "%");
		
		totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		
		System.out.println(" Naive Bayes Accuracy Overall :" + totalAccuracy + "%");
		
		
		/**
		 * 
		 * **********Logistic Regression Before Removing Stopping Words
		 * 
		 **/
		
		long startTime = System.currentTimeMillis();
		// Assign Random Weights To all Keywords
		AssignRandomWeights(MapWordInformation);
		
		// Calculate Word Counts of Each key in every file
		CalculateWordCountInFile(MapWordInformation);
		
		// Calculate Weights For Every key in Map
		for(int i=0;i<updateCount;i++)
			CalculateWeightsForKey(MapWordInformation);
		
		positiveSpam = 0; positiveHam =0;
	
		spamAccuracy = CalculateAccuracyLR(MapWordInformation,dir_Spam_Test,true,true);
		hamAccuracy = CalculateAccuracyLR(MapWordInformation,dir_Ham_Test,false,true);
		
		System.out.println("\n*** Logistic Regression Accuracy Before Removing Stop Words ***");
		
		System.out.println(" Logistic Regression Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Logistic Regression Accuracy in Ham :" + hamAccuracy + "%");
		
		totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		
		System.out.println(" Logistic Regression Accuracy Overall :" + totalAccuracy + "%");
		
		/**
		 * 
		 * **********Logistic Regression After Removing Stopping Words
		 * 
		 **/
		
		positiveSpam = 0; positiveHam =0;
		spamAccuracy = CalculateAccuracyLR(MapWordInformation,dir_Spam_Test,true,false);
		hamAccuracy = CalculateAccuracyLR(MapWordInformation,dir_Ham_Test,false,false);
		
		System.out.println("\n*** Logistic Regression Accuracy After Removing Stop Words ***");
		
		System.out.println(" Logistic Regression Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Logistic Regression Accuracy in Ham :" + hamAccuracy + "%");
		
		totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		
		System.out.println(" Logistic Regression Accuracy Overall :" + totalAccuracy + "%");
		
		long endTime = System.currentTimeMillis();
		long time = (endTime - startTime);
		System.out.println("\n   Time = "+ time);
		
	}
	
	
	// Assign Random Weights :
	public static void AssignRandomWeights(HashMap<String, WordData> mapWeightAssignment) {
		
		// Assign weight W0
		WordData.weightZero = GenerateRandom();
		WordData.weightZeroSkippedList = GenerateRandom();
		// Assign all weights
		Set<String> keySet = mapWeightAssignment.keySet();
		for(String key: keySet)
		{
			WordData temp = mapWeightAssignment.get(key);
			double random = GenerateRandom();
			temp.setWeight(random);
			
		}
	}
	
	// Calculate Weight Vector and update at the end
	public static void CalculateWeightsForKey(HashMap<String, WordData> map)
	{
		MapFileProbability.clear();
		FillMapFileProbability(map);
		CalculateWeights(map);
		System.out.print("*");
		UpdateWeights(map);
	}
	
	

	// Calculate Weights for each keyword
	public static void CalculateWeights(HashMap<String, WordData> map) {
		
		double probOfY = 0;
		File trainData = new File(dir_Spam_Training);
		File[] listOfFilesSpam = trainData.listFiles();
		
		trainData = new File(dir_Ham_Training);
		File[] listOfFilesHam = trainData.listFiles();
		Set<String> keySet = map.keySet();
		//int i =0;
		// In SPAM folder
		for(String key:keySet)
		{
			probOfY = 1;
			double summation = 0,countInFile =0,probOfKeyInFiles=0;
			WordData tempMapData = map.get(key);
			for(File file : listOfFilesSpam)
			{
				//summation = 0;
				countInFile = 0;
				if(tempMapData.mapfileNameCount.get(file.getAbsolutePath())!=null)
					countInFile = (double)tempMapData.mapfileNameCount.get(file.getAbsolutePath());
				probOfKeyInFiles = MapFileProbability.get(file.getAbsolutePath());
				summation = summation + (countInFile * (probOfY - probOfKeyInFiles));
			}
			
			probOfY = 0;
			// IN HAM folder
			for(File file : listOfFilesHam)
			{
				//summation = 0;
				countInFile = 0;
				if(tempMapData.mapfileNameCount.get(file.getAbsolutePath())!=null)
					countInFile = (double)tempMapData.mapfileNameCount.get(file.getAbsolutePath());
				probOfKeyInFiles = MapFileProbability.get(file.getAbsolutePath());
				summation = summation + (countInFile * (probOfY - probOfKeyInFiles));
			}
			
			double tempWeight = tempMapData.getWeight() + (Eta * summation) - (Eta * Lambda * tempMapData.getWeight());
			tempMapData.setTempWeight(tempWeight);
		}
	}


	//Fill HashMap of each word - HashMap <FileName><wordCount>
	public static void FillMapFileProbability(HashMap<String, WordData> map)
	{
		File trainingDataSpamFiles  = new File(dir_Spam_Training);
		File trainingDataHamFiles = new File(dir_Ham_Training);
		
		ArrayList<File> allFiles = new ArrayList<File>(Arrays.asList(trainingDataSpamFiles.listFiles()));
		allFiles.addAll(Arrays.asList(trainingDataHamFiles.listFiles()));
		
		// List of all files in SPAM and HAM folder
		//File[] listOfAllFiles = new File[allFiles.size()];
		//listOfAllFiles = allFiles.toArray(listOfAllFiles);
		Set<String> mapKeySet = map.keySet();
		for(File file:allFiles)
		{
			double probValue = 0;
			for(String key : mapKeySet)
			{
				WordData data = map.get(key);
				double count = 0;
				if(data.mapfileNameCount.get(file.getAbsolutePath())!= null)
					count = (double)data.mapfileNameCount.get(file.getAbsolutePath());
				
				probValue = probValue + count * data.getWeight();
			}
			probValue = probValue + WordData.weightZero;
			
			probValue = CalculateSigmoid(probValue);
			
			// Fill the Probability Map
			MapFileProbability.put(file.getAbsolutePath(), probValue);
		}
	}
	
	// Calculate Probabilities of word in each file:
	public static void CalculateWordCountInFile(HashMap<String, WordData> mapWeightAssignment) throws IOException {
		
		Set<String> keySet = mapWeightAssignment.keySet();
		// Get all the files in SPAM And HAM folder:
		File trainingDataSpamFiles  = new File(dir_Spam_Training);
		File trainingDataHamFiles = new File(dir_Ham_Training);
		
		ArrayList<File> allFiles = new ArrayList<File>(Arrays.asList(trainingDataSpamFiles.listFiles()));
		allFiles.addAll(Arrays.asList(trainingDataHamFiles.listFiles()));
		
		// List of all files in SPAM and HAM folder
		// File[] listOfAllFiles = new File[allFiles.size()];
		// listOfAllFiles = allFiles.toArray(listOfAllFiles);
			
		// Read each file in the Training Directory
		for(File tempFile : allFiles)
		{
			File file = tempFile.getAbsoluteFile();
			BufferedReader br = new BufferedReader(new FileReader(file));
			//wordCountInFile = 0;
			String line ="";
			while((line = br.readLine())!= null)
			{
				line = line.replaceAll("[^\\w\\s]", "");
				String[] words = line.split(java.util.regex.Pattern.quote(" "));
				// Get the latest word Set
				for(String word: words)
				{
					if(keySet.contains(word.toLowerCase()))
					{
						WordData tempData = mapWeightAssignment.get(word.toLowerCase());
						Set<String> fileSet = tempData.mapfileNameCount.keySet();
						if(!fileSet.contains(file.getAbsolutePath()))
							tempData.mapfileNameCount.put(file.getAbsolutePath(), 1);
						else
						{
							int cntVal = tempData.mapfileNameCount.get(file.getAbsolutePath());
							tempData.mapfileNameCount.put(file.getAbsolutePath(), cntVal+1);
						}
					}
				}
			}
			
			// After End of the file Read:
			br.close();
		}
	}

	
	// Update the weight vector with new Weights
	public static void UpdateWeights(HashMap<String, WordData> map)
	{
		Set<String> keySet = map.keySet();
		for(String key:keySet)
		{
			WordData mapWord = map.get(key);
			// Assign tempWeight value to Weight value
			mapWord.setWeight(mapWord.getTempWeight());
		}
		
	}
	
	// Generate Random Weight : Number
	public static double GenerateRandom()
	{
		double defaultNum = 0.5;
		Random rNo = new Random();
		double randomNumber = -1 + (1 - (-1)) * rNo.nextDouble();
		if(Double.compare(randomNumber, 0)!=0)
			return randomNumber;
		else
			return defaultNum;
	}
	
	// Fill the Stopping words in the List
	public static void FillStoppingWordsList(String stopWordFile) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(stopWordFile));
		String line;
		// Fill the List
		while((line = br.readLine())!= null)
		{
			// Remove Special Characters: Before adding the word
			line = line.replaceAll("[^\\w\\s]", "");
			stoppingWords.add(line.trim().toLowerCase());
		}
		br.close();
	}
	

	// Calculate Logistic Regression Accuracy 
	public static double CalculateAccuracyLR(HashMap<String, WordData> mapInfo,String path, boolean isSpam, boolean isWithoutStopWords) throws IOException {
		double accuracy = 0;
		
		HashMap<String,Double> testWordCount = new HashMap<String, Double>();
		File testData = new File(path);
		double classifySpam = 0,classifyHam =0;
		File[] listOfFiles = testData.listFiles();
		
		if(isSpam)
			totalSpamTests = listOfFiles.length;
		else
			totalHamTests = listOfFiles.length;
		
		//Keys in Training file :
		Set<String> wordsInTrainingMap = mapInfo.keySet();
		
		for(File temp : listOfFiles)
		{
			File file = temp.getAbsoluteFile();
			testWordCount.clear();
			
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line ="";

			while((line = br.readLine())!= null)
			{
				String wordsCollection = "";
				// Remove Special characters in the string
				wordsCollection = line.replaceAll("[^\\w\\s]", "");
				String[] words = wordsCollection.split(java.util.regex.Pattern.quote(" "));
				// Get the latest word Set
				Set<String> fileSpecificWords = testWordCount.keySet();
				for(String word: words)
				{
					String wordInLower = word.toLowerCase();
					if(isWithoutStopWords)// Without Removing Stop words
					{
						if(!fileSpecificWords.contains(wordInLower))
							testWordCount.put(wordInLower, 1.0);
						else // Word Already exist
							testWordCount.put(wordInLower,testWordCount.get(wordInLower)+1);
					}
					else // Removing Stop words
					{
						// If it is not a stop word
						if(!stoppingWords.contains(wordInLower))
						{
							if(!fileSpecificWords.contains(wordInLower))
								testWordCount.put(wordInLower, 1.0);
							else // Word Already exist
								testWordCount.put(wordInLower,testWordCount.get(wordInLower)+1);
						}
					}
				}
					
			}
			
			double sum = 0,weight =0;
			sum = WordData.weightZero;
			Set<String> wordsInFile = testWordCount.keySet();
			
			for(String wordInTestFile: wordsInFile)
			{
				if(wordsInTrainingMap.contains(wordInTestFile))
				{
					weight =0;
					WordData tempData = mapInfo.get(wordInTestFile);
					weight = tempData.getWeight() * testWordCount.get(wordInTestFile);
					sum = sum + weight;
				}
			}
		
			classifySpam = CalculateSigmoid(sum);
			classifyHam = 1 - classifySpam;	
			if(isSpam)
			{
				if(Double.compare(classifySpam, classifyHam)>0)
					positiveSpam++;
			}
			else
			{	
				if(Double.compare(classifyHam, classifySpam)>0)
					positiveHam++;
			}
			br.close();
		}
		
		if(isSpam)
			accuracy = ((double)positiveSpam / totalSpamTests)* 100;
		else
			accuracy = ((double)positiveHam / totalHamTests)* 100;
		
		return accuracy;

	}


	// Return Sigmoid of Value
	public static double CalculateSigmoid(double sum) {
		if(sum > 100)
			return 1.0;
		else if(sum < -100)
			return 0.0;
		else
			return (Math.exp(sum)/(1+ Math.exp(sum)));
	}


	// Accuracy using NB
	public static double CalculateAccuracyNB(HashMap<String, WordData> mapToCheckAccuracy, String path,boolean isSpam,boolean isNormalMap) throws IOException {
		
		double accuracy = 0; 
		File testData = new File(path);
		
		File[] listOfFiles = testData.listFiles();
		
		if(isSpam)
			totalSpamTests = listOfFiles.length;
		else
			totalHamTests = listOfFiles.length;
		
		double logSpamClassProb = probOfSpamClass;
		double logHamClassProb = probOfHamClass;
		
		for(File temp : listOfFiles)
		{
			double logProbSpam = 0;
			double logProbHam = 0;
			
			File file = temp.getAbsoluteFile();
			BufferedReader br = new BufferedReader(new FileReader(file));
			
			String line ="";
			while((line = br.readLine())!= null)
			{
				String wordsCollection = "";
				
				// Remove Special characters in the string
				wordsCollection = line.replaceAll("[^\\w\\s]", "");
				String[] words = wordsCollection.split(java.util.regex.Pattern.quote(" "));
				
				// Get the latest word Set
				Set<String> wordSet = mapToCheckAccuracy.keySet();
				for(String word: words)
				{
					// If mail is "SPAM"
					String wordInLower = word.trim().toLowerCase();
					if(!wordSet.contains(wordInLower)) // If Word is already present in Info.
					{
						if(!isNormalMap)
						{
							if(!stoppingWords.contains(wordInLower))
							{
								logProbSpam = logProbSpam + probNewWord_Spam;
								logProbHam = logProbHam + probNewWord_Ham;
							}
						}
						else
						{
							logProbSpam = logProbSpam + probNewWord_Spam;
							logProbHam = logProbHam + probNewWord_Ham;
						}
					}
					else
					{
						if(!isNormalMap)
						{
							if(!stoppingWords.contains(wordInLower))
							{
								WordData wordData = mapToCheckAccuracy.get(wordInLower);
								logProbSpam = logProbSpam + wordData.getProbGivenInSpam();
								logProbHam = logProbHam + wordData.getProbGivenInHam();
							}
						}
						else
						{
							WordData wordData = mapToCheckAccuracy.get(wordInLower);
							logProbSpam = logProbSpam + wordData.getProbGivenInSpam();
							logProbHam = logProbHam + wordData.getProbGivenInHam();
						}
					}
				}
				
			} // End of read line
			br.close();
			
			// After end of the a particular File scanning:
			double classifySpam = (logProbSpam + logSpamClassProb);
			double classifyHam = (logProbHam + logHamClassProb);
			
			if(isSpam)
			{
				if(Double.compare(classifySpam, classifyHam)>0)
					positiveSpam++;
			}
			else
			{
				if(Double.compare(classifyHam, classifySpam)>0)
					positiveHam++;
			}
		}
		
		if(isSpam)
			accuracy = ((double)positiveSpam / totalSpamTests)* 100;
		else
			accuracy = ((double)positiveHam / totalHamTests)* 100;
		
		return accuracy;
	}

	// Set the class probabilities:
	public static void SetClassProbabilities() {
		probOfSpamClass = Math.log(((double)spam_class)/(spam_class + ham_class));
		probOfHamClass = Math.log(((double)ham_class)/(spam_class + ham_class));
	}


	// Calculate the Probability of each word given it is in SPAM or in HAM
	public static void CalculateProbabilities() {
		
		Set<String> mapWordSet = MapWordInformation.keySet();
		
		// Calculate the probabilities
		for(String word : mapWordSet)
		{
			WordData tempData = MapWordInformation.get(word);
			tempData.probGivenInSpam = Math.log(((double)tempData.getSpamOccurence() + 1)/ (wordCount_Spam + MapWordInformation.size()));
			tempData.probGivenInHam = Math.log(((double)tempData.getHamOccurence() + 1)/ (wordCount_Ham + MapWordInformation.size()));
		}
		
		probNewWord_Spam =  Math.log(1 /(double)(wordCount_Spam + MapWordInformation.size()));
		probNewWord_Ham =  Math.log(1 /(double)(wordCount_Ham + MapWordInformation.size()));
		
		Set<String> mapNewWordSet = MapSkippingStopWords.keySet();
		//Calculate probabilities For New MAP
		for(String word : mapNewWordSet)
		{
			WordData tempData = MapSkippingStopWords.get(word);
			tempData.probGivenInSpam = Math.log(((double)tempData.getSpamOccurence() + 1)/ (wordCount_Skipped_Spam + MapSkippingStopWords.size()));
			tempData.probGivenInHam = Math.log(((double)tempData.getHamOccurence() + 1)/ (wordCount_Skipped_Ham + MapSkippingStopWords.size()));
		}
	}

	// Fill the Map WordInformation.
	public  static void FillMapWordInformation(boolean isSpam) throws IOException {
		
		File trainingData;
		
		if(isSpam)
			trainingData = new File(dir_Spam_Training);
		else
			trainingData = new File(dir_Ham_Training);
		
		File[] listOfFiles = trainingData.listFiles();
		
		// Set the class count
		if(isSpam)
			spam_class = listOfFiles.length;
		else 
			ham_class = listOfFiles.length;
		
		// Read each file in the Training Directory
		for(File temp : listOfFiles)
		{
			File file = temp.getAbsoluteFile();
			BufferedReader br = new BufferedReader(new FileReader(file));
			
			String line ="";
			while((line = br.readLine())!= null)
			{
				String wordsCollection = "";
				
				// Remove Special characters in the string
				wordsCollection = line.replaceAll("[^\\w\\s]", "");
				String[] words = wordsCollection.split(java.util.regex.Pattern.quote(" "));
				
				// Get the latest word Set
				for(String word: words)
				{
					if(!word.equalsIgnoreCase(""))
					{
						String wordInLower = word.trim().toLowerCase();
						
						FillKeyValuePairInMap(wordInLower,MapWordInformation,isSpam,true);
							
						//if word contains in Stop words list : Skip the Word.
						if(!stoppingWords.contains(wordInLower))
							FillKeyValuePairInMap(wordInLower,MapSkippingStopWords,isSpam,false);								
							
					}// END of if(word.equals("")) 
					
				}// End of Foreach - words

			} // End of read line
			br.close();
		}// End of Foreach File

		
	}


	// Fill the Maps
	public static void FillKeyValuePairInMap(String wordInLower,HashMap<String, WordData> mapToFill, boolean isSpam, boolean isNormalMap) {
		
		Set<String> wordSet = mapToFill.keySet();
		
		if(isSpam) // If it is SPAM
		{
			if(isNormalMap)
				wordCount_Spam++;
			else
				wordCount_Skipped_Spam++;
			
			if(wordSet.contains(wordInLower)) // If Word is already present in Info.
			{
				WordData wordData = mapToFill.get(wordInLower);
				wordData.setSpamOccurence((wordData.getSpamOccurence()) + 1);
			}
			else // Add the word in WordInfo
			{
				WordData tempWord = new WordData();
				tempWord.setSpamOccurence(1);
				// All files are SPAM:
				mapToFill.put(wordInLower, tempWord);
			}
		}
		else // If it is HAM
		{
			if(isNormalMap)
				wordCount_Ham++;
			else
				wordCount_Skipped_Ham++;
			
			if(wordSet.contains(wordInLower)) // If Word is already present in Info.
			{
				WordData wordData = mapToFill.get(wordInLower);
				wordData.setHamOccurence((wordData.getHamOccurence()) + 1);
			}
			else // Add the word in WordInfo
			{
				WordData tempWord = new WordData();
				tempWord.setHamOccurence(1);
				// All files are SPAM:
				mapToFill.put(wordInLower, tempWord);
			}
		}	
	}
}
