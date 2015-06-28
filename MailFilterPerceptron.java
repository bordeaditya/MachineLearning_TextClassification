/*	
 * Author Name : Aditya Borde
 * Function : Mail Filter Using Perceptron
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;


// Word data class
class WordData
{
	double wordCount =0;
	double tempWeight =0 ,weight =0;
	static double weightZero = 0;
	HashMap<String,Double> mapfileNameCount = new HashMap<String, Double>();
	
	
	public double getWordCount() {
		return wordCount;
	}
	public void setWordCount(double wordCount) {
		this.wordCount = wordCount;
	}
	
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
}

public class MailFilterPerceptron {
	
	public static int iterationCount = 300;
	public static HashMap<String,WordData> MapWordFileNameCount = new HashMap<String, WordData>();
	public final static double Eta = 0.3;
	
	static String dir_training,dir_test,dir_Spam_Training,dir_Ham_Training,dir_Spam_Test,dir_Ham_Test;
	
	static int totalSpamTests,positiveSpam,totalHamTests,positiveHam;
	
	// Stop Words List
	static ArrayList<String> stoppingWords = new ArrayList<String>();
	
	static ArrayList<String> allFilesInSpam_Training = new ArrayList<String>();
	static ArrayList<String> allFilesInHam_Training = new ArrayList<String>();
	
	
	// Main Function
	public static void main(String[] args) throws IOException {
		
		dir_training = args[0];
		dir_test = args[1];
		String stopWordFile = args[2];
		
		long startTime = System.currentTimeMillis();
		// Fill the Stop Words in the List
		FillStoppingWordsList(stopWordFile);
		
		// Set Training File Path
		SetPaths(dir_training,true);
		
		// Set Test File Path
		SetPaths(dir_test,false);
		
		// Fill the Map With Word File Name and Count
		FillMapInformation(MapWordFileNameCount);
		
		// Assign Random Weights to keys
		AssignRandomWeights(MapWordFileNameCount);
		
		int i = 0;
		
		while(i != iterationCount)
		{
			System.out.print("*");
			i++;
			// Perceptron Learning for SPAM
			PerceptronLearning(MapWordFileNameCount,allFilesInSpam_Training,true);
			// Perceptron Learning for HAM
			PerceptronLearning(MapWordFileNameCount,allFilesInHam_Training,false);
		}
		
		positiveSpam = 0; positiveHam =0;
		double spamAccuracy = CalculateAccuracyPTR(MapWordFileNameCount,dir_Spam_Test,true,true);
		double hamAccuracy = CalculateAccuracyPTR(MapWordFileNameCount,dir_Ham_Test,false,true);
		
		System.out.println("\n*** Perceptron Accuracy Before Removing Stop Words ***");
		
		System.out.println(" Perceptron Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Perceptron Accuracy in Ham :" + hamAccuracy + "%");
		
		double totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		System.out.println(" Perceptron Accuracy (Overall) :" + totalAccuracy + "%");
		
		/*
		 * After Removing Stop Words 
		 */
		
		RemoveStopWords(MapWordFileNameCount);
		AssignRandomWeights(MapWordFileNameCount);
		
		i =0;
		while(i != iterationCount)
		{
			System.out.print("*");
			i++;
			// Perceptron Learning for SPAM
			PerceptronLearning(MapWordFileNameCount,allFilesInSpam_Training,true);
			// Perceptron Learning for HAM
			PerceptronLearning(MapWordFileNameCount,allFilesInHam_Training,false);
		}
		
		positiveSpam = 0; positiveHam =0;
		spamAccuracy = CalculateAccuracyPTR(MapWordFileNameCount,dir_Spam_Test,true,true);
		hamAccuracy = CalculateAccuracyPTR(MapWordFileNameCount,dir_Ham_Test,false,true);
		
		System.out.println("\n*** Perceptron Accuracy After Removing Stop Words ***");
		
		System.out.println(" Perceptron Accuracy in Spam :" + spamAccuracy + "%");
		System.out.println(" Perceptron Accuracy in Ham :" + hamAccuracy + "%");
		
		totalAccuracy = ((double)(positiveHam + positiveSpam)/(totalHamTests + totalSpamTests))* 100;
		System.out.println(" Perceptron Accuracy (Overall) :" + totalAccuracy + "%");

		long endTime = System.currentTimeMillis();
		System.out.println("Time = " +(endTime-startTime));
	}
	
	
	// Remove Stop Words From the Map
	private static void RemoveStopWords(HashMap<String, WordData> map) {
		
		for(String stopWord:stoppingWords)
		{
			if(map.containsKey(stopWord))
				map.remove(stopWord);
		}
		
	}


	// Calculate Accuracy using Perceptron
	public static double CalculateAccuracyPTR(HashMap<String, WordData> map,String dir_Name, boolean isSpam, boolean isNormalMap) throws IOException {
		
		HashMap<String,Double> testWordCount = new HashMap<String, Double>();
		
		File testData = new File(dir_Name);
		ArrayList<String> paths = GetAbsolutePaths(testData.listFiles());
		
		double accuracy =0,activation =0 ;
		
		if(isSpam)
			totalSpamTests = paths.size();
		else
			totalHamTests = paths.size();
		
		//Keys in Training file :
		Set<String> wordsInTrainingMap = map.keySet();
		
		for(String path : paths)
		{
			activation =0; 
			//File file = temp.getAbsoluteFile();
			testWordCount.clear();
			
			BufferedReader br = new BufferedReader(new FileReader(path));
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
					String wordInLower = word.trim().toLowerCase();
					if(!wordInLower.equalsIgnoreCase(""))
					{
						if(!testWordCount.containsKey(wordInLower))
							testWordCount.put(wordInLower, 1.0);
						else // Word Already exist
							testWordCount.put(wordInLower,testWordCount.get(wordInLower)+1);
					}
				}	
			}
			br.close();
			
			Set<String> wordsInFile = testWordCount.keySet();
			
			// Every Word in Test File
			for(String wordInTestFile: wordsInFile)
			{	
				// if that word is in MAP
				if(wordsInTrainingMap.contains(wordInTestFile))
				{
					WordData tempData = map.get(wordInTestFile);
					activation = activation + (tempData.getWeight() * testWordCount.get(wordInTestFile));
				}
			}
			activation = activation + WordData.weightZero;
			
			if(isSpam)
			{
				if(Double.compare(activation, 0.0) > 0)
					positiveSpam ++;
			}
			else
			{
				if(Double.compare(activation, 0.0) < 0)
					positiveHam ++;
			}
				
		}
		
		if(isSpam)
			accuracy = ((double)positiveSpam / totalSpamTests)* 100;
		else
			accuracy = ((double)positiveHam / totalHamTests)* 100;
		
		return accuracy;
	}


	// Learning using Perceptron
	public static void PerceptronLearning(HashMap<String, WordData> map, ArrayList<String> allFiles, boolean isSpam) {
		
		double classValue = 0, target = 0;
		double perceptronOutput = 0;
		
		if(isSpam)
			target = 1;
		else
			target = -1;
		
		for(String path: allFiles)
		{
			classValue = CalculateClass(map,path,target);
			
			if(classValue > 0)
				perceptronOutput = 1;
			else
				perceptronOutput = -1;
			
			if(Double.compare(perceptronOutput, target)!=0)
				UpdateWeights(map,path,target,perceptronOutput);
		}
	}

	// Update weights 
	public static void UpdateWeights(HashMap<String, WordData> map,String path, double target, double perceptronOutput) {
		
		Set<String> keySet = map.keySet();
		double tempWeightValue = 0;
		// Update weight Zero
		WordData.weightZero = WordData.weightZero + target;
		
		for(String key:keySet)
		{
			WordData temp = map.get(key);
			if(temp.mapfileNameCount.get(path)!=null)
			{
				tempWeightValue = temp.getWeight() + (Eta * (target - perceptronOutput) * (temp.mapfileNameCount.get(path)));
				temp.setWeight(tempWeightValue);
			}
		}
		
	}

	// Calculate Class value:
	public static double CalculateClass(HashMap<String, WordData> map, String path, double target) {
		
		double activation =0;
		Set<String> keySet = map.keySet();
		
		for(String key:keySet)
		{
			WordData temp = map.get(key);
			if(temp.mapfileNameCount.get(path)!=null)
				activation = activation + (temp.getWeight() * temp.mapfileNameCount.get(path));
		}
		return (activation + WordData.weightZero);
		
	}
	
	
	// Assign Random Weights :
	public static void AssignRandomWeights(HashMap<String, WordData> mapWeightAssignment) {
		
		// Assign weight W0
		WordData.weightZero = GenerateRandom();
		// Assign all weights
		Set<String> keySet = mapWeightAssignment.keySet();
		for(String key: keySet)
		{
			WordData temp = mapWeightAssignment.get(key);
			double random = GenerateRandom();
			temp.setWeight(random);
			
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
	
	// Fill the Map Information
	private static void FillMapInformation(HashMap<String, WordData> map) throws IOException {
		
		File trainData = new File(dir_Spam_Training);
		
		ArrayList<String> pathSpam = GetAbsolutePaths(trainData.listFiles());
		
		trainData = new File(dir_Ham_Training);
		ArrayList<String> pathHam = GetAbsolutePaths(trainData.listFiles());
		
		// Fill the All Paths List
		allFilesInSpam_Training.addAll(pathSpam);
		allFilesInHam_Training.addAll(pathHam);
		
		// For Every File in SPAM
		FillMapKeyFileNameCount(pathSpam,map);
		
		// For Every File in HAM
		FillMapKeyFileNameCount(pathHam,map);
	}
	
	// Fill the Map
	private static void FillMapKeyFileNameCount(ArrayList<String> paths, HashMap<String, WordData> map) throws IOException {
		
		BufferedReader br = null;
		for(String tempFilePath : paths)
		{
			br = new BufferedReader(new FileReader(tempFilePath));
			//wordCountInFile = 0;
			String line ="";
			while((line = br.readLine())!= null)
			{
				line = line.replaceAll("[^\\w\\s]", "");
				String[] words = line.split(java.util.regex.Pattern.quote(" "));
			
				// Get the latest word Set
				for(String word: words)
				{
					String wordInLower = word.trim().toLowerCase();
					Set<String> keys = map.keySet();
					WordData tempWordData;
					// Skip Blank letter
					if(!wordInLower.equalsIgnoreCase(""))
					{
						if(keys.contains(wordInLower)) // If Word already exist in MAP
						{
							tempWordData = map.get(wordInLower);
					
							// If File path is present in MAP against the KEY
							if(tempWordData.mapfileNameCount.get(tempFilePath)!= null)
							{
								double preValue = tempWordData.mapfileNameCount.get(tempFilePath);
								tempWordData.mapfileNameCount.put(tempFilePath, preValue + 1);
							}
							else // If file path is not present in MAP
							{
								tempWordData.mapfileNameCount.put(tempFilePath,1.0);
							}
								
						}
						else // If word is not present in MAP
						{
							tempWordData = new WordData();
							tempWordData.mapfileNameCount.put(tempFilePath,1.0);
							map.put(wordInLower, tempWordData);
						}
					}
				}
			}
			// End of File
			br.close();
		}
	}

	// Get Absolute paths of every file in a directory
	public static ArrayList<String> GetAbsolutePaths(File[] files)
	{
		ArrayList<String> filePaths = new ArrayList<String>();
		for(File file: files)
		{
			String path = file.getAbsolutePath();
			filePaths.add(path);
		}
		return filePaths;
	}

	// Set the Directory paths
	// Set the file paths of Training and test Data
	private static void SetPaths(String dir_Name, boolean isTrain) {
		
		// Get the directory Files:
		File file = new File(dir_Name);
		File[] dirNames = file.listFiles();
		
		// Get the directory Names
		for(File dir: dirNames)
		{
			// If file is SPAM
			if(dir.getAbsolutePath().contains("spam"))
			{
				if(isTrain)
					dir_Spam_Training = dir.getAbsolutePath();
				else
					dir_Spam_Test = dir.getAbsolutePath();
			}	
			else if(dir.getAbsolutePath().contains("ham"))
			{
				if(isTrain)
					dir_Ham_Training = dir.getAbsolutePath();
				else
					dir_Ham_Test = dir.getAbsolutePath();
			}
				
		}
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

}
