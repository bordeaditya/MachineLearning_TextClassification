/*	
 * Author Name : Aditya Borde
 * Net Id : asb140930 	
 * Function : Collaborative Filtering for Netflix
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;


//Rating class for user for particular Movie
class Rating
{
	double avgRating = 0.0;
	HashMap<String, Double> mapMovieIdRating = new HashMap<String, Double>();
	
	public double getAvgRating() {
		return avgRating;
	}
	public void setAvgRating(double avgRating) {
		this.avgRating = avgRating;
	}
	
}

// Test Data Object
class TestData
{
	double rating;
	String movieId,userId;
	
	public double getRating() {
		return rating;
	}
	public void setRating(double rating) {
		this.rating = rating;
	}
	public String getMovieId() {
		return movieId;
	}
	public void setMovieId(String movieId) {
		this.movieId = movieId;
	}
	public String getUserId() {
		return userId;
	}
	public void setUserId(String userId) {
		this.userId = userId;
	}
	
}
public class CollaborativeFiltering {

	public static HashMap<String, Rating> MapUserIdRatings = new HashMap<String, Rating>();
	public static ArrayList<TestData> testData = new ArrayList<TestData>();
	
	public static double k = 1;
	public static String path_trainingSet,path_testSet;
	
	public static double meanAbsError =0.0, rootMeanSqError = 0.0;
	public static void main(String[] args) throws IOException {
		
		path_trainingSet = args[0];
		path_testSet = args[1];

		// Fill the Training Data
		FillMapUserIdRatings(MapUserIdRatings,path_trainingSet);
		
		// Calculate Average RATING
		CalculateAverageRating(MapUserIdRatings);
		
		// Fill the Test Data
		FillTestData(testData,path_testSet);
		meanAbsError = 0; rootMeanSqError =0;
		
		CalculatePrediction(MapUserIdRatings,testData);
		System.out.println("\n*** Collaborative Filtering ***");
		System.out.println(" Mean Absolute Error = " + meanAbsError);
		System.out.println(" Root Mean Squared Error = " + rootMeanSqError);
	}
	
	// Calculate Accuracy
	public static void CalculatePrediction(HashMap<String, Rating> mapUserIdRatings,ArrayList<TestData> testData) {
		
		// Every Record on test Set
		for(TestData testTemp : testData)
		{
			// For matched user Id
			if(mapUserIdRatings.containsKey(testTemp.getUserId()))
			{
				System.out.print("*");
				Rating movieIdRatings = mapUserIdRatings.get(testTemp.getUserId());
				double summation = CalculateSummation(mapUserIdRatings,testTemp.getUserId(),testTemp.getMovieId());
				double prediction_a_j =  movieIdRatings.getAvgRating() + (k * summation);
				
				if(Double.compare(prediction_a_j, testTemp.getRating())!=0)
				{
					meanAbsError = meanAbsError + Math.abs(prediction_a_j - testTemp.getRating());
					rootMeanSqError = rootMeanSqError + Math.pow((prediction_a_j - testTemp.getRating()), 2);
				}
			}
		}
		
		meanAbsError = meanAbsError/(double)testData.size();
		rootMeanSqError = rootMeanSqError/(double)testData.size();
		rootMeanSqError = Math.sqrt(rootMeanSqError);
	}

	// Calculate Summation Value in Formula (1)
	public static double CalculateSummation(HashMap<String, Rating> map, String userId, String movieId) {
		
		double weight = 0, sumValue =0, v_ij = 0, v_i_bar =0;
		Set<String> userIdSet = map.keySet();
		double weightSummation = 0;
		for(String tempUserId : userIdSet)
		{
			weight = GetWeightValue(map,userId,tempUserId);
			weightSummation = weightSummation + Math.abs(weight);
			Rating tempRating = map.get(tempUserId);
			v_ij = 0; v_i_bar =0;
			if(tempRating!=null)
			{
				if(tempRating.mapMovieIdRating.get(movieId)!=null)
				{
					v_ij = tempRating.mapMovieIdRating.get(movieId);
					v_i_bar = tempRating.getAvgRating();
					//weight = GetWeightValue(map,userId,tempUserId);
					//weightSummation = weightSummation + Math.abs(weight);
				}
			}
			
			sumValue = sumValue + weight * (v_ij - v_i_bar);
		}
		
		if(Double.compare(weightSummation,0)!=0)
			k = (1 / weightSummation);
		else
			k = 0;
		
		//System.out.println("SumValue="+ sumValue+"K="+k);
		return sumValue;
	}

	// Calculate Weight(a,i) : from (2)
	public static double GetWeightValue(HashMap<String, Rating> map,String userId_a, String userId_i) {
		Rating tempA = map.get(userId_a);
		Rating tempI = map.get(userId_i);
		double weightVal = 0;
		
		Set<String> movieIdFromA = new HashSet<String>(tempA.mapMovieIdRating.keySet());
		Set<String> movieIdFromI = new HashSet<String>(tempI.mapMovieIdRating.keySet());
		
		movieIdFromA.retainAll(movieIdFromI);
		
		double nR = 0, dR = 0, v_a_j = 0, v_a_bar = 0, v_i_j = 0, v_i_bar =0;
		double part1 = 0, part2 = 0;
		if(movieIdFromA.size()>0)
		{
			for(String commonMovie: movieIdFromA)
			{
				v_a_j = tempA.mapMovieIdRating.get(commonMovie);
				v_a_bar = tempA.getAvgRating();
				
				v_i_j = tempI.mapMovieIdRating.get(commonMovie);
				v_i_bar = tempI.getAvgRating();
				
				nR = nR + ((v_a_j - v_a_bar) * (v_i_j - v_i_bar));
				
				part1 = part1 + Math.pow((v_a_j - v_a_bar), 2);
				part2 = part2 + Math.pow((v_i_j - v_i_bar), 2);
				
			}
			dR = Math.sqrt(part1 * part2);
			
			if(Double.compare(dR, 0)==0)
				weightVal = 0;
			else
				weightVal = (nR/dR);
		}
		return weightVal;
	}

	// Calculate Average Rating
	public static void CalculateAverageRating(HashMap<String, Rating> mapUserRatings) {
		
		Set<String> userIds = mapUserRatings.keySet();
		
		for(String userId : userIds)
		{
			Rating tempR = mapUserRatings.get(userId);
			Set<String> movieIds = tempR.mapMovieIdRating.keySet();
			double total = (double)movieIds.size();
			if(Double.compare(total, 0.0)>0)
			{
				double rating = 0;
				for(String movieId : movieIds)
					rating = rating + tempR.mapMovieIdRating.get(movieId);
			
				double avgRating = (rating/total); 
				tempR.setAvgRating(avgRating);
			}
		}
	}

	// Fill the map from Training Set
	public static void FillMapUserIdRatings(HashMap<String, Rating> map, String path_training) throws IOException {

		BufferedReader br = new BufferedReader(new FileReader(path_training));
		String line ="";
		while((line = br.readLine())!= null)
		{
			String[] stringParts = line.split(java.util.regex.Pattern.quote(","));
			String movieId = stringParts[0].trim();
			String userId = stringParts[1].trim();
			// stringParts[0] - movieId
			// stringParts[1] - userId 
			// stringParts[2] - Rating
			double rating = Double.parseDouble(stringParts[2].trim());
			if(!map.containsKey(userId))
			{
				Rating temp = new Rating();
				temp.mapMovieIdRating.put(movieId, rating);
				map.put(userId, temp);
			}
			else
			{
				Rating tempRating = map.get(userId);
				
				if(tempRating.mapMovieIdRating.get(movieId)!=null)
				{
					//tempRating.mapMovieIdRating.put(movieId, rating);
				}
				else // MovieId is not present
				{	
					tempRating.mapMovieIdRating.put(movieId, rating);
				}
			}
		}
		
		br.close();
	}
	
	// Fill Test Data from the Test Set
	public static void FillTestData(ArrayList<TestData> testData, String path_testSet) throws NumberFormatException, IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(path_testSet));
		String line ="";
		while((line = br.readLine())!= null)
		{
			String[] stringParts = line.split(java.util.regex.Pattern.quote(","));
			String movieId = stringParts[0].trim();
			String userId = stringParts[1].trim();
			// stringParts[0] - movieId
			// stringParts[1] - userId 
			// stringParts[2] - Rating
			double rating = Double.parseDouble(stringParts[2].trim());
			
			// Fill the ArrayList
			TestData tempTestData = new TestData();
			tempTestData.setMovieId(movieId);
			tempTestData.setRating(rating);
			tempTestData.setUserId(userId);
			testData.add(tempTestData);
		}
		br.close();
		
		Collections.sort(testData, new Comparator<TestData>() {
	        public int compare(TestData o1, TestData o2) {
	            return o2.getUserId().compareTo(o1.getUserId());
	        }
	    });
	}
}
