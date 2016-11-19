// Francisco Samuel Rios
// November 13, 2016
// CAP 4630 - Program 3
// Weka Perceptron Classifier
// This program implements the perceptron learning algorithm as it is 
// detailed within our lecture.

import weka.classifiers.Classifier;
import java.text.DecimalFormat;
import weka.core.*;

public class Perceptron implements weka.classifiers.Classifier
{
	String fileName;
	int epoch = 0; 
	int bias = 0;
	int weightUpdates = 0;
	double learningRate = 0.0;
	double[] weights;
		
	// Constructor for the Perceptron class
	// Pre-conditions: An array of strings containing the filename to inspect, the desired number of epochs,
	// 				   and the desired learning rate
	// Post-conditions: The constructor creates a new perceptron object
	public Perceptron(String[] options)
	{
		// Print the header for the output
		System.out.println("\nUniversity of Central Florida ");
		System.out.println("CAP4630 Artificial Intelligence - Fall 2016");
		System.out.println("Perceptron Classifier by Francisco Samuel Rios \n");
		
		// Parse and assign the passed in options
		this.bias             = 1;
		this.fileName 		  = options[0];
		this.epoch    		  = Integer.parseInt(options[1]);
		this.learningRate     = Double.parseDouble(options[2]);
	}
	
	// Method that builds the classifier for the perceptron
	// Pre-Conditions:  Takes in a collection of instance objects called data
	// Post-conditions: N/A
	public void buildClassifier(Instances data) throws Exception
	{
		// Gets the number of instances in the data set
        int numInst = data.numInstances();
		if(numInst == 0)
		{
			// Ends if there are no instances
			return;
		}
		
		// Gets and stores the first piece of data in the collection of instances
		Instance firstInst = data.firstInstance();
		
		// Gets and stores the number of attributes in that first instance
		int numAttributes = firstInst.numAttributes();
		
		// Creates an array of doubles to store the weights for each attribute.
		weights = new double[numAttributes];
		
		// Initializes the weights of all of the attributes in the first instance to 0.0
		for(int i = 0; i < firstInst.numAttributes(); i++)
		{
			firstInst.attribute(i).setWeight(0.0);
		}
		
		// The beginning of the iteration loop
		for(int i = 0; i < this.epoch; i++)
		{
			// Numbers the epoch (iteration)
			System.out.print("Epoch " + i + ": ");
		
			// Begins analyzing each piece of data in the current instance
			for(int j = 0; j < numInst; j++)
			{
				// Get the instance at location j
				Instance inst = data.instance(j);
				
				// Get the number of attributes for that instance
				int x = inst.numAttributes();
				
				// Create a new array to hold the values of the attributes
				double[] attributes = new double[x-1];
				
				// Assigns the value at position k of the current instance, and stores it in the attributes array at position k 
				for(int k = 0; k < x-1; k++)
				{
					attributes[k] = inst.value(k);
				}
				
				// This summation will reset upon each loop of the 'j' for loop
				double attributeSum = 0.0;
				
				// Sums the values of all the attribute values times their weights
				for(int l = 0; l < x-1; l++)
				{
					attributeSum += inst.attribute(l).weight()*attributes[l];
				}
				
				// Calculates the bias weighting
				double biasWeight = inst.attribute(x-1).weight() * this.bias;
				
				// Adds it to the sum of the attributes
				double total = biasWeight + attributeSum;
				int biasTotal;
				
				// Determines a value for biasTotal depending on the sum of the biasWeight and sum of attributes
				if(total < 0)
				{
					// If the total with the biasWeight is less than 0, the biasTotal should be -1
					biasTotal = -1;
				}
				else
				{
					// Otherwise it should be +1
					biasTotal = 1;
				}
				
				// Calculates the expected total
				int expectedTotal;
				double tempValue = inst.value(inst.attribute(x-1));
				
				// If the the value retrieved above is equal to +1, then our expected total is -1
				if(tempValue == 1.0)
				{
					expectedTotal = -1;
				}
				// If the value retreieved is equal to 0, then our expected total is 1
				else if(tempValue == 0.0)
				{
					expectedTotal = 1;
				}
				// And if the value retrieved is equal to 1, then our expected total is 0
				else
				{
					expectedTotal = 0;
				}
				
				// Compares the actual total and the expected total and determines the class of the piece of data
				if(biasTotal != expectedTotal)
				{
					// If they're not equal, print a 0
					System.out.print("0");
					
					// Increase the number of times the weight has been updated by 1
					this.weightUpdates++;
					
					// Case where biasTotal is positive and expectedTotal is negative
					if(biasTotal == 1 && expectedTotal == -1)
					{
						// Go through each attribute and iteratively set the weight according to the weights at m, 
						// the attribute at m, and the learning rate 
						for(int m = 0; m < x-1; m++)
						{
							data.attribute(m).setWeight((data.attribute(m).weight()) - 2*this.learningRate*attributes[m]);
						}
						// Adjust the weight of the last attribute after iterating through all of the other weights.
						data.attribute(x-1).setWeight((data.attribute(x-1).weight()) - 2*this.learningRate*1.0);
					}
					// Case where biasTotal is negative and expectedTotal is positive
					else if(biasTotal == -1 && expectedTotal == 1)
					{
						// Go through each attribute and iteratively set the weight according to the weights at m, 
						// the attribute at m, and the learning rate 
						for(int m = 0; m < x-1; m++)
						{
							data.attribute(m).setWeight((data.attribute(m).weight()) + 2*this.learningRate*attributes[m]);
						}
						// Adjust the weight of the last attribute after iterating through all of the other weights.
						data.attribute(x-1).setWeight((data.attribute(x-1).weight()) + 2*this.learningRate*1.0);
					}
				}
				// Otherwise, we can classify it, and we label it as a 1
				else
				{
					System.out.print("1");
				}
			}
			// End of the epoch, go to the next line
			System.out.print("\n");
		}
		
		// Store all of the weights in data back in the weights[] array
		for(int i = 0; i < numAttributes; i++)
		{
			this.weights[i] = data.attribute(i).weight();
		}
	}
	
	// Predicts the class memberships for the given instance.
	// Pre-Conditions:  Takes in an instance object called data
	// Post-Conditions: Returns an array of doubles
	public double[] distributionForInstance(Instance data)
	{
		// Creates the double array we will be returning
    	double[] distResult = new double[2];
		
		// Creates the array that holds all of the attributes
		int x = data.numAttributes();
		double[] attributes = new double[x-1];
		
		// Assigns a piece of data to each attribute
		for(int i = 0; i < x-1; i++)
		{
			// Assigns a data value to attributes
			attributes[i] = data.value(i);
		}
		
		double sum = 0.0;
		for(int j = 0; j < x-1; j++)
		{
			sum += data.attribute(j).weight() * (attributes[j]);
		}
 
		double biasWeight = data.attribute(x-1).weight()*this.bias;
		double result = sum + biasWeight;
		int total;
		
		// Calculates the total based on the sum of "sum" and the biasWeight
		if(result >= 0.0)
		{
			total = 1;
		}
		else
		{
			total = -1;
		}
		
		// Has two cases, and two seperate value arrangements depending on the value of total
		if(total == 1)
		{
			distResult[0] = 1.0;
			distResult[1] = 0.0;
		}
		else
		{
			distResult[0] = 0.0;
			distResult[1] = 1.0;
		}
		
		// Returns the array of doubles
		return distResult;
	}
	
	// Empty concrete definition of getCapabilities() as required by the implementation
	public Capabilities getCapabilities() { return null; }

	// Empty concrete definition of classifyInstance() as required by the implementation
	public double classifyInstance(Instance data) throws Exception { return 0.0; }

	// Prints out the final set of data with statistics about the classifier and the final weights
	// Pre-Conditions:  N/A
	// Post-Conditions: Returns an empty string after printing all required data
	public String toString()
	{
		// Compiles a string consisting of the values of the final weights
		String finalWeights = "";
		
		// Creates a new instance of Decimal Format to restrict the print-outs of the 
		// final weights as shown in simple-out.png
		DecimalFormat df = new DecimalFormat("#0.000");
		
		// Creates the string containing the final weights
		for(int i = 0; i < this.weights.length; i++)
		{
			finalWeights += (df.format(this.weights[i]) + "\n");
		}
	
		// Prints out the required data
		System.out.println("Source file: " + this.fileName);
		System.out.println("Training epochs: " + this.epoch);
		System.out.println("Learning rate: " + this.learningRate + "\n");
		
		System.out.println("Total # weight updates = " + this.weightUpdates);
		System.out.println("Final weights: \n" + finalWeights); 
    	return " ";
	}
}