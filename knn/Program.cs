using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

namespace knn
{
  class KNNProgram
  {
    private static string CSV_PATH = @"C:\Users\jakub.prusakiewicz\RiderProjects\knn\knn\iris.csv";
    private static int CSV_LENGTH = 150;
    private static bool IS_CLASS_NAME = true;
    
    //random generated data
    // private static string CSV_PATH = @"C:\Users\jakub.prusakiewicz\RiderProjects\knn\knn\MOCK_DATA_cztery.csv";
    // private static int CSV_LENGTH = 1000;
    // private static bool IS_CLASS_NAME = false;
    
    private static string FIRST_CLASS_NAME = "setosa";
    private static string SECOND_CLASS_NAME = "versicolor";
    private static string THIRD_CLASS_NAME = "virginica";
    static int k = 4;
    static int numFeatures = 4;
    static int numClasses = 4;
    
    static void Main(string[] args)
    {
      Console.WriteLine("Begin k-NN classification demo ");
      double[][] trainData = LoadData();
      double[][] validationData = SplitData(trainData, out trainData);

      int[] trueLabel = new int[validationData.Length-1];
      int[] predLabel = new int[validationData.Length-1];
      double[][] unknown = new double[validationData.Length-1][];
      for (int i = 0; i < validationData.Length-1; i++)
      {
        trueLabel[i] = (int) validationData[i][4];
        unknown[i] = validationData[i].Where((item, index) => index != validationData[i].Length-1).ToArray();
      }

      for (int i = 0; i < validationData.Length - 1; i++)
      {
        predLabel[i] = Classify(unknown[i], trainData,
          numClasses, k, numFeatures);
        Console.WriteLine("pred: "+ predLabel[i]+ " | true: " + trueLabel[i]);
      }
      var confusionMatrix = BuildConfusionMatrix(trueLabel, predLabel, numClasses);
      Console.WriteLine("Confusion matrix:");
      Console.WriteLine("` 0 1 2 3     Horizontally : predicted label     "+ "Vertically: true label");
      for (int i = 0; i < confusionMatrix.GetLength(0); i++)
      {
        Console.Write(i+"|");
        for (int l = 0; l < confusionMatrix.GetLength(0); l++) {
          Console.Write(confusionMatrix[i][l]+" ");
        }
        Console.WriteLine();
      }
    }
    private static int[][] BuildConfusionMatrix(int[] actual, int[] preds, int numClass)
    {
      int[][] matrix = new int[numClass][];
      for (int i = 0; i < numClass; i++)
      {
        matrix[i] = new int[numClass];
      }

      for (int i = 0; i < actual.Length; i++)
      {
        matrix[actual[i]][preds[i]] += 1;
      }

      return matrix;
    }

    static int Classify(double[] unknown,
      double[][] trainData, int numClasses, int k, int numFeatures)
    {
      int n = trainData.Length;
      IndexAndDistance[] info = new IndexAndDistance[n];
      for (int i = 0; i < n; ++i)
      {
        IndexAndDistance curr = new IndexAndDistance();
        double dist = Distance(unknown, trainData[i]);
        curr.idx = i;
        curr.dist = dist;
        info[i] = curr;
      }

      Array.Sort(info); // sort by distance
      Console.WriteLine("Nearest / Distance / Class");
      Console.WriteLine("==========================");
      for (int i = 0; i < k; ++i)
      {
        int c = (int) trainData[info[i].idx][numFeatures]; //to zmienilem
        string dist = info[i].dist.ToString("F3");
        Console.WriteLine("( " + trainData[info[i].idx][0] +
                          "," + trainData[info[i].idx][1] + " )  :  " +
                          dist + "        " + c);
      }
      int result = Vote(info, trainData, numClasses, k, numFeatures);
      return result;
    } // Classify

    static int Vote(IndexAndDistance[] info,
      double[][] trainData, int numClasses, int k, int numFeatures) {
      int[] votes = new int[numClasses];  // One cell per class
      for (int i = 0; i < k; ++i) {       // Just first k
        int idx = info[i].idx;            // Which train item
        int c = (int)trainData[idx][numFeatures];   // Class in last cell //to zmienilem
        ++votes[c];
      }
      int mostVotes = 0;
      int classWithMostVotes = 0;
      for (int j = 0; j < numClasses; ++j) {
        if (votes[j] > mostVotes) {
          mostVotes = votes[j];
          classWithMostVotes = j;
        }
      }
      return classWithMostVotes;
    }
    static double Distance(double[] unknown,
      double[] data) {
      double sum = 0.0;
      for (int i = 0; i < unknown.Length; ++i)
        sum += (unknown[i] - data[i]) * (unknown[i] - data[i]);
      return Math.Sqrt(sum);
    }
    static double[][] LoadData() {
      double[][] data = new Double[CSV_LENGTH][];
      using(var reader = new StreamReader(CSV_PATH))
      {
        var firstLine = reader.ReadLine();
        Debug.Assert(firstLine != null, nameof(firstLine) + " != null");
        int columnsCount = firstLine.Split(new[] {','}, StringSplitOptions.RemoveEmptyEntries).Length;
        
        Console.WriteLine("Loading data: " + firstLine);
        Console.WriteLine("number of Columns: " + columnsCount);
        Console.WriteLine("---");
        int i = 0;
        while (!reader.EndOfStream)
        {
          var line = reader.ReadLine();
          // Console.WriteLine(line); // uncomment to see training data
          if (line != null)
          {
            var values = line.Split(',');
            double classNum;
            if (IS_CLASS_NAME)
            {

              if (values[4] == FIRST_CLASS_NAME)
                classNum = 0;
              else if (values[4] == SECOND_CLASS_NAME)
                classNum = 1;
              else if (values[4] == THIRD_CLASS_NAME)
                classNum = 2;
              else
                throw new System.ArgumentException("unclear training data", "class_name");
              
              data[i] = new Double[] { 
                double.Parse(values[0], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[1],NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[2], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[3], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo), 
                classNum};
            }
            else
            {
              data[i] = new Double[] { 
                double.Parse(values[0], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[1],NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[2], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo),
                double.Parse(values[3], NumberStyles.AllowDecimalPoint, NumberFormatInfo.InvariantInfo), 
                double.Parse(values[4])};
            }
            
          }

          i++;
        }

      }
      // randomize data
      Random rnd=new Random();
      data = data.OrderBy(x => rnd.Next()).ToArray();    
      return data;
    }
    static public T[] Split<T>(T[] array, int index, out T[] first) {
      first = array.Skip(index).ToArray();
       return array.Take(index).ToArray();
    }

    static public T[] SplitData<T>(T[] array, out T[] first)
    {
      var len = array.Length / 10;
      double[][] validationData = new double[len][];
      return Split(array, array.Length / 10, out first);
    }
  } // Program class
  public class IndexAndDistance : IComparable<IndexAndDistance>
  {
    public int idx;  // Index of a training item
    public double dist;  // To unknown
    // Need to sort these to find k closest
    public int CompareTo(IndexAndDistance other)
    {
      if (this.dist < other.dist) return -1;
      else if (this.dist > other.dist) return +1;
      else return 0;
    }
  }
  
} // ns