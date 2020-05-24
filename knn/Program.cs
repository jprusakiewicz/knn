using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace knn
{
  class KNNProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("Begin k-NN classification demo ");
      double[][] trainData = LoadData();
      int numFeatures = 4;
      int numClasses = 3;
      double[] unknown = new double[] { 7.2, 3.0, 5.8, 1.6};
      Console.WriteLine("Predictor values: 5.25 1.75 ");
      int k = 1;
      Console.WriteLine("With k = 1");
      int predicted = Classify(unknown, trainData,
        numClasses, k);
      Console.WriteLine("Predicted class = " + predicted);
      k = 4;
      Console.WriteLine("With k = 4");
      predicted = Classify(unknown, trainData,
        numClasses, k);
      Console.WriteLine("Predicted class = " + predicted);
      Console.WriteLine("End k-NN demo ");
      Console.ReadLine();
    }

    static int Classify(double[] unknown,
      double[][] trainData, int numClasses, int k)
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
        int c = (int) trainData[info[i].idx][4]; //to zmienilem
        string dist = info[i].dist.ToString("F3");
        Console.WriteLine("( " + trainData[info[i].idx][0] +
                          "," + trainData[info[i].idx][1] + " )  :  " +
                          dist + "        " + c);
      }
      int result = Vote(info, trainData, numClasses, k);
      return result;
    } // Classify

    static int Vote(IndexAndDistance[] info,
      double[][] trainData, int numClasses, int k) {
      int[] votes = new int[numClasses];  // One cell per class
      for (int i = 0; i < k; ++i) {       // Just first k
        int idx = info[i].idx;            // Which train item
        int c = (int)trainData[idx][4];   // Class in last cell //to zmienilem
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
      //sum = data.Select((x, i) => (x - unknown[i]) * (x - unknown[i])).Sum(); //nie dziala
      //sum = Math.Sqrt(data.Zip(unknown, (a, b) => (a - b)*(a - b)).Sum()); //stackoverflow.com/questions/8914669
      for (int i = 0; i < unknown.Length; ++i)
        sum += (unknown[i] - data[i]) * (unknown[i] - data[i]);
      return Math.Sqrt(sum);
    }
    static double[][] LoadData() {
      double[][] data = new Double[150][];
      using(var reader = new StreamReader(@"C:\Users\jakub.prusakiewicz\RiderProjects\knn\knn\iris.csv"))
      {
        
        var firstLine = reader.ReadLine();
        int columnsCount = 0;
        if (firstLine != null)
        {
          columnsCount = firstLine.Split(new[] {','}, StringSplitOptions.RemoveEmptyEntries).Length;
        }
        Console.WriteLine("Loading data: " + firstLine);
        Console.WriteLine("number of Columns: " + columnsCount);
        Console.WriteLine("---");
        int i = 0;
        while (!reader.EndOfStream)
        {
          var line = reader.ReadLine();
          Console.WriteLine(line);
          var values = line.Split(',');
          double class_num = -1;
          if (values[4] == "setosa")
            class_num = 0; //todo change for 0
          else if (values[4] == "versicolor")
            class_num = 1;
          else if (values[4] == "virginica")
            class_num = 2;
          else
            Console.WriteLine("CIPARAKIETA");

          data[i] = new Double[] { double.Parse(values[0], System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.NumberFormatInfo.InvariantInfo),
            double.Parse(values[1],System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.NumberFormatInfo.InvariantInfo),
            double.Parse(values[2], System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.NumberFormatInfo.InvariantInfo),
            double.Parse(values[3], System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.NumberFormatInfo.InvariantInfo), 
            class_num};
          i++;
        }

      }
      return data;
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