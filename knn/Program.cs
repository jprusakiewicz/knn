using System;
namespace knn
{
  class KNNProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("Begin k-NN classification demo ");
      double[][] trainData = LoadData();
      int numFeatures = 2;
      int numClasses = 3;
      double[] unknown = new double[] { 5.25, 1.75 };
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
        int c = (int) trainData[info[i].idx][2];
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
        int c = (int)trainData[idx][2];   // Class in last cell
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
      double[][] data = new double[33][];
      data[0] = new double[] { 2.0, 4.0, 0 };
      data[1] = new double[] { 2.0, 1.0, 2 };
      data[2] = new double[] { 3.0, 1.0, 2 };
      data[3] = new double[] { 1.0, 4.0, 2 };
      data[4] = new double[] { 8.0, 3.0, 1 };
      data[5] = new double[] { 2.0, 2.0, 2 };
      data[6] = new double[] { 3.0, 2.0, 2 };
      data[7] = new double[] { 3.0, 3.0, 0 };
      data[8] = new double[] { 4.0, 3.0, 0 };
      data[9] = new double[] { 5.0, 3.0, 0 };
      data[10] = new double[] { 2.0, 6.0, 0 };
      data[11] = new double[] { 2.0, 5.0, 0 };
      data[12] = new double[] { 3.0, 4.0, 1 };
      data[13] = new double[] { 6.0, 4.0, 0 };
      data[14] = new double[] { 6.0, 5.0, 0 };
      data[15] = new double[] { 6.0, 6.0, 0 };
      data[16] = new double[] { 3.0, 7.0, 0 };
      data[17] = new double[] { 4.0, 7.0, 0 };
      data[18] = new double[] { 5.0, 7.0, 0 };
      data[19] = new double[] { 3.0, 6.0, 1 };
      data[20] = new double[] { 3.0, 5.0, 1 };
      data[21] = new double[] { 4.0, 4.0, 1 };
      data[22] = new double[] { 4.0, 5.0, 1 };
      data[23] = new double[] { 4.0, 6.0, 1 };
      data[24] = new double[] { 5.0, 4.0, 1 };
      data[25] = new double[] { 5.0, 5.0, 1 };
      data[26] = new double[] { 5.0, 6.0, 1};
      data[27] = new double[] { 6.0, 1.0, 1 };
      data[28] = new double[] { 7.0, 1.0, 1 };
      data[29] = new double[] { 7.0, 2.0, 1 };
      data[30] = new double[] { 8.0, 1.0, 1 };
      data[31] = new double[] {8.0, 2.0, 1};
      data[32] = new double[] { 4.0, 2.0, 2 };
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