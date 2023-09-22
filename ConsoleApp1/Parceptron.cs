using System;
using System.ComponentModel;

public class Neurons
{
    private int numNeuronLin;
    private int numNeuronL1;
    private int numNeuronLout;
    private double[,] w1;
    private double[] b1;
    private double[] delta1;
    private double[,] w2;
    private double[] b2;
    private double[] delta2;
    private double Error = 0;
    private double E = 0.1;
    private double[] innerLayer;
    private double[] res ;//выход сети

    public Neurons(int numInput, int numHidden, int numOutput)
    {
        this.numNeuronLin = numInput;
        this.numNeuronL1 = numHidden;
        this.numNeuronLout = numOutput;

        w1 = new double[numInput, numHidden];
        b1 = new double[numHidden];
        delta1 = new double[numHidden];
        w2 = new double[numHidden, numOutput];
        b2 = new double[numOutput];
        delta2 = new double[numOutput];
        innerLayer = new double[numHidden];
        res= new double[numOutput];
        Init();
    }
    private void Init()
    {
        Random rnd = new Random();
        for (int i = 0; i< numNeuronLin; i++)
            for (int j = 0; j< numNeuronL1; j++)
            {
                w1[i, j] = rnd.NextDouble();
                b1[j]= rnd.NextDouble();
            }
        for (int i = 0; i < numNeuronL1; i++)
            for (int j = 0; j < numNeuronLout; j++)
            {
                w2[i, j] = rnd.NextDouble();
                b2[j] = rnd.NextDouble();
            }

    }
    public void train(double[][]data, double[][] resData)
    {
        bool flag = false;
        int iter = 0;
        do
        {
            flag = false;
            for (int i=0;i<data.GetLength(0);i++)
            {
                Error = 0;
                test(data[i], res );
                for (int j = 0; j < numNeuronLout; j++)
                    Error+= (resData[i][j]  - res[j]) * (resData[i][j] - res[j]);
                if (Error>0.0001)
                {
                    flag = true;
                    backpropagation(resData, i);//ОРО
                    adjustingWights(i, data);//корректируем веса
                }
                iter++;
            }
            if (iter > 200000) break;
        } while (flag);
        Console.WriteLine("Количество итераций:{0} ", iter);
    }
    public void test(double[] input,  double[] output)//здесь прямое распространение
    {
        for (int j = 0; j < numNeuronL1; j++)
        {
            innerLayer[j] = 0;
            for (int i = 0; i < numNeuronLin; i++)
                innerLayer[j] += input[i] * w1[i, j];
            innerLayer[j] -= b1[j];
            innerLayer[j] = Sigmoid(innerLayer[j]);//активация 
        }

        for (int j = 0; j < numNeuronLout; j++)
        {
            output[j] = 0;
            for (int i = 0; i < numNeuronL1; i++)
                output[j] += innerLayer[i] * w2[i, j];
            output[j] -= b2[j];
            output[j] = Sigmoid(output[j]);//активация 
        }


    }
    private void backpropagation(double[][] realRes, int k)
    {
        for (int j = 0; j < numNeuronLout; j++)
           delta2[j] = -7*res[j] * (1 - res[j]) * (realRes[k][j] - res[j]);

        double sum = 0;
        for (int j = 0; j < numNeuronL1; j++)
        {
            sum = 0;
            for (int i = 0; i < numNeuronLout; i++)
                sum += delta2[i] * w2[j, i];    
            delta1[j] = 7 * innerLayer[j] * (1 - innerLayer[j]) * sum;
        }
            
    }
    private void adjustingWights(int k, double[][] data)
    {
        for (int i = 0; i < numNeuronL1; i++)
            for (int j = 0; j < numNeuronLout; j++)
            {
                w2[i, j] -= E * delta2[j] * innerLayer[i];
                b2[j] += E * delta2[j];
            }
        for (int i = 0; i < numNeuronLin; i++)
            for (int j = 0; j < numNeuronL1; j++)
            {
                w1[i, j]-= E * delta1[j] * data[k][i];
                b1[j]+= E * delta1[j];
            }
    }
    
    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-7*x));
    }

}