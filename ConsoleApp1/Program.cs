using System;
using System.Xml.Linq;

double[][] data = {
    new double[] { 0, 0, 0 },
    new double[] { 0, 0, 1 },
    new double[] { 0, 1, 0 },
    new double[] { 0, 1, 1 },
    new double[] { 1, 0, 0 },
    new double[] { 1, 0, 1 },
    new double[] { 1, 1, 0 },
    new double[] { 1, 1, 1 },
};
double[][] resData = {
    new double[] { 0 },
    new double[] { 1 },
    new double[] { 1 },
    new double[] { 0 },
    new double[] { 1 },
    new double[] { 0 },
    new double[] { 0 },
    new double[] { 1 },
};

Neurons n= new Neurons(3,4,1);
n.train(data, resData);
double[] ans= new double[1];
for(int i=0;i<8;i++)
{
    n.test(data[i],ans);
        Console.WriteLine("Ожидаем:{0} получено: {1}", resData[i][0], ans[0]);
}
