using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.IO;

class FourierTransform
{
    public static double Amplitude(Complex x)
    {
        return Math.Sqrt(Math.Pow(x.Real, 2) + Math.Pow(x.Imaginary, 2));
    }

    public static double Phase(Complex x)
    {
        double r = x.Real;
        double im = x.Imaginary;
        if (Math.Abs(r) < 1e-7) r = 0.0;
        if (Math.Abs(im) < 1e-7) im = 0.0;

        if (Math.Abs(r) == 0) return Math.Sign(im) * Math.PI / 2;
        if (r > 0) return Math.Atan(im / r);
        if (r < 0 && im >= 0) return Math.PI + Math.Atan(im / r);
        if (r < 0 && im < 0) return Math.Atan(im / r) - Math.PI;
        return 0;
    }

    public static void PrintTable(List<Complex> z, List<Complex> newZ)
    {
        Console.WriteLine("{0,-5} | {1,-20} | {2,-20} | {3,-20} | {4,-30} | {5,-20}", "m", "Re(Z)", "Re(Z^)", "Im(Z^)", "|Z^|", "Phi");
        Console.WriteLine(new string('-', 120));
        for (int j = 0; j < z.Count; j++)
        {
            if (Amplitude(newZ[j]) > 1e-7 || Phase(newZ[j]) > 1e-7)
            {
                Console.WriteLine("{0,-5} | {1,-20} | {2,-20} | {3,-20} | {4,-30} | {5,-20}", j, z[j].Real, newZ[j].Real, newZ[j].Imaginary, Amplitude(newZ[j]), Phase(newZ[j]));
                Console.WriteLine(new string('-', 120));
            }
        }
    }

    public static void Filtration(ref List<Complex> z)
    {
        double maxAmplitude = 0.0;

        foreach (var el in z)
        {
            double amplitude = Amplitude(el);
            if (amplitude >= maxAmplitude && amplitude > 1e-7)
            {
                maxAmplitude = amplitude;
            }
        }

        for (int j = 0; j < z.Count; j++)
        {
            double amplitude = Amplitude(z[j]);
            if (amplitude < maxAmplitude - 1 && amplitude > 1e-7)
            {
                z[j] = Complex.Zero;
            }
        }
    }

    public static void VectorToFile(List<Complex> z, string filename)
    {
        using (StreamWriter file = new StreamWriter(filename))
        {
            foreach (var el in z)
            {
                file.WriteLine(el.Real);
            }
        }
    }
}

class DFT
{
    public static List<Complex> Forward(List<Complex> input)
    {
        int N = input.Count;
        List<Complex> output = new List<Complex>(new Complex[N]);
        for (int k = 0; k < N; ++k)
        {
            output[k] = Complex.Zero;
            for (int n = 0; n < N; ++n)
            {
                double angle = -2.0 * Math.PI * k * n / N;
                output[k] += input[n] * Complex.Exp(new Complex(0, angle));
            }
        }
        return output;
    }

    public static List<Complex> Inverse(List<Complex> input)
    {
        int N = input.Count;
        List<Complex> output = new List<Complex>(new Complex[N]);
        for (int k = 0; k < N; ++k)
        {
            output[k] = Complex.Zero;
            for (int n = 0; n < N; ++n)
            {
                double angle = 2.0 * Math.PI * k * n / N;
                output[k] += input[n] * Complex.Exp(new Complex(0, angle));
            }
            output[k] /= N;
        }
        return output;
    }
}

class FFT
{
    public static List<Complex> Forward(List<Complex> input)
    {
        int N = input.Count;
        if (N <= 1) return input;

        if ((N & (N - 1)) != 0)
        {
            throw new ArgumentException("Size of input must be a power of 2.");
        }

        List<Complex> even = new List<Complex>();
        List<Complex> odd = new List<Complex>();
        for (int i = 0; i < N; ++i)
        {
            if (i % 2 == 0) even.Add(input[i]);
            else odd.Add(input[i]);
        }

        List<Complex> evenFFT = Forward(even);
        List<Complex> oddFFT = Forward(odd);

        List<Complex> output = new List<Complex>(new Complex[N]);
        for (int k = 0; k < N / 2; ++k)
        {
            Complex t = Complex.Exp(new Complex(0, -2.0 * Math.PI * k / N)) * oddFFT[k];
            output[k] = evenFFT[k] + t;
            output[k + N / 2] = evenFFT[k] - t;
        }
        return output;
    }

    public static List<Complex> Inverse(List<Complex> input)
    {
        int N = input.Count;
        if (N <= 1) return input;

        List<Complex> conjugatedInput = new List<Complex>();
        foreach (var item in input)
        {
            conjugatedInput.Add(Complex.Conjugate(item));
        }

        List<Complex> output = Forward(conjugatedInput);
        for (int i = 0; i < N; ++i)
        {
            output[i] = Complex.Conjugate(output[i]) / N;
        }
        return output;
    }
}

class Program
{
    public static Func<double, double> GetSignalFunction(int N, double A, double B, double omega, double phi)
    {
        return (x) => A + B * Math.Cos(2 * Math.PI * omega * x / N + phi);
    }

    public static Func<double, double> GetNoisedSignalFunction(int N, double omega)
    {
        return (x) => Math.Cos(2 * Math.PI * x / N) + 0.01 * Math.Cos(2 * Math.PI * omega * x / N);
    }

    public static void Main()
    {
        int N = 512;
        double A = -7;
        double B = 0.58;
        double omega = 491;
        double phi = Math.PI / 6;

        List<Complex> z = new List<Complex>();
        List<Complex> zNoised = new List<Complex>();
        List<Complex> newZ = new List<Complex>();

        var signal = GetSignalFunction(N, A, B, omega, phi);
        var noisedSignal = GetNoisedSignalFunction(N, omega);

        for (int i = 0; i < N; i++)
        {
            double x = signal(i);
            double xNoised = noisedSignal(i);
            z.Add(new Complex(x, 0));
            zNoised.Add(new Complex(xNoised, 0));
        }

        // DFT
        var start = DateTime.Now;
        for (int i = 0; i < 100; i++)
        {
            newZ = DFT.Forward(z);
        }
        var end = DateTime.Now;
        var duration = (end - start).TotalMilliseconds / 100;
        FourierTransform.PrintTable(z, newZ);
        Console.WriteLine($"DFT time: {duration} ms");

        Console.WriteLine();

        // FFT
        start = DateTime.Now;
        for (int i = 0; i < 100; i++)
        {
            newZ = FFT.Forward(z);
        }
        end = DateTime.Now;
        duration = (end - start).TotalMilliseconds / 100;
        FourierTransform.PrintTable(z, newZ);
        Console.WriteLine($"FFT time: {duration} ms");

        // Saving noisy signal to file
        FourierTransform.VectorToFile(zNoised, "noisedZ.txt");
        newZ = DFT.Forward(zNoised);
        FourierTransform.PrintTable(zNoised, newZ);

        // Filtration and saving filtered signal
        Console.WriteLine();
        FourierTransform.Filtration(ref newZ);
        FourierTransform.PrintTable(zNoised, newZ);
        newZ = DFT.Inverse(newZ);
        FourierTransform.VectorToFile(newZ, "filteredZ.txt");
    }
}
