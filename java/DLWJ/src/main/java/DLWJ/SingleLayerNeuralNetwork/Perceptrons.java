// Perceptrons.java
package DLWJ.SingleLayerNeuralNetworks;

import java.util.Random;

import DLWJ.util.GaussianDistribution;
import static DLWJ.util.ActivationFunction.step;

public class Perceptrons {
  public int nIn;
  public double[] w;

  public Perceptrons(int nIn) {
    this.nIn = nIn;
    w = new double[nIn];
  }

  
  public int train(double[] x, int y, double learningRate) {
    int classified = 0;
    double c = 0;

    // Check if data is correctly classified
    for (int i = 0; i < nIn; i++) {
      c += w[i] * x[i] * y;
    }

    // Apply steepest descent method if wrongly classified
    if (c > 0) {
      classified = 1;
    } else {
      for (int i = 0; i < nIn; i++) {
        w[i] += learningRate * x[i] * y;
      }
    }
    return classified;
  }


  public int predict(double[] x) {
    double preActivation = 0.;

    for (int i = 0; i < nIn; i++) {
      preActivation += w[i] * x[i];
    }
    return step(preActivation);
  }


  public static void main(String[] args) {
    final int nTrain = 1000;
    final int nTest = 200;
    final int nIn = 2;
    double[][] xTrain = new double[nTrain][nIn];
    int[] yTrain = new int[nTrain];
    double[][] xTest = new double[nTest][nIn];
    int[] yTest = new int[nTest];
    int[] preds = new int[nTest];
    final int epochs = 2000;
    final double learningRate = 1.;

    // Create train and test data
    // Class 1: x1 ~ N(-2, 1), y1 ~ N(2, 1)
    // Class 2: x2 ~ N(2, 1),  y2 ~ N(-2, 1)
    final Random rng = new Random(1103);
    GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
    GaussianDistribution g2 = new GaussianDistribution(2.0,  1.0, rng);

    // Class 1
    for (int i = 0; i < nTrain/2 - 1; i++) {
      xTrain[i][0] = g1.random();
      xTrain[i][1] = g2.random();
      yTrain[i] = 1;
    }
    for (int i = 0; i < nTest/2 - 1; i++) {
      xTest[i][0] = g1.random();
      xTest[i][1] = g2.random();
      yTest[i] = 1;
    }

    // Class 2
    for (int i = nTrain / 2; i < nTrain; i++) {
      xTrain[i][0] = g2.random();
      xTrain[i][1] = g1.random();
      yTrain[i] = -1;
    }
    for (int i = nTest / 2; i < nTest; i++) {
      xTest[i][0] = g2.random();
      xTest[i][1] = g1.random();
      yTest[i] = -1;
    }

    // Build SingleLayerNN model
    int epoch = 0;
    Perceptrons classifier = new Perceptrons(nIn);

    // train
    while (epoch <= epochs) {
      int classified_ = 0;
      for (int i = 0; i < nTrain; i++) {
        classified_ += classifier.train(xTrain[i], yTrain[i], learningRate);
      }
      if (classified_ == nTrain) break; // all data correctly classified
      epoch++;
    }

    // test
    for (int i = 0; i < nTest; i++) {
      preds[i] = classifier.predict(xTest[i]);
    }

    // Evaluate
    int[][] confusionMatrix = new int[2][2];
    double accuracy = 0.;
    double precision = 0.;
    double recall = 0.;
    for (int i = 0; i < nTest; i++) {
      if (preds[i] > 0) {
        if (yTest[i] > 0) {
          accuracy += 1;
          precision += 1;
          recall += 1;
          confusionMatrix[0][0] += 1;
        } else {
          confusionMatrix[1][0] += 1;
        }
      } else {
        if (yTest[i] > 0) {
          confusionMatrix[0][1] += 1;
        } else {
          accuracy += 1;
          confusionMatrix[1][1] += 1;
        }
      }
    }

    accuracy /= nTest;
    precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
    recall /= confusionMatrix[0][0] + confusionMatrix[0][1];
    System.out.println("----------------------------");
    System.out.println("Preceptrons Model Evaluation");
    System.out.println("----------------------------");
    System.out.printf("Accuracy:  %.2f\n", accuracy);
    System.out.printf("Precision: %.2f\n", precision);
    System.out.printf("Recall:    %.2f\n", recall);
  }
}



