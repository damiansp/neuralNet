package DLWJ.SingleLayerNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

//import DLWJ.util.ActivationFunction.softmax;
import DLWJ.util.GaussianDistribution;

import static DLWJ.util.ActivationFunction.softmax;


public class LogisticRegression {
  public int nIn;
  public int nOut;
  public double[][] W;
  public double[] b;

  
  public LogisticRegression(int nIn, int nOut) {
    this.nIn = nIn;
    this.nOut = nOut;
    W = new double[nOut][nIn];
    b = new double[nOut];
  }


  public double[][] train(
      double[][] X, int[][] Y, int batchSize, double learningRate) {
    double[][] dW = new double[nOut][nIn];
    double[] db = new double[nOut];
    double[][] dY = new double[batchSize][nOut];

    // SGD
    // 1. Calculate gradients for W, b
    for (int n = 0; n < batchSize; n++) { // n: records
      double[] pred = output(X[n]);
      for (int j = 0; j < nOut; j++) { // j: classes
        dY[n][j] = pred[j] - Y[n][j];
        for (int i = 0; i < nIn; i++) { // i: features
          dW[j][i] += dY[n][j] * X[n][i];
        }
        db[j] += dY[n][j];
      }
    }

    // 2. Update params
    for (int j = 0; j < nOut; j++) {
      for (int i = 0; i < nIn; i++) {
        W[j][i] -= learningRate * dW[j][i] / batchSize;
      }
      b[j] -= learningRate * db[j] / batchSize;
    }
    return dY;
  }


  public double[] output(double[] x) {
    double[] preActivation = new double[nOut];

    for (int j = 0; j < nOut; j++) {
      for (int i = 0; i < nIn; i++) {
        preActivation[j] += W[j][i] * x[i];
      }
      preActivation[j] += b[j];
    }
    return softmax(preActivation, nOut);
  }


  public Integer[] predict(double[] x) {
    double[] yProb = output(x); // network prediction (as probability)
    Integer[] pred = new Integer[nOut]; // as binary class label
    int argmax = -1;
    double max = 0.;

    for (int i = 0; i < nOut; i++) {
      if (max < yProb[i]) {
        max = yProb[i];
        argmax = i;
      }
    }
    for (int i = 0; i < nOut; i++) {
      pred[i] = i == argmax ? 1 : 0;
    }
    return pred;
  }


  public static void main(String[] args) {
    final Random rand = new Random(1103);
    final int CLASSES = 3;
    final int N_TRAIN = 400 * CLASSES;
    final int N_TEST  =  60 * CLASSES;
    final int N_IN = 2;
    final int N_OUT = CLASSES;
    final int EPOCHS = 2000;
    final int BATCH = 64;
    final int N_BATCHES = N_TRAIN / BATCH;
    double learningRate = 0.2;
    double[][] xTrain = new double[N_TEST][N_IN];
    int[][] yTrain = new int[N_TRAIN][N_OUT];
    double[][] xTest = new double[N_TEST][N_IN];
    Integer[][] yTest = new Integer[N_TEST][N_OUT];
    Integer[][] preds = new Integer[N_TEST][N_OUT];
    double [][][] xTrainBatches = new double[N_BATCHES][BATCH][N_IN];
    int[][][] yTrainBatches = new int[N_BATCHES][BATCH][N_OUT];
    List<Integer> batchIndex = new ArrayList<>();

    for (int i = 0; i < N_TRAIN; i++) { batchIndex.add(i); }
    Collections.shuffle(batchIndex, rand);

    // Data
    // Class x        y
    // 1     N(-2, 1) N( 2, 1)
    // 2     N( 2, 1) N(-2, 1)
    // 3     N( 0, 1) N( 0, 1)
    GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rand);
    GaussianDistribution g2 = new GaussianDistribution( 2.0, 1.0, rand);
    GaussianDistribution g3 = new GaussianDistribution( 0.0, 1.0, rand);

    // Class 1
    for (int i = 0; i < N_TRAIN/CLASSES - 1; i++) {
      xTrain[i][0] = g1.random();
      xTrain[i][1] = g2.random();
      yTrain[i] = new int[]{1, 0, 0};
    }
    for (int i = 0; i < N_TEST/CLASSES - 1; i++) {
      xTest[i][0] = g1.random();
      xTest[i][1] = g2.random();
      yTest[i] = new Integer[]{1, 0, 0};
    }

    // Class 2
    for (int i = N_TRAIN/CLASSES - 1; i < 2*N_TRAIN/CLASSES - 1; i++) {
      xTrain[i][0] = g2.random();
      xTrain[i][1] = g1.random();
      yTrain[i] = new int[]{0, 1, 0};
    }
    for (int i = N_TEST/CLASSES - 1; i < 2*N_TEST/CLASSES - 1; i++) {
      xTest[i][0] = g2.random();
      xTest[i][1] = g1.random();
      yTest[i] = new Integer[]{0, 1, 0};
    }

    // Class 3
    for (int i = 2*N_TRAIN/CLASSES - 1; i < N_TRAIN; i++) {
      xTrain[i][0] = g3.random();
      xTrain[i][1] = g3.random();
      yTrain[i] = new int[]{0, 0, 1};
    }
    for (int i = 2*N_TEST/CLASSES - 1; i < N_TEST; i++) {
      xTest[i][0] = g1.random();
      xTest[i][1] = g2.random();
      yTest[i] = new Integer[]{1, 0, 0};
    }

    // Create mini-batches with training data
    for (int i = 0; i < N_BATCHES; i++) {
      for (int j = 0; j < BATCH; j++) {
        xTrainBatches[i][j] = xTrain[batchIndex.get(i * BATCH + j)];
        yTrainBatches[i][j] = yTrain[batchIndex.get(i * BATCH + j)];
      }
    }

    // Build model
    LogisticRegression classifier = new LogisticRegression(N_IN, N_OUT);

    // Train
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
      for (int batch = 0; batch < N_BATCHES; batch++) {
        classifier.train(
          xTrainBatches[batch], yTrainBatches[batch], BATCH, learningRate);
      }
      learningRate *= 0.99;
    }

    // Test
    for (int i = 0; i < N_TEST; i++) {
      preds[i] = classifier.predict(xTest[i]);
    }

    // Evaluate model
    int[][] confusionMatrix = new int[CLASSES][CLASSES];
    double accuracy = 0.;
    double[] precision = new double[CLASSES];
    double[] recall = new double[CLASSES];

    for (int i = 0; i < N_TEST; i++) {
      int pred = Arrays.asList(preds[i]).indexOf(1);
      int actual = Arrays.asList(yTest[i]).indexOf(1);

      confusionMatrix[actual][pred] += 1;
    }

    for (int i = 0; i < CLASSES; i++) {
      double col = 0.;
      double row = 0.;

      for (int j = 0; j < CLASSES; j++) {
        if (i == j) {
          accuracy += confusionMatrix[i][j];
          precision[i] += confusionMatrix[j][i];
          recall[i] += confusionMatrix[i][j];
        }
        col += confusionMatrix[j][i];
        row += confusionMatrix[i][j];
      }
      precision[i] /= col;
      recall[i] /= row;
    }
    accuracy /= N_TEST;

    System.out.println("------------------------------------");
    System.out.println("Logistic Regression Model Evaluation");
    System.out.println("------------------------------------");
    System.out.printf("Accuracy: %.2f%%\n", 100 * accuracy);
    System.out.println("Precision:");
    for (int i = 0; i < CLASSES; i++) {
      System.out.printf(" class %d: %.2f%%\n", i + 1, 100 * precision[i]);
    }
    System.out.println("Recall:");
    for (int i = 0; i < CLASSES; i++) {
      System.out.printf(" class %d: %.2f%%\n", i + 1, 100 * recall[i]);
    }
  }
}
