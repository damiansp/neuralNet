package DLWJ.DeepNeuralNetworks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static DLWJ.util.ActivationFunction.sigmoid;
import static DLWJ.util.RandomGenerator.*;


public class RestrictedBoltzmannMachines {
  public int nVisible;
  public int nHidden;
  public double[][] W;
  public double[] bVisible;
  public double[] bHidden;
  public Random rng;

  public RestrictedBoltzmannMachines(
      int nVisible, int nHidden, double[][] W, double bVisible, double bHidden,
      Random rng) {
    if (rng == null) {
      rng = new Random(1234); // seed
    }
  }
}
