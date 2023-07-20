import java.util.List;
import java.util.Random;
//e
public class NeuralNetwork {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double[][] inputHiddenWeights;
    private final double[][] hiddenOutputWeights;
    private final double[] hiddenBiases;
    private final double[] outputBiases;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        inputHiddenWeights = new double[inputSize][hiddenSize];
        hiddenOutputWeights = new double[hiddenSize][outputSize];
        hiddenBiases = new double[hiddenSize];
        outputBiases = new double[outputSize];

        initializeWeightsAndBiases();
    }

    private void initializeWeightsAndBiases() {
        Random random = new Random();

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                inputHiddenWeights[i][j] = random.nextGaussian();
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenOutputWeights[i][j] = random.nextGaussian();
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            hiddenBiases[i] = random.nextGaussian();
        }

        for (int i = 0; i < outputSize; i++) {
            outputBiases[i] = random.nextGaussian();
        }
    }

    private double[] softmax(double[] x) {
        double max = x[0];
        for (double value : x) {
            if (value > max) {
                max = value;
            }
        }

        double sumExp = 0;
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i] - max);
            sumExp += result[i];
        }

        for (int i = 0; i < x.length; i++) {
            result[i] /= sumExp;
        }

        return result;
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        }
        return result;
    }

    private double[] forward(double[] input) {
        double[] hiddenOutput = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                hiddenOutput[j] += input[i] * inputHiddenWeights[i][j];
            }
            hiddenOutput[j] += hiddenBiases[j];
        }
        hiddenOutput = sigmoid(hiddenOutput);

        double[] output = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < hiddenSize; i++) {
                output[j] += hiddenOutput[i] * hiddenOutputWeights[i][j];
            }
            output[j] += outputBiases[j];
        }
        output = softmax(output);

        return output;
    }

    public int predict(double[] input) {
        double[] output = forward(input);
        int predictedLabel = -1;
        double maxProbability = -1;

        for (int i = 0; i < output.length; i++) {
            if (output[i] > maxProbability) {
                maxProbability = output[i];
                predictedLabel = i;
            }
        }

        return predictedLabel;
    }

    // Training the neural network using gradient descent.
    public void train(List<double[]> inputs, List<Integer> labels, int epochs, double learningRate) {
        int dataSize = inputs.size();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            for (int i = 0; i < dataSize; i++) {
                double[] input = inputs.get(i);
                int label = labels.get(i);

                // Forward pass
                double[] hiddenOutput = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < inputSize; k++) {
                        hiddenOutput[j] += input[k] * inputHiddenWeights[k][j];
                    }
                    hiddenOutput[j] += hiddenBiases[j];
                }
                hiddenOutput = sigmoid(hiddenOutput);

                double[] output = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++) {
                        output[j] += hiddenOutput[k] * hiddenOutputWeights[k][j];
                    }
                    output[j] += outputBiases[j];
                }
                output = softmax(output);

                // Compute loss
                double[] target = new double[outputSize];
                target[label] = 1.0;

                double loss = 0.0;
                for (int j = 0; j < outputSize; j++) {
                    loss -= target[j] * Math.log(output[j]);
                }

                totalLoss += loss;

                // Backpropagation
                double[] outputError = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    outputError[j] = output[j] - target[j];
                }

                double[] hiddenError = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        hiddenError[j] += outputError[k] * hiddenOutputWeights[j][k];
                    }
                    hiddenError[j] *= hiddenOutput[j] * (1 - hiddenOutput[j]);
                }

                // Update weights and biases
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        hiddenOutputWeights[j][k] -= learningRate * outputError[k] * hiddenOutput[j];
                    }
                }

                for (int j = 0; j < inputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++) {
                        inputHiddenWeights[j][k] -= learningRate * hiddenError[k] * input[j];
                    }
                }

                for (int j = 0; j < outputSize; j++) {
                    outputBiases[j] -= learningRate * outputError[j];
                }

                for (int j = 0; j < hiddenSize; j++) {
                    hiddenBiases[j] -= learningRate * hiddenError[j];
                }
            }

            System.out.println("Epoch " + epoch + ", Loss: " + totalLoss);
        }
    }
}
