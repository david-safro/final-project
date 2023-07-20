import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int inputSize = 784; // 28x28 images flattened
        int hiddenSize = 128;
        int outputSize = 10; // 10 classes (0 to 9)

        // Load MNIST data from CSV files
        List<double[]> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader("data/mnist_train.csv"));
            String line;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] input = new double[inputSize];

                for (int i = 0; i < inputSize; i++) {
                    input[i] = Double.parseDouble(values[i + 1]) / 255.0; // Normalize pixel values to [0, 1]
                }

                inputs.add(input);
                labels.add(Integer.parseInt(values[0]));
            }

            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize);
//e
        int epochs = 50;
        double learningRate = 0.1;

        neuralNetwork.train(inputs, labels, epochs, learningRate);

        // Test the trained model
        int correctPredictions = 0;
        int totalTestSamples = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader("data/mnist_test.csv"));
            String line;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] input = new double[inputSize];

                for (int i = 0; i < inputSize; i++) {
                    input[i] = Double.parseDouble(values[i + 1]) / 255.0; // Normalize pixel values to [0, 1]
                }

                int label = Integer.parseInt(values[0]);
                int predictedLabel = neuralNetwork.predict(input);

                if (predictedLabel == label) {
                    correctPredictions++;
                }

                totalTestSamples++;
            }

            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double accuracy = (double) correctPredictions / totalTestSamples * 100.0;
        System.out.println("Test Accuracy: " + accuracy + "%");
    }
}
