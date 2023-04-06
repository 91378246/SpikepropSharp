namespace SpikepropSharp;

public class ConfusionMatrix
{
    /// <summary>
    /// The amount of correct positive predictions
    /// </summary>
    public double TruePositives { get; set; }
    /// <summary>
    /// The amount of correct negative predictions
    /// </summary>
    public double TrueNegatives { get; set; }
    /// <summary>
    /// The amount of false positive predictions
    /// </summary>
    public double FalsePositives { get; set; }
    /// <summary>
    /// The amount of false negative predictions
    /// </summary>
    public double FalseNegatives { get; set; }
    /// <summary>
    /// The total amount of positive labels
    /// </summary>
    public double TotalPositives => TruePositives + FalseNegatives;
    /// <summary>
    /// The total amount of negative labels
    /// </summary>
    public double TotalNegatives => TrueNegatives + FalsePositives;
    /// <summary>
    /// How likely the model is to detect a true positive
    /// Also called recall
    /// </summary>
    public double Sensitivity => (TruePositives + FalseNegatives) != 0 ? TruePositives / (TruePositives + FalseNegatives) : -1;
    /// <summary>
    /// How likely the model is to detect a true negative
    /// </summary>
    public double Specificity => (FalsePositives + TrueNegatives) != 0 ? TrueNegatives / (FalsePositives + TrueNegatives) : -1;
    /// <summary>
    /// How trustfully the positive predictions of the model are
    /// </summary>
    public double Precision => (TruePositives + FalsePositives) != 0 ? TruePositives / (TruePositives + FalsePositives) : -1;
    /// <summary>
    /// The likelihood that a negative predictions is an actual true negative
    /// </summary>
    public double NegativePredictiveValue => (TrueNegatives + FalseNegatives) != 0 ? TrueNegatives / (TrueNegatives + FalseNegatives) : -1;
    /// <summary>
    /// The probability of a false positive
    /// </summary>
    public double FalsePositiveRate => (FalsePositives + TrueNegatives) != 0 ? FalsePositives / (FalsePositives + TrueNegatives) : -1;
    /// <summary>
    /// The probability of a false negative
    /// </summary>
    public double FalseNegativeRate => (FalseNegatives + TruePositives) != 0 ? FalseNegatives / (FalseNegatives + TruePositives) : -1;
    /// <summary>
    /// The probability of making type 1 errors (incorrect rejections of the null hypothesis)
    /// </summary>
    public double FalseDiscoveryRate => (FalsePositives + TruePositives) != 0 ? FalsePositives / (FalsePositives + TruePositives) : -1;
    /// <summary>
    /// The probability to predict correctly.
    /// Use this metric over the F1Score, if the true predictions are more important
    /// </summary>
    public double Accuracy => (TotalPositives + TotalNegatives) != 0 ? (TruePositives + TrueNegatives) / (TotalPositives + TotalNegatives) : -1;
    /// <summary>
    /// The harmonic mean of precision and sensitivity.
    /// Use this metric over the accuracy, if the false predictions are more important
    /// </summary>
    public double F1Score => (TruePositives + FalsePositives + FalseNegatives) != 0 ? 2 * TruePositives / (2 * TruePositives + FalsePositives + FalseNegatives) : -1;
    /// <summary>
    /// Represents the relationship between the predictions and the labels.
    /// 0 = no relationship, +/-1 = perfect positive/negative relation ship (negative = the variables change in the opposite direction)
    /// </summary>
    public double MatthewsCorrelationCoefficient
    {
        get
        {
            double sqrt = Math.Sqrt((TruePositives + FalsePositives) * (TruePositives + FalseNegatives) * (TrueNegatives + FalsePositives) * (TrueNegatives + FalseNegatives));
            return sqrt != 0 ? (TruePositives * TrueNegatives - FalsePositives * FalseNegatives) / sqrt : -1;
        }
    }

    public ConfusionMatrix()
    {

    }

    public ConfusionMatrix(int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
    {
        TruePositives = truePositives;
        TrueNegatives = trueNegatives;
        FalsePositives = falsePositives;
        FalseNegatives = falseNegatives;
    }

    public void MergeWith(ConfusionMatrix confusionMatrix)
    {
        TruePositives += confusionMatrix.TruePositives;
        TrueNegatives += confusionMatrix.TrueNegatives;
        FalsePositives += confusionMatrix.FalsePositives;
        FalseNegatives += confusionMatrix.FalseNegatives;
    }

    public static ConfusionMatrix FromPrediction(bool[][] predictions, bool[][] labels)
    {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int t = 0; t < (predictions.Rank > 1 ? predictions.GetLength(1) : predictions[0].Length); t++)
        {
            for (int outNeuronI = 0; outNeuronI < predictions.GetLength(0); outNeuronI++)
            {
                // True prediction
                if (predictions[outNeuronI][t] == labels[outNeuronI][t])
                {
                    if (predictions[outNeuronI][t])
                    {
                        truePositives++;
                    }
                    else
                    {
                        trueNegatives++;
                    }
                }
                // False prediction
                else
                {
                    if (predictions[outNeuronI][t])
                    {
                        falsePositives++;
                    }
                    else
                    {
                        falseNegatives++;
                    }
                }
            }
        }

        return new ConfusionMatrix(truePositives, trueNegatives, falsePositives, falseNegatives);
    }

    public override string ToString()
    {
        return $"Pos:{TotalPositives}, Neg:{TotalNegatives}\n" +
            $"TP: {TruePositives}, FP: {FalsePositives}\n" +
            $"TN: {TrueNegatives}, FN: {FalseNegatives}\n" +
            $"Sensitivity: {Sensitivity}, Specificity: {Specificity}\n" +
            $"Accuracy: {Convert.ToInt32(Accuracy * 100)}%, F1Score: {F1Score}";
    }
}

