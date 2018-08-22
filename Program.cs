using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Sentimentos
{
    class Program
    {
        static void Main(string[] args)
        {
            const string dataPath = @"./dataset/data.txt";
            const string testDataPath = @"./dataset/test.txt";

            string[] classNames = { "Positive", "Negative" };

            var classification = new Classification();
            var model = classification.Train(dataPath);

            var result = classification.Test(testDataPath, model);
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {result.Accuracy}%");
            Console.WriteLine($"Auc: {result.Auc}%");
            Console.WriteLine($"F1Score: {result.F1Score}%");
            Console.WriteLine();

            var test = new List<string>
            {
                "Uowww ... I love this place.",
                "It's not good.",
                "It did not taste good and the texture was a bit strange ...",
                "I spent the holiday of late May to see Rick Steve's recommendations and I loved it.",
            };

            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("------------------------------------------");
            var predict = classification.Predict(test, model);
            foreach (var (data, prediction) in predict)
                Console.WriteLine("Prediction: {0} | Text: '{1}'",
                            (prediction.Class ? classNames[0] : classNames[1]), data.Text);
            Console.WriteLine();
        }
    }

    public class Data
    {
        [Column(ordinal: "0", name: "Text")]
        public string Text;

        [Column(ordinal: "1", name: "Label")]
        public float Sentiment;
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public bool Class;
    }

    public class Classification
    {
        public Classification() { }

        public PredictionModel<Data, Prediction> Train(string datasetPath)
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(datasetPath).CreateFrom<Data>());
            pipeline.Add(new TextFeaturizer("Features", "Text"));
            pipeline.Add(new FastTreeBinaryClassifier { NumLeaves = 8, NumTrees = 5, MinDocumentsInLeafs = 2 });
            var model = pipeline.Train<Data, Prediction>();
            return model;
        }

        public BinaryClassificationMetrics Test(string testDataPath, PredictionModel<Data, Prediction> model)
        {
            var testData = new TextLoader(testDataPath).CreateFrom<Data>();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            return metrics;
        }

        public IEnumerable<(Data, Prediction)> Predict(List<string> predict, PredictionModel<Data, Prediction> model)
            => this.Predict(predict.Select(p => new Data { Text = p }), model);

        public IEnumerable<(Data, Prediction)> Predict(IEnumerable<Data> predicts, PredictionModel<Data, Prediction> model)
        {
            var predictions = model.Predict(predicts);
            var sentiments = predicts.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            return sentiments;
        }
    }
}