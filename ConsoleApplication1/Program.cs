using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using OpenCvSharp;
using System.Threading;


namespace CSPredictionSample
{
    static class Program
    {
        static string res = " ";
        static void Main()
        {


            var haarCascade1 = new CascadeClassifier("C:\\Users\\dsouz\\Desktop\\haar\\fist.xml");


            Mat image = new Mat();
            VideoCapture capture = new VideoCapture(0);
            capture.Grab();
            int count = 0;
            while (capture.IsOpened())
            {
                capture.Read(image);



                Scalar cl = new Scalar(0, 0, 255);
                Point p = new Point(100, 100);
                Size s = new Size(150, 150);
                Rect r = new Rect(p, s);
                Cv2.Rectangle(image, r, cl);


                Mat temp = new Mat(image, r);

                count++;
                if(count == 200)
                {

                    new Thread(() => {
                        MakePredictionRequest(temp).Wait();
                     //   Cv2.PutText(image, res, new Point(50, 100), HersheyFonts.HersheyPlain, 2, Scalar.Yellow);


                    }).Start();
                    count = 0;
                }
              Console.WriteLine(count);
                MatOfByte3 mat3 = new MatOfByte3(image); // cv::Mat_<cv::Vec3b>
                var indexer = mat3.GetIndexer();
                for (int x = 100; x < 250; x++)
                {
                    for (int y = 40; y < 90; y++)
                    {
                        Vec3b color = new Vec3b(161,101,0);

                        indexer[y,x] = color;
                    }
                }
                Cv2.PutText(image, res, new Point(150, 80), HersheyFonts.HersheySimplex, 2, Scalar.White);
                Cv2.ImShow("Frame1", image);
                Cv2.ImShow("Frame2", temp);

                Cv2.WaitKey(1);
            }
        }

        static byte[] GetImageAsByteArray(string imageFilePath)
        {
            FileStream fileStream = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read);
            BinaryReader binaryReader = new BinaryReader(fileStream);
            return binaryReader.ReadBytes((int)fileStream.Length);
        }

        static async Task MakePredictionRequest(Mat img)
        {
            var client = new HttpClient();

            // Request headers - replace this example key with your valid subscription key.
            client.DefaultRequestHeaders.Add("Prediction-Key", "a062270b9eeb46109e41fbf97cf586d3");

            // Prediction URL - replace this example URL with your valid prediction URL.
            string url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/52f29cd5-8bd0-452e-b85e-e7a36a76fcee/image?iterationId=41fa7b2f-94d4-4070-a614-b10030e16981";

            HttpResponseMessage response;

            // Request body. Try this sample with a locally stored image.
            byte[] byteData = img.ToBytes();

            using (var content = new ByteArrayContent(byteData))
            {
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                response = await client.PostAsync(url, content);
                string x = await response.Content.ReadAsStringAsync();
                dynamic obj = JObject.Parse(x);
                double maxprob = 0;
                Console.WriteLine(obj);
                foreach (dynamic s in obj.predictions)
                {
                    if (s.probability > maxprob)
                    {
                        res = s.tagName;

                        maxprob = s.probability;
                        Console.WriteLine(maxprob);
                    }

                }
                Console.WriteLine(res);
                Console.ReadLine();
            }
        }
    }
}