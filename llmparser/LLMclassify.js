import { LLMParser } from 'llmparser';
const fs = require("fs");

// classify text into 'World', Tech etc
const categories = [{
  name: "World News",
  description: "the news articles is related to world news.",
},
{
  name: "Sports News",
  description: "the news articles is related to sports or games related news.",
},
{
  name: "Business News",
  description: "the news articles is related to business related news.",
},
{
  name: "Science or Tech News",
  description: "the news articles is related to Science or technology related news.",
}];



// read the contents of the text file - this contains the training texts ~100 randomly chosen data points for LLM annotation
fs.readFile('texts_sample.txt', function (err, data1) {
  if (err) {
    console.error(err);
    return;
  }

  const data = data1.toString();

  let array = data.split("\n\n")

  // instantiate the parser
  const parser = new LLMParser({
    apiKey: process.env.OPENAI_API_KEY,
    categories,
  });

for (let i = 0; i < array.length; i++) {

  console.log(array[i]);

  setTimeout(() => {

  // classify the job posting
  // parser.parse({ document: jobPosting })
  parser.parse({ document: array[i] })
    .then(classification => {
      console.log("classification results: " + i);
      console.log(classification);
    })
    .catch(error => {
      console.error(error);
    });
  }, 5000);
  }
});