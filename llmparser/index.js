// This file just demoes how to use LLM parser to classify text using LLM

import { LLMParser } from 'llmparser';

// classify text into 'Software Engineering' or 'Head of Community'
const categories = [{
  name: "Software Engineering",
  description: "this job description Software engineers design, develop, and maintain software systems.",
},
{
  name: "Head of Community",
  description: "Data scientists use data to solve problems.",
}];

// fake job posting
const jobPosting = `Head of Community at Notion (View all jobs)
San Francisco, California; New York, New York;
About Us: We're on a mission to make...`;

(async () => {
  // instantiate the parser
  const parser = new LLMParser({
    apiKey: process.env.OPENAI_API_KEY,
    categories,
  });

  // classify the job posting
  const classification = await parser.parse({
    document: jobPosting,
  });

  console.log("classification results: ");
  console.log(classification);
})();