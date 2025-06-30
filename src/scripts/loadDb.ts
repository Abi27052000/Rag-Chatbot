import { GoogleGenerativeAI } from "@google/generative-ai";
import "dotenv/config";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";

type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

const {
  GEMINI_API_KEY,
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
} = process.env;

// Initialize Google Generative AI
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

const f1Data = [
  "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship",
  "https://en.wikipedia.org/wiki/2026_Formula_One_World_Championship",
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const createCollection = async (
  similarityMetric: SimilarityMetric = "dot_product"
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 768,
      metric: similarityMetric,
    },
  });

  console.log("Collection created:", res);
};

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true, // Run in headless mode
    },
    gotoOptions: {
      waitUntil: "domcontentloaded", // Wait until DOM is ready
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });

  return (await loader.scrape())?.replace(/<[^>]*>?/gm, "");
};

const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);

  for await (const url of f1Data) {
    console.log(`Processing URL: ${url}`);
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);

    for await (const chunk of chunks) {
      try {
        // Generate embedding using Gemini model
        const embeddingResponse = await genAI
          .getGenerativeModel({ model: "text-embedding-004" })
          .embedContent(chunk);

        // Extract the embedding (array of floats)
        const vector = embeddingResponse.embedding.values;

        // Insert chunk and embedding into Astra DB
        const res = await collection.insertOne({
          text: chunk,
          $vector: vector,
        });
        console.log(res);

        console.log(`Inserted chunk from ${url}`);
      } catch (error) {
        console.error(`Error processing chunk from ${url}:`, error);
      }
    }
  }

  console.log("Data loading complete.");
};

createCollection()
  .then(() => loadSampleData())
  .catch((error) => console.error("Error:", error));
