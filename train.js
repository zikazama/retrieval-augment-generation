const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const dotenv = require("dotenv");
dotenv.config();

const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");
const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { Document } = require("@langchain/core/documents");
const { PromptTemplate } = require("@langchain/core/prompts");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { formatDocumentsAsString } = require("langchain/util/document");
const { ChatGroq } = require("@langchain/groq");

// Initialize embeddings and ChromaDB
const embeddings = new OllamaEmbeddings({
  model: "llama3.2", // Specify the embedding model
  baseUrl: "http://localhost:11434", // Default local Ollama server
});

const vectorStore = new Chroma(embeddings, {
  collectionName: "pdf_collection",
  url: "http://localhost:8000", // ChromaDB server URL
  collectionMetadata: { "hnsw:space": "cosine" },
});

// Initialize ChatGroq model
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "mixtral-8x7b-32768",
  temperature: 0,
});

// Prompts
const condenseQuestionTemplate = `Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(
  condenseQuestionTemplate
);

const answerTemplate = `Answer the question based only on the following context:
{context}

Question: {question}`;

const ANSWER_PROMPT = PromptTemplate.fromTemplate(answerTemplate);

const formatChatHistory = (chatHistory) =>
  chatHistory
    .map((turn) => `Human: ${turn[0]}\nAssistant: ${turn[1]}`)
    .join("\n");

// Function to extract text from a PDF
async function extractTextFromPdf(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = await pdfParse(dataBuffer);
  return pdfData.text; // Extracted text
}

// Function to split text into chunks
function splitTextIntoChunks(text, chunkSize = 500) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

// Function to process a single PDF
async function processPdf(pdfPath) {
  console.log(`Processing: ${pdfPath}`);

  // Step 1: Extract text from PDF
  const text = await extractTextFromPdf(pdfPath);

  // Step 2: Split text into chunks
  const chunks = splitTextIntoChunks(text);

  // Step 3: Convert chunks into Document format
  const documents = chunks.map(
    (chunk, idx) =>
      new Document({
        pageContent: chunk,
        metadata: { source: pdfPath, chunkIndex: idx },
      })
  );

  // Step 4: Store in ChromaDB
  const ids = documents.map((_, idx) => `${pdfPath}_chunk_${idx}`);
  await vectorStore.addDocuments(documents, { ids });

  console.log(`Finished processing: ${pdfPath}`);
}

// Function to process a directory of PDFs
async function processPdfDirectory(pdfDir) {
  const pdfFiles = fs.readdirSync(pdfDir).filter((file) => file.endsWith(".pdf"));

  for (const pdfFile of pdfFiles) {
    const pdfPath = path.join(pdfDir, pdfFile);
    await processPdf(pdfPath);
  }

  console.log("All PDFs processed successfully.");
}

// Function to query ChromaDB
async function queryChroma(question, chatHistory = []) {
  const retriever = vectorStore.asRetriever();

  const standaloneQuestionChain = RunnableSequence.from([
    {
      question: (input) => input.question,
      chat_history: (input) => formatChatHistory(input.chat_history),
    },
    CONDENSE_QUESTION_PROMPT,
    model,
    new StringOutputParser(),
  ]);

  const answerChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
  ]);

  const conversationalRetrievalQAChain = standaloneQuestionChain.pipe(
    answerChain
  );

  const result = await conversationalRetrievalQAChain.invoke({
    question,
    chat_history: chatHistory,
  });

  console.log(result.content);
}

// Main Execution
(async () => {
  // Step 1: Process PDFs
  const pdfDir = "./files"; // Path to your PDFs
  await processPdfDirectory(pdfDir);

  // Step 2: Query Example
  await queryChroma("What is the powerhouse of the cell?", []);
})();
