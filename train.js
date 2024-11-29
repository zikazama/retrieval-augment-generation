const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const dotenv = require("dotenv");
dotenv.config();

const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");
const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { Document } = require("@langchain/core/documents");

// Initialize embeddings and ChromaDB
const embeddings = new OllamaEmbeddings({
  model: "llama3.2", // Specify the embedding model
  baseUrl: "http://localhost:11434", // Default local Ollama server
});

const vectorStore = new Chroma(embeddings, {
  collectionName: "cv-knowledge-collection",
  url: "http://localhost:8000", // ChromaDB server URL
  collectionMetadata: { "hnsw:space": "cosine" },
});

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

// Main Execution
(async () => {
  // Step 1: Process PDFs
  const pdfDir = "./files"; // Path to your PDFs
  await processPdfDirectory(pdfDir);

})();
