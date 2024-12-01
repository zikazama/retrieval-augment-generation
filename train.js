const fs = require("fs");
const path = require("path");
const { PDFLoader } = require("langchain/document_loaders");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitters");
const { OllamaEmbeddings } = require("langchain/embeddings");
const { Chroma } = require("langchain/vectorstores");

// Function to load and split documents in a folder
const loadAndSplitDocsInFolder = async (folderPath) => {
  // Get all files in the folder
  const files = fs.readdirSync(folderPath).filter((file) => file.endsWith(".pdf"));
  let allSplits = [];

  for (const file of files) {
    const filePath = path.join(folderPath, file);

    console.log(`Processing file: ${filePath}`);

    // Load the PDF
    const loader = new PDFLoader(filePath);
    const docs = await loader.load();

    // Split the documents into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600, // Set chunk size
      chunkOverlap: 0, // No overlap
    });
    const splits = await textSplitter.splitDocuments(docs);

    // Combine all splits
    allSplits = allSplits.concat(splits);
  }

  console.log(`Total splits created: ${allSplits.length}`);
  return allSplits;
};

// Function to save splits into a ChromaDB vector store
const vectorSave = async (splits) => {
  const embeddings = new OllamaEmbeddings();

  // Initialize Chroma DB
  await Chroma.fromDocuments(splits, embeddings, {
    collectionName: "test-cv", // Replace with your preferred collection name
    persistDirectory: "./vector_store", // Directory for persistent storage
  });

  console.log("Data saved to ChromaDB.");
};