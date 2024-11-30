const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { PDFLoader } = require("@langchain/community/document_loaders/fs/pdf");
const { OllamaEmbeddings, ChatOllama } = require("@langchain/ollama");
const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { PromptTemplate } = require("@langchain/core/prompts");

const loadAndSplitTheDocs = async (file_path) => {
  // load the uploaded file data
  const loader = new PDFLoader(file_path);
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  return allSplits;
};

const vectorSaveAndSearch = async (splits, question) => {
  const embeddings = new OllamaEmbeddings();

  // Initialize Chroma DB
  const vectorStore = await Chroma.fromDocuments(splits, embeddings, {
    collectionName: "my_documents", // Replace with your preferred collection name
  });

  // Perform similarity search
  const searches = await vectorStore.similaritySearch(question, 4); // 4: Number of results
  return searches;
};

const generatePrompt = async (searches, question) => {
  let context = "";
  searches.forEach((search) => {
    context = context + "\n\n" + search.pageContent;
  });

  const prompt = PromptTemplate.fromTemplate(`
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
`);

  const formattedPrompt = await prompt.format({
    context: context,
    question: question,
  });
  return formattedPrompt;
};

const generateOutput = async (prompt) => {
  const ollamaLlm = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "llama3.2", // Default value
  });

  const response = await ollamaLlm.invoke(prompt);
  return response;
};

const result = generateOutput('bagaimana cara membuat cv ?');
result.then((res) => {
  console.log(res.content);
});