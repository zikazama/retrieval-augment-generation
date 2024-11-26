const dotenv = require("dotenv");
dotenv.config();
const { OllamaEmbeddings } = require("@langchain/community/embeddings/ollama");
const { ChatGroq } = require("@langchain/groq");
const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { Document } = require("@langchain/core/documents");
const { PromptTemplate } = require("@langchain/core/prompts");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { formatDocumentsAsString } = require("langchain/util/document")

const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "mixtral-8x7b-32768",
  temperature: 0,
});

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

const formatChatHistory = (chatHistory) => {
  return chatHistory
    .map(
      (turn) => `Human: ${turn[0]}\nAssistant: ${turn[1]}`
    )
    .join("\n");
};

// Initialize OpenAI Embeddings and Chroma VectorStore
const embeddings = new OllamaEmbeddings({
  model: "llama3.2", // Default value
  baseUrl: "http://localhost:11434", // Default value
});

const vectorStore = new Chroma(embeddings, {
  collectionName: "a-test-collection",
  url: "http://localhost:8000", // Optional, will default to this value
  collectionMetadata: {
    "hnsw:space": "cosine", // Specify distance function
  },
});

// Define Documents for Chroma storage
const document1 = new Document({
  pageContent: "The powerhouse of the cell is the mitochondria",
  metadata: { source: "https://example.com" },
});

const document2 = new Document({
  pageContent: "Buildings are made out of brick",
  metadata: { source: "https://example.com" },
});

const document3 = new Document({
  pageContent: "Mitochondria are made out of lipids",
  metadata: { source: "https://example.com" },
});

const document4 = new Document({
  pageContent: "The 2024 Olympics are in Paris",
  metadata: { source: "https://example.com" },
});

const document5 = new Document({
  pageContent: "Fauzi is a person have a job as a programmer",
  metadata: { source: "https://example.com" },
});

// Array of documents
const documents = [document1, document2, document3, document4, document5];

// Add documents to Chroma
async function addDocumentsToChroma() {
  await vectorStore.addDocuments(documents, { ids: ["1", "2", "3", "4", "5"] });
}

addDocumentsToChroma().then(() => {
  const retriever = vectorStore.asRetriever(
    {
      filter: { source: "https://example.com" },
      k: 2,
    }
  );

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

  const conversationalRetrievalQAChain =
    standaloneQuestionChain.pipe(answerChain);

  // Query example 1
  conversationalRetrievalQAChain.invoke({
    question: "Who is fauzi?",
    chat_history: [],
  }).then(result1 => {
    console.log(result1.content);
    /*
      AIMessage { content: "The powerhouse of the cell is the mitochondria." }
    */
  });

});
