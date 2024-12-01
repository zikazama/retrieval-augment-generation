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
    chunkSize: 600,
    chunkOverlap: 0,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  return allSplits;
};

const vectorSave = async (splits) => {
  const embeddings = new OllamaEmbeddings();

  // Initialize Chroma DB
  await Chroma.fromDocuments(splits, embeddings, {
    collectionName: "test-cv", // Replace with your preferred collection name
    persistDirectory: "./vector_store",
  });
};

const search = async (question) => {
  const embeddings = new OllamaEmbeddings();

  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "test-cv", // Replace with your collection name
    persistDirectory: "./vector_store",
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

loadAndSplitTheDocs("./files/1-NPS-Steps-to-Success-Resume-Guide.pdf").then(
  (splits) => {
    console.log("split: ");
    console.log(splits);
    vectorSave(splits).then(() => {
      search("Anda adalah seorang interviewer cv professional").then(
        (searches) => {
          console.log("searches: ");
          console.log(searches);
          generatePrompt(
            searches,
            "Anda adalah seorang interviewer cv professional"
          ).then((prompt) => {
            console.log("prompt: ");
            console.log(prompt);
            const promptCombine =
              prompt +
              `
                Silahkan rating cv berikut tiap sectionnya, lalu perbaiki, tulislah contoh perbaikannya
    
                Fauzi Fadhlurrohman
    
                SUMMARY
    
                Experienced software engineer with expertise in full-stack development using PHP,
                JavaScript, and Golang. Proficient in Agile methodologies, continuous integration, and
                test-driven development. Dedicated to delivering high-quality software solutions,
                enhancing customer satisfaction, and reducing errors. Proven ability to reduce project
                timelines by 20%, increase customer satisfaction rates by 25%, and achieve a 30%
                reduction in errors.
    
                SKILLS
    
                ● Backend Frameworks: Codeigniter, Laravel, ExpressJS, NestJS, Springboot
                ● Frontend Technologies: React, Flutter, React Native, NextJS
                ● Cloud Platforms: Google Cloud Platform, Amazon Web Services
                ● DevOps: CI/CD, Docker, GitHub Actions, Jenkins, Kong
                ● Other Skills: Mentoring, Tutoring
    
                WORK EXPERIENCE
    
                Member.id | Backend Developer
                November 2022 - Present
                ● Engineered backend systems for a loyalty app using NestJS, improving response
                times by 30%.
                ● Architected microservice architecture and security systems, reducing security
                incidents by 25%.
    
                Spacetoon (Entertainment Industry) | Full Stack Developer
                August 2023 - March 2024
                ● Created a scalable TVOD application, increasing performance by 200%.
                ● Modernized legacy code, reducing technical debt by 40%.
    
                Digital Envision Pty Ltd | Backend Developer
                December 2021 - November 2022
                ● Enhanced data retrieval speed by 25% with ExpressJS and ORM Sequelize.
                ● Increased code coverage to 90% with Jest.
                ● Reduced deployment errors by 30% through CI/CD automation.
    
                PT. Lapi Divusi | Full Stack Developer
                December 2020 - December 2021
                ● Launched a student admissions application, boosting user engagement by 25%.
                ● Improved MySQL query performance by 15%.
    
                PT. Len Industri (Persero) | Site Manager
                October 2020 – December 2020
                ● Secured 75+ essential permits, expediting project timelines by 20%.
    
                PT. Len Industri (Persero) | Web Developer (Internship)
                September 2019 – January 2020
                ● Built a student management application, increasing efficiency by 30%.
    
                Vortex Store | Freelancer Web Developer
                2017 – present
                ● Promoted on social media, increasing client engagement by 35%.
                ● Enhanced user experience and reduced bounce rates by 20% on WordPress sites.
    
                PROJECT OVERVIEW
    
                ● BDR Polsub (2020): Launched a work-from-home attendance application,
                supporting 500+ daily active users.
                ● Linumuda (2019): Established a community website attracting 10,000+ monthly
                visitors.
                ● PT Guna Cahaya Teknik (2019): Developed a company landing page with
                integrated SEO strategies, increasing site traffic by 50%.
                ● Cibeusi Tourism Web (2018): Created a tourism marketing website, increasing
                tourist engagement by 60%.
                ● Jepara Furniture Web (2017): Built an e-commerce landing page, improving
                online sales by 25%.
                ● PTU Web Contractor (2017): Designed a contracting company landing page,
                increasing client inquiries by 30%.
    
                ACHIEVEMENT
                ● 2024: Mentor of ODP BNI at Rakamin
                ● 2022: Mentor at Bangkit
                ● 2022: Class Instructor at Generasi Gigih 2.0
                ● 2021: Architecting on AWS by Dicoding
                ● 2021: Red Hat OpenShift I: Containers & Kubernetes by Red Hat
                ● 2021: Facilitator for Become a Front-End Web Developer Expert by Dicoding
                ● 2021: Learn Flutter Application Fundamentals by Dicoding
                ● 2021: Practical DevOps with the IBM Cloud by Dicoding
                ● 2021: Become a Front-End Developer Expert by Dicoding
                ● 2020: Third Best Graduate in Information Systems
                ● 2020: Facebook Digital Entrepreneurship by Kominfo
                ● 2020: Database Administration Fundamentals by Microsoft
                ● 2019: Java English Competition Semi-Finalist (Essay Category)
                ● 2019: 3rd Winner of the National Essay Competition of IEC
                ● 2019: MAPRES II Informatics Management
                ● 2019: Software Development Fundamentals by Microsoft
                ● 2018: Best Student Odd Period 2018/2019
                ● 2017: Best Student Odd Period 2017/2018
    
                EDUCATION
    
                ● Binus University
                B.S. in Information Systems (2022 - 2024)
    
                ● State Polytechnic of Subang
                Associate Degree in Information Systems (2017 - 2020)
    
                ORGANIZATION
    
                ● 2022 - Present: Member, Google Developer Student Club Binus
                ● 2021 - Present: Member, Google Developer Group Bandung
                ● 2018 – 2019: Leader, English Students Alliance POLSUB
                ● 2017 – 2018: Member, Majelis Permusyawaratan Mahasiswa POLSUB
                ● 2017 - 2020: Member, Himpunan Mahasiswa Manajemen Informatika POLSUB
                ● 2017 - 2018: Member, PASMA POLSUB
                ● 2017 - 2018: Member, TIM Fight FKMPI
                `;
            generateOutput(promptCombine).then((response) => {
              console.log(response.content);
            });
          });
        }
      );
    });
  }
);
