import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.error("Missing GEMINI_API_KEY. Copy .env.example to .env and add your key from https://aistudio.google.com/apikey");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

const prompt = process.argv[2] ?? "Say hello in one short sentence.";

try {
  const result = await model.generateContent(prompt);
  console.log(result.response.text());
} catch (err) {
  console.error("Gemini API error:", err.message);
  process.exit(1);
}
