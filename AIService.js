import { GoogleGenerativeAI } from "@google/generative-ai";
import { z } from "zod";

const MAX_RETRIES = 3;
const LOW_CONFIDENCE_THRESHOLD = 0.7;
const DEFAULT_MODEL = process.env.GEMINI_MODEL ?? "gemini-2.0-flash";

const confidenceSchema = z.number().min(0).max(1);

export const LeadQualificationSchema = z.object({
  qualified: z.boolean(),
  leadScore: z.number().min(0).max(100),
  reason: z.string().min(1),
  recommendedAction: z.enum(["contact", "nurture", "disqualify", "follow_up"]),
  tags: z.array(z.string()),
  confidence: confidenceSchema,
});

export const SupportTicketSchema = z.object({
  category: z.enum([
    "billing",
    "technical",
    "account",
    "feature_request",
    "shipping",
    "other",
  ]),
  priority: z.enum(["low", "medium", "high", "urgent"]),
  sentiment: z.enum(["positive", "neutral", "negative", "frustrated"]),
  summary: z.string().min(1),
  suggestedTeam: z.string().min(1),
  confidence: confidenceSchema,
});

export const EmailDraftSchema = z.object({
  subject: z.string().min(1),
  body: z.string().min(1),
  tone: z.string().min(1),
  confidence: confidenceSchema,
});

export const DataExtractionSchema = z.object({
  extractedData: z.record(
    z.union([z.string(), z.number(), z.boolean(), z.null()])
  ),
  fieldsFound: z.array(z.string()),
  missingFields: z.array(z.string()),
  confidence: confidenceSchema,
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function enrichWithReviewFlag(data) {
  return {
    ...data,
    needsHumanReview: data.confidence < LOW_CONFIDENCE_THRESHOLD,
  };
}

function extractJson(text) {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  return JSON.parse(candidate);
}

function formatZodError(error) {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "root"}: ${issue.message}`)
    .join("; ");
}

export class AIService {
  #model;

  constructor(apiKey = process.env.GEMINI_API_KEY, model = DEFAULT_MODEL) {
    if (!apiKey) {
      throw new Error(
        "Missing GEMINI_API_KEY. Set it in .env or pass it to the AIService constructor."
      );
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    this.#model = genAI.getGenerativeModel({
      model,
      generationConfig: {
        responseMimeType: "application/json",
        temperature: 0.2,
      },
    });
  }

  async #generateValidatedJson(prompt, schema, taskName) {
    let lastError;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const result = await this.#model.generateContent(prompt);
        const rawText = result.response.text();
        const parsed = extractJson(rawText);
        const validated = schema.parse(parsed);
        return enrichWithReviewFlag(validated);
      } catch (err) {
        lastError = err;

        if (err instanceof z.ZodError) {
          lastError = new Error(
            `${taskName} validation failed on attempt ${attempt}: ${formatZodError(err)}`
          );
        } else if (!(err instanceof SyntaxError)) {
          lastError = new Error(
            `${taskName} API call failed on attempt ${attempt}: ${err.message}`
          );
        } else {
          lastError = new Error(
            `${taskName} returned invalid JSON on attempt ${attempt}: ${err.message}`
          );
        }

        if (attempt < MAX_RETRIES) {
          await sleep(500 * attempt);
        }
      }
    }

    throw lastError;
  }

  /**
   * Score and qualify a sales lead from raw lead details.
   */
  async qualifyLead(leadDetails) {
    const prompt = `
You are a B2B sales lead qualification assistant.

Analyze the lead information and return ONLY valid JSON with this shape:
{
  "qualified": boolean,
  "leadScore": number (0-100),
  "reason": string,
  "recommendedAction": "contact" | "nurture" | "disqualify" | "follow_up",
  "tags": string[],
  "confidence": number (0-1, your confidence in this assessment)
}

Lead information:
${leadDetails}
`.trim();

    return this.#generateValidatedJson(prompt, LeadQualificationSchema, "Lead qualification");
  }

  /**
   * Classify a customer support ticket.
   */
  async classifySupportTicket(ticketText) {
    const prompt = `
You are a customer support triage assistant.

Classify the ticket and return ONLY valid JSON with this shape:
{
  "category": "billing" | "technical" | "account" | "feature_request" | "shipping" | "other",
  "priority": "low" | "medium" | "high" | "urgent",
  "sentiment": "positive" | "neutral" | "negative" | "frustrated",
  "summary": string,
  "suggestedTeam": string,
  "confidence": number (0-1, your confidence in this classification)
}

Support ticket:
${ticketText}
`.trim();

    return this.#generateValidatedJson(
      prompt,
      SupportTicketSchema,
      "Support ticket classification"
    );
  }

  /**
   * Draft a professional email from context and intent.
   */
  async draftEmail({ context, intent, recipient = "there", tone = "professional" }) {
    const prompt = `
You are a professional email writing assistant.

Draft an email and return ONLY valid JSON with this shape:
{
  "subject": string,
  "body": string,
  "tone": string,
  "confidence": number (0-1, your confidence that this draft fits the request)
}

Recipient: ${recipient}
Desired tone: ${tone}
Intent: ${intent}
Context:
${context}
`.trim();

    return this.#generateValidatedJson(prompt, EmailDraftSchema, "Email drafting");
  }

  /**
   * Extract structured fields from unstructured raw text.
   */
  async extractData(rawText, fields = []) {
    const fieldList =
      fields.length > 0
        ? fields.join(", ")
        : "name, email, phone, company, date, amount, address, notes";

    const prompt = `
You are a data extraction assistant.

Extract structured information from the text below.
Target fields: ${fieldList}

Return ONLY valid JSON with this shape:
{
  "extractedData": { "<fieldName>": string | number | boolean | null, ... },
  "fieldsFound": string[],
  "missingFields": string[],
  "confidence": number (0-1, your confidence in the extracted values)
}

Use null for fields that are not present. Do not invent data.

Raw text:
${rawText}
`.trim();

    return this.#generateValidatedJson(prompt, DataExtractionSchema, "Data extraction");
  }
}

export default AIService;
