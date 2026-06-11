import "dotenv/config";
import AIService from "./AIService.js";

const ai = new AIService();

const lead = await ai.qualifyLead(`
Name: Jane Doe
Company: Acme Corp (500 employees)
Role: VP of Engineering
Message: Looking for an enterprise plan with SSO and a demo this week.
Budget: $50k/year
`);

console.log("\n--- Lead Qualification ---");
console.log(JSON.stringify(lead, null, 2));

const ticket = await ai.classifySupportTicket(`
I've been charged twice for my subscription and I can't log in anymore.
This is urgent — I need a refund today.
`);

console.log("\n--- Support Ticket ---");
console.log(JSON.stringify(ticket, null, 2));

const email = await ai.draftEmail({
  context: "Customer asked for a project timeline update. Phase 1 is done, Phase 2 starts Monday.",
  intent: "Send a concise status update and confirm next steps",
  recipient: "Alex",
  tone: "friendly and professional",
});

console.log("\n--- Email Draft ---");
console.log(JSON.stringify(email, null, 2));

const extracted = await ai.extractData(
  "Contact John Smith at john@example.com, phone 555-1234. Invoice #9921 for $1,250 due March 15.",
  ["name", "email", "phone", "invoiceNumber", "amount", "dueDate"]
);

console.log("\n--- Data Extraction ---");
console.log(JSON.stringify(extracted, null, 2));
