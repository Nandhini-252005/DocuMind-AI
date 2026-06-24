import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def ask_groq(question: str, context_chunks: list[dict]) -> str:
    """Friendly conversational answer using explicitly tagged document source file metadata."""
    if not context_chunks:
        return "Hmm, I couldn't find relevant content in the document for that. Try rephrasing! 😊"

    # CRUCIAL MULTI-DOC ENHANCEMENT: Injects clear filename markers directly above the text
    context_text = "\n\n---\n\n".join(
        [f"[DOCUMENT SOURCE FILE: {c.get('doc_name','Unknown File')}]:\n{c['text']}"
         for c in context_chunks]
    )

    prompt = f"""You are DocuMind AI — a friendly, helpful, conversational document assistant.

You are orchestrating responses inside a Combined Multi-Document Workspace. The user can reference specific documents by their exact filenames (e.g., 'Nandhini_certificate', 'Nandhini_N_Resume'), text locations, or relative file ordering. 

Always use the [DOCUMENT SOURCE FILE: ...] metadata header tags provided inside the DOCUMENT CONTEXT block to trace and isolate information matching the user's specific document constraints.

YOUR PERSONALITY:
- Warm, approachable, encouraging — like a knowledgeable friend
- Use natural conversational language, not robotic bullet lists
- Add light friendliness: "Great question!", "Sure!", "Happy to help!" where natural
- For greetings or thanks — respond warmly without mentioning the document
- Use emojis occasionally 😊
- Keep answers concise but complete
- If the user asks about a filename that is absent from the provided metadata context headers, let them know what files you DO see available in the current context blocks.
- If not in document context: "Hmm, I couldn't find that in the documents — feel free to ask anything else! 😊"

DOCUMENT CONTEXT:
{context_text}

USER MESSAGE: {question}

YOUR FRIENDLY RESPONSE:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.6
    )
    return response.choices[0].message.content.strip()


def generate_summary(chunks: list[dict]) -> str:
    """Auto-generate a document summary from first chunks."""
    sample_text = " ".join([c["text"] for c in chunks[:8]])[:3000]

    prompt = f"""You are DocuMind AI. Summarize this document clearly and concisely.

Format your response EXACTLY like this:
📄 **Document Overview**
[2-3 sentence overview of what this document is about]

🔑 **Key Points**
• [Point 1]
• [Point 2]
• [Point 3]
• [Point 4]
• [Point 5]

💡 **Main Takeaway**
[One sentence — the most important thing to know]

DOCUMENT TEXT:
{sample_text}

SUMMARY:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


def generate_quiz(chunks: list[dict], num_questions: int = 5) -> list[dict]:
    """Generate MCQ quiz from document content."""
    sample_text = " ".join([c["text"] for c in chunks])[:4000]

    prompt = f"""You are DocuMind AI. Create {num_questions} multiple choice questions from this document.

Return ONLY valid JSON array, no other text:
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "A) Option 1",
    "explanation": "Brief explanation why this is correct"
  }}
]

DOCUMENT TEXT:
{sample_text}

JSON:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.4
    )
    raw = response.choices[0].message.content.strip()

    # Clean and parse JSON
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return [{"question": "Quiz generation failed — please try again.", "options": [], "answer": "", "explanation": ""}]