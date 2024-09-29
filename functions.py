import json
from os import getenv

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq = Groq(api_key=getenv("GROQ"))


def load_events(filepath: str = "events.json") -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def use_groq(**kwargs) -> str:
    completion = groq.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def transcribe(audio: bytes) -> str:
    transcription = groq.audio.transcriptions.create(
        file=("audio.wav", audio), model="whisper-large-v3"
    )
    return transcription.text


model = "llama-3.1-70b-versatile"


def refine_question(question: str) -> dict[str, list[str]]:
    question = question.lower()
    output = use_groq(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'You are an assistant that helps clarify user queries about student events. Extract the key topics and interests from the user\'s input. Do not answer the question directly. Just provide the keywords related to the types of events the user is interested in.\n\nThe keywords, or categories, are more general but some examples include: hackathon, festival, social, music, clubs, community, technology, coding, study, academics, math, science, gaming, competition, etc.\n\nTarget audiences can be either highschool or university. Specify both if the audience is not specified.\n\nIf the user specifies a specific audience, like a specific high school or university, then include it as part of specific audience. If not, then write "any".\n\nOutput your response in JSON format with the following keys: "categories", "target_audience", "specific_audience".\n\nExample:\n```json\n{\n  "categories": ["hackathon", "technology", "coding"],\n  "target_audience": ["highschool"],\n  "specific_audience": ["any"]\n}\n```',
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0,
        max_tokens=300,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {
            "categories": [],
            "target_audience": [],
            "specific_audience": [],
        }


def filter_and_rank_events(
    events: list[dict], refined_query: dict[str, list[str]]
) -> list[dict]:
    filtered_events = []
    for event in events:
        score = 0
        for category in refined_query["categories"]:
            if category in event.get("category", []) or category in event.get(
                "tags", []
            ):
                score += 2
            if (
                category in event.get("name", "").lower()
                or category in event.get("description", "").lower()
            ):
                score += 1

        if score > 0 and (
            "any" in refined_query["target_audience"]
            or any(
                audience == a["level"]
                for a in event.get("targetAudience", [])
                for audience in refined_query["target_audience"]
            )
        ):
            filtered_events.append({"event": event, "score": score})

    filtered_events.sort(key=lambda x: x["score"], reverse=True)

    return [item["event"] for item in filtered_events]


def respond_to_question(
    question: str,
    refined: dict[str, list[str]],
    filtered_events: list[dict],
    history: list[dict],
) -> str:
    history.append({"role": "user", "content": question})

    if not filtered_events:
        return "I'm sorry, I couldn't find any events that match your query."

    messages = [
        {
            "role": "system",
            "content": 'You are the AI of yeetup.com - a place for youth, such as high school and university students, to find meetups and events relevant to them.\n\nIn the next message, you will be provided 3 items:\n1. The question or statement from the user. This is what the user is looking for.\n2. The keywords associated with that question, which are computed by an algorithm.\n3. The relevant events, filtered from a repository of events using the generated keywords from the question.\n\nI want to know one thing:\n- If the question is relevant to finding events or describing what the user is interested in, and the keywords are accurately describing what events to look for, and the provided events exist and are relevant to the question, respond "yes".\n- Otherwise, respond, "no".',
        },
        *history[:-1],
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nKeywords:\n{refined}\n\nEvents:\n{filtered_events}",
        },
    ]

    response = use_groq(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1,
        top_p=1,
        stream=False,
        stop=None,
    )

    if response == "no":
        return "I'm sorry, I couldn't find any events that match your query."

    messages = [
        {
            "role": "system",
            "content": "You are the AI of yeetup.com - a place for youth, such as high school and university students, to find meetups and events relevant to them. You are a friendly and helpful chatbot designed for high school and university students. Your goal is to help them discover events and opportunities related to their interests. Respond concisely and informatively, providing relevant event information in a clear and easy-to-understand format. Use encouraging language and ask clarifying questions to guide the user and ensure you're providing the most helpful recommendations. Maintain an approachable tone, similar to a helpful friend, while still being professional and credible.\n\nIn the next message, you will be provided 3 items:\n1. The question or statement from the user. This is what the user is looking for.\n2. The keywords associated with that question, which are computed by an algorithm.\n3. The relevant events, filtered from a repository of events using the generated keywords from the question.\n\nPlease respond to the user with the information provided.",
        },
        *history[:-1],
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nKeywords:\n{refined}\n\nEvents:\n{filtered_events}",
        },
    ]

    response = use_groq(
        model=model,
        messages=messages,
        temperature=1,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response
