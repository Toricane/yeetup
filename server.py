from flask import Flask, jsonify, make_response, render_template, request

from functions import (
    filter_and_rank_events,
    load_events,
    refine_question,
    respond_to_question,
    transcribe,
)

app = Flask(__name__)
events = load_events("events.json")
histories = {}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("question")
        session_id = request.cookies.get("session_id")
        if session_id is None:
            import uuid

            session_id = str(uuid.uuid4())
        if session_id not in histories:
            histories[session_id] = []

        refined_query = refine_question(user_input)
        filtered_events = filter_and_rank_events(events, refined_query)
        response = respond_to_question(
            user_input, refined_query, filtered_events, histories[session_id]
        )
        histories[session_id].append({"role": "assistant", "content": response})
        histories[session_id].append({"role": "user", "content": user_input})
        return jsonify({"response": response, "session_id": session_id})

    session_id = request.cookies.get("session_id")
    if session_id is None:
        import uuid

        session_id = str(uuid.uuid4())
    resp = make_response(render_template("index.html", session_id=session_id))
    resp.set_cookie("session_id", session_id)
    return resp


@app.route("/process_audio", methods=["POST"])
def process_audio():
    transcription = transcribe(request.files["audio"].read())
    print(transcription)
    return transcription


if __name__ == "__main__":
    app.run(debug=True)
