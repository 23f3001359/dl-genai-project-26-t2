import gradio as gr
import pandas as pd
import torch
import random
import os

from transformers import AutoTokenizer, AutoModelForMultipleChoice

# CONFIGURATION

CSV_PATH = "train.csv"
MODEL_PATH = "./debertav3"

NUM_QUESTIONS = 10
MAX_LENGTH = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIONS = ["A", "B", "C", "D", "E"]

LABEL_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}



# LOAD DATA


df = pd.read_csv(CSV_PATH)

required_columns = ["prompt", "A", "B", "C", "D", "E"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"CSV is missing these required columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns).reset_index(drop=True)

if len(df) < NUM_QUESTIONS:
    raise ValueError(
        f"Dataset must contain at least {NUM_QUESTIONS} questions."
    )

HAS_ANSWER_COLUMN = "answer" in df.columns

print(f"Loaded {len(df)} questions.")

if HAS_ANSWER_COLUMN:
    print("Answer column detected.")
else:
    print(
        "Warning: No answer column detected. "
        "Accuracy scores will not be calculated."
    )



# LOAD MODEL


print(f"Loading model on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")



# ANSWER NORMALIZATION


def normalize_answer(answer, option_texts=None):
    """
    Converts possible answer formats into A/B/C/D/E.

    Supported examples:
        A
        a
        Option A
        Answer: A
        1
        2
        Full option text
    """

    if answer is None:
        return None

    if pd.isna(answer):
        return None

    value = str(answer).strip().upper()

    if not value:
        return None

    if value in OPTIONS:
        return value

    cleaned = (
        value.replace("ANSWER:", "")
        .replace("ANSWER", "")
        .replace("OPTION:", "")
        .replace("OPTION", "")
        .strip()
    )

    if cleaned in OPTIONS:
        return cleaned

    
    if cleaned in {"1", "2", "3", "4", "5"}:
        return OPTIONS[int(cleaned) - 1]

    
    if option_texts is not None:
        normalized_options = [
            str(option).strip().upper()
            for option in option_texts
        ]

        if value in normalized_options:
            return OPTIONS[normalized_options.index(value)]

    return None

def predict_answer(question, options):
    """
    Returns:
        predicted_option: A/B/C/D/E
        confidence: confidence percentage
        top3_string: top-three prediction display
    """

    texts = [
        f"{question} [SEP] {options[i]}"
        for i in range(len(options))
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(DEVICE)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)

    predicted_index = torch.argmax(probabilities).item()
    predicted_option = OPTIONS[predicted_index]

    confidence = probabilities[predicted_index].item() * 100

    top_k = min(3, len(OPTIONS))
    top_values, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    top3_lines = []

    for probability, index in zip(top_values, top_indices):
        option = OPTIONS[index.item()]
        percentage = probability.item() * 100

        top3_lines.append(
            f"{option} — {percentage:.2f}%"
        )

    top3_string = "\n".join(
        f"{i + 1}. {line}"
        for i, line in enumerate(top3_lines)
    )

    return predicted_option, confidence, top3_string



# QUIZ STATE


def create_quiz():
    selected_indices = random.sample(
        range(len(df)),
        NUM_QUESTIONS,
    )

    return {
        "questions": selected_indices,
        "current": 0,
        "human_score": 0,
        "ai_score": 0,
        "agreements": 0,
        "results": [],
    }


def get_question(quiz):
    index = quiz["questions"][quiz["current"]]
    row = df.iloc[index]

    question_number = quiz["current"] + 1

    question_text = (
        f"### Question {question_number} / {NUM_QUESTIONS}\n\n"
        f"{row['prompt']}"
    )

    options_text = (
        f"**A.** {row['A']}\n\n"
        f"**B.** {row['B']}\n\n"
        f"**C.** {row['C']}\n\n"
        f"**D.** {row['D']}\n\n"
        f"**E.** {row['E']}"
    )

    return question_text, options_text



# START QUIZ


def start_quiz():
    quiz = create_quiz()

    question_text, options_text = get_question(quiz)

    return (
        quiz,
        question_text,
        options_text,
        gr.update(value=None),
        "",
        "",
        "",
        "",
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )



# SUBMIT HUMAN ANSWER


def submit_answer(human_answer, quiz):
    if quiz is None:
        return (
            "Please start a quiz first.",
            "",
            "",
            "",
            "",
            quiz,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    if human_answer is None:
        return (
            "⚠️ Please select an answer before submitting.",
            "",
            "",
            "",
            "",
            quiz,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    current_index = quiz["questions"][quiz["current"]]
    row = df.iloc[current_index]

    question = str(row["prompt"])

    options = [
        str(row["A"]),
        str(row["B"]),
        str(row["C"]),
        str(row["D"]),
        str(row["E"]),
    ]

    human_answer = normalize_answer(
        human_answer,
        option_texts=options,
    )

    if human_answer is None:
        return (
            "⚠️ Invalid answer selected.",
            "",
            "",
            "",
            "",
            quiz,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    # Read and normalize the correct answer from the CSV
    correct_answer = None

    if HAS_ANSWER_COLUMN:
        correct_answer = normalize_answer(
            row["answer"],
            option_texts=options,
        )

    # Run AI prediction
    ai_answer, confidence, top3 = predict_answer(
        question,
        options,
    )

    # Compare human and AI answers
    agreement = human_answer == ai_answer

    if agreement:
        quiz["agreements"] += 1

    # Calculate scores only when a valid answer exists
    if correct_answer is not None:
        if human_answer == correct_answer:
            quiz["human_score"] += 1

        if ai_answer == correct_answer:
            quiz["ai_score"] += 1

    quiz["results"].append(
        {
            "question": question,
            "human": human_answer,
            "ai": ai_answer,
            "confidence": confidence,
            "correct": correct_answer,
            "agreement": agreement,
        }
    )

    if agreement:
        comparison = "### 🤝 Human and AI agree!"
    else:
        comparison = "### ⚔️ Human and AI disagree!"

    human_display = f"**Your choice:** {human_answer}"

    ai_display = (
        f"**AI choice:** {ai_answer}\n\n"
        f"**AI confidence:** {confidence:.2f}%"
    )

    top3_display = (
        "### AI Top 3 predictions\n\n"
        f"{top3}"
    )

    if correct_answer is not None:
        human_result = (
            "✅ Correct"
            if human_answer == correct_answer
            else f"❌ Incorrect — correct answer: **{correct_answer}**"
        )

        ai_result = (
            "✅ AI Correct"
            if ai_answer == correct_answer
            else f"❌ AI Incorrect — correct answer: **{correct_answer}**"
        )

        score_display = (
            f"**Your result:** {human_result}\n\n"
            f"**AI result:** {ai_result}"
        )
    else:
        score_display = (
            "ℹ️ The answer column is missing or contains an invalid "
            "answer format. Human-versus-AI comparison is still available."
        )

    is_last_question = (
        quiz["current"] == len(quiz["questions"]) - 1
    )

    if is_last_question:
        next_button_update = gr.update(visible=False)
        final_button_update = gr.update(visible=True)

        print(
            "Final question submitted. "
            "Showing final-results button."
        )
    else:
        next_button_update = gr.update(visible=True)
        final_button_update = gr.update(visible=False)

    return (
        comparison,
        human_display,
        ai_display,
        top3_display,
        score_display,
        quiz,
        next_button_update,
        final_button_update,
    )



# NEXT QUESTION


def next_question(quiz):
    if quiz is None:
        return (
            quiz,
            "Please start a quiz first.",
            "",
            gr.update(value=None),
            "",
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    if quiz["current"] >= len(quiz["questions"]) - 1:
        return (
            quiz,
            "The quiz is complete. Click **View Final Results**.",
            "",
            gr.update(value=None),
            "",
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    quiz["current"] += 1

    question_text, options_text = get_question(quiz)

    return (
        quiz,
        question_text,
        options_text,
        gr.update(value=None),
        "",
        "",
        "",
        "",
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )



# FINAL RESULTS


def show_final_results(quiz):
    if quiz is None:
        return "⚠️ No quiz data found. Please start a quiz first."

    answered_questions = len(quiz["results"])

    if answered_questions == 0:
        return "⚠️ No answers have been submitted yet."

    human_score = quiz["human_score"]
    ai_score = quiz["ai_score"]
    agreements = quiz["agreements"]

    human_percentage = human_score / answered_questions * 100
    ai_percentage = ai_score / answered_questions * 100
    agreement_percentage = agreements / answered_questions * 100

    if human_score > ai_score:
        winner = "🏆 **Human wins!**"
    elif ai_score > human_score:
        winner = "🤖 **AI wins!**"
    else:
        winner = "🤝 **It's a tie!**"

    result = f"""
# 🎉 Final Results

{winner}

## Score

| Player | Score | Accuracy |
|---|---:|---:|
| 👤 Human | **{human_score}/{answered_questions}** | **{human_percentage:.1f}%** |
| 🤖 AI | **{ai_score}/{answered_questions}** | **{ai_percentage:.1f}%** |

## Human vs AI

- Questions where Human and AI agreed: **{agreements}/{answered_questions}**
- Agreement rate: **{agreement_percentage:.1f}%**

---

## Question Breakdown
"""

    for i, item in enumerate(quiz["results"]):
        agreement_icon = "🤝" if item["agreement"] else "⚔️"

        if item["correct"] is None:
            human_correct = "?"
            ai_correct = "?"
            correct_display = "Unavailable"
        else:
            human_correct = (
                "✅"
                if item["human"] == item["correct"]
                else "❌"
            )

            ai_correct = (
                "✅"
                if item["ai"] == item["correct"]
                else "❌"
            )

            correct_display = item["correct"]

        result += f"""
### Question {i + 1} {agreement_icon}

**Human:** `{item["human"]}` {human_correct}

**AI:** `{item["ai"]}` {ai_correct}

**Correct answer:** `{correct_display}`

**AI confidence:** `{item["confidence"]:.2f}%`

---
"""

    return result



# GRADIO USER INTERFACE


with gr.Blocks(
    title="AI vs Human — MCQ Challenge"
) as app:

    gr.Markdown(
        """
# 🧠 AI vs Human — MCQ Challenge

**Can you beat the AI?**

You will be given 10 randomly selected questions.
Answer each question before seeing the model's prediction.
"""
    )

    quiz_state = gr.State(None)

    question_display = gr.Markdown(
        "Click **Start New Quiz** to begin."
    )

    options_display = gr.Markdown()

    human_answer = gr.Radio(
        choices=OPTIONS,
        label="Your Answer",
        interactive=True,
    )

    with gr.Row():
        start_button = gr.Button(
            "🎯 Start New Quiz",
            variant="primary",
        )

        submit_button = gr.Button(
            "Submit Answer",
            variant="secondary",
        )

        next_button = gr.Button(
            "Next Question →",
            visible=False,
        )

        final_button = gr.Button(
            "🏁 View Final Results (Scroll Down Further to view it)",
            visible=False,
        )

    comparison_display = gr.Markdown()
    human_display = gr.Markdown()
    ai_display = gr.Markdown()
    top3_display = gr.Markdown()
    score_display = gr.Markdown()

    final_results = gr.Markdown()

    # START BUTTON   

    start_button.click(
        fn=start_quiz,
        inputs=[],
        outputs=[
            quiz_state,
            question_display,
            options_display,
            human_answer,
            comparison_display,
            human_display,
            ai_display,
            top3_display,
            score_display,
            final_results,
            next_button,
            submit_button,
            final_button,
        ],
    )    
    # SUBMIT BUTTON   

    submit_button.click(
        fn=submit_answer,
        inputs=[
            human_answer,
            quiz_state,
        ],
        outputs=[
            comparison_display,
            human_display,
            ai_display,
            top3_display,
            score_display,
            quiz_state,
            next_button,
            final_button,
        ],
    )    
    # NEXT BUTTON  

    next_button.click(
        fn=next_question,
        inputs=[
            quiz_state,
        ],
        outputs=[
            quiz_state,
            question_display,
            options_display,
            human_answer,
            comparison_display,
            human_display,
            ai_display,
            top3_display,
            score_display,
            final_results,
            next_button,
            submit_button,
            final_button,
        ],
    )

    # FINAL RESULTS BUTTON
    
    final_button.click(
        fn=show_final_results,
        inputs=[
            quiz_state,
        ],
        outputs=[
            final_results,
        ],
    )
# LAUNCH

if __name__ == "__main__":
    app.launch()

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 7860))
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=port
#     )