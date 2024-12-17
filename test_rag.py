from query_data import query_rag
from langchain_community.llms.ollama import Ollama


#using local llm for unit testing

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )

def test_3gpp_ts_23_003_1():
    assert query_and_validate(
        question="What are the three main components of an International Mobile Subscriber Identity (IMSI)? How do these components help in identifying a subscriber?",
        expected_response="Mobile Country Code (MCC): A three-digit code that uniquely identifies the country of domicile of the mobile subscriber. Mobile Network Code (MNC): A two or three-digit code that identifies the home Public Land Mobile Network (PLMN) of the subscriber within the country. Mobile Subscriber Identification Number (MSIN): A unique identification number for the mobile subscriber within the PLMN.",
    )

def test_3gpp_ts_23_003_2():
    assert query_and_validate(
        question="Of what does the TMSI cinsists of?",
        expected_response="The TMSI consists of 4 octets.",
    )    

def test_3gpp_ts_23_003_3():
    assert query_and_validate(
        question="What are Voice Group Call and Voice Broadcast Call References?",
        expected_response='Specific instances of voice group calls (VGCS) and voice broadcast calls (VBS) within a given group call area are known by a "Voice Group Call Reference" or by a "Voice Broadcast Call Reference" respectively.',
    )   


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )