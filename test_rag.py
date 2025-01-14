from query_data import query_rag
from langchain_community.llms.ollama import Ollama


#using local llm for unit testing

from query_data import query_rag
from langchain_ollama import OllamaLLM


#using local llm for unit testing

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


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

def test_3gpp_ts_31_101():
    assert query_and_validate(
        question="What is the main purpose of the 3GPP TS 31.101 specification?",
        expected_response="It defines the UICC-terminal interface, covering physical and logical characteristics for 3GPP telecom network operation​.",
    )

def test_3gpp_ts_31_120():
    assert query_and_validate(
        question="Which specification defines the physical, electrical, and logical test specifications for the UICC-terminal interface?",
        expected_response="3GPP TS 31.120​.",
    )

def test_efimpu():
    assert query_and_validate(
        question="What is the purpose of EFIMPU in the ISIM application?",
        expected_response="EFIMPU contains one or more public SIP identities (SIP URIs) of the user, with the first record used for emergency registration or as the default SIP identity​.",
    )

def test_uicc_terminal_classes():
    assert query_and_validate(
        question="What are the four classes of operating conditions specified for the UICC-terminal electrical interface?",
        expected_response="Class A, Class B, Class C, and Class D​.",
    )

def test_hpsim_application():
    assert query_and_validate(
        question="Describe the purpose of the HPSIM application.",
        expected_response="The HPSIM application provides mechanisms for authentication and provisioning of the Hosting Party in a Home (evolved) Node B (H(e)NB) environment​.",
    )

def test_encode_nai():
    assert query_and_validate(
        question="Generate code to encode a Network Access Identifier (NAI) for EFIMPI in UTF-8 as specified in RFC 3629.",
        expected_response="""from urllib.parse import quote

def encode_nai(nai):
    return quote(nai, safe='')

# Example usage
encoded_nai = encode_nai("user@example.com")
print(encoded_nai)""",
    )

def test_extract_domain():
    assert query_and_validate(
        question="Write a function to extract the domain name from an EFDOMAIN file encoded as a UTF-8 string.",
        expected_response="""def extract_domain(encoded_domain):
    return bytes.fromhex(encoded_domain).decode('utf-8')

# Example usage
encoded = "6578616d706c652e636f6d"
print(extract_domain(encoded))  # Output: example.com""",
    )

def test_efad():
    assert query_and_validate(
        question="What is the significance of EFAD in the ISIM application?",
        expected_response="EFAD provides operational mode information, such as normal operation or type approval modes. It also includes terminal feature activation indications​.",
    )

def test_hpsim_authentication():
    assert query_and_validate(
        question="What authentication mechanism does the HPSIM application use?",
        expected_response="HPSIM uses EAP-AKA for authentication​.",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OllamaLLM(model="mistral")
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