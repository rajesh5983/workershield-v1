"""
Golden dataset for WorkerShield RAGAS evaluation.

8 queries spanning all three domains (SafeShift, FairDesk, HealthNav),
each with a ground-truth answer and the expected document IDs that should
appear in retrieved context.
"""

GOLDEN_DATASET = [
    {
        "query": "What are my obligations if a FIFO worker has a mental health condition and wants to reduce hours?",
        "expected_domains": ["fairdesk", "healthnav"],
        "ground_truth": (
            "Employers have duties under WHS law to manage psychosocial hazards including mental health, "
            "and under Fair Work to consider flexible working requests for employees with disability or "
            "caring responsibilities."
        ),
        "expected_doc_ids": ["HN01", "FD03"],
    },
    {
        "query": "What are the rules around casual employee overtime on public holidays?",
        "expected_domains": ["fairdesk"],
        "ground_truth": (
            "Casual employees are entitled to public holiday provisions under the NES, including penalty "
            "rates and the right to refuse to work on public holidays in some circumstances."
        ),
        "expected_doc_ids": ["FD01", "FD02"],
    },
    {
        "query": "What psychosocial hazards must I manage under WHS law?",
        "expected_domains": ["safeshift", "healthnav"],
        "ground_truth": (
            "PCBUs must manage psychosocial hazards including job demands, low job control, poor support, "
            "lack of role clarity, poor organisational change management, and exposure to traumatic events, "
            "as far as reasonably practicable."
        ),
        "expected_doc_ids": ["HN01", "SS03b"],
    },
    {
        "query": "What are my obligations when a worker is injured and needs to return to work?",
        "expected_domains": ["healthnav"],
        "ground_truth": (
            "Employers must provide suitable duties where possible, maintain communication with the injured "
            "worker, and develop a return to work plan in consultation with the worker and treating practitioners."
        ),
        "expected_doc_ids": ["HN03"],
    },
    {
        "query": "What is a Code of Practice under WHS legislation?",
        "expected_domains": ["safeshift"],
        "ground_truth": (
            "A Code of Practice provides practical guidance on how to achieve the standards required under "
            "the WHS Act and Regulations, and can be used as evidence of what is reasonably practicable."
        ),
        "expected_doc_ids": ["SS01", "SS02"],
    },
    {
        "query": "What entitlements does a casual employee have under the NES?",
        "expected_domains": ["fairdesk"],
        "ground_truth": (
            "Casual employees are entitled to unpaid carer's leave, unpaid compassionate leave, community "
            "service leave, and the Fair Work Information Statement and Casual Employment Information Statement."
        ),
        "expected_doc_ids": ["FD01", "FD02"],
    },
    {
        "query": "What does the WHS Act say about a worker's duty to take reasonable care?",
        "expected_domains": ["safeshift"],
        "ground_truth": (
            "Workers must take reasonable care for their own health and safety, take reasonable care that "
            "their acts or omissions do not adversely affect others, comply with reasonable instructions, "
            "and cooperate with reasonable policies and procedures."
        ),
        "expected_doc_ids": ["SS03a", "SS03b"],
    },
    {
        "query": "How can fatigue affect workplace safety?",
        "expected_domains": ["healthnav"],
        "ground_truth": (
            "Fatigue can impair alertness, judgement, reaction time and decision-making, increasing the "
            "risk of incidents. Employers must manage fatigue as a hazard under WHS law, especially for "
            "shift workers."
        ),
        "expected_doc_ids": ["HN02"],
    },
]
