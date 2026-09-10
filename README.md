\# Hiver SDE Intern Take-Home — AppleSupport AI Support Agent



\## Overview



This project builds an AI support agent for the AppleSupport brand using the Customer Support on Twitter (TWCS) dataset.



Given a customer message, the agent:



1\. Classifies the customer's primary support intent.

2\. Retrieves historically similar AppleSupport support interactions.

3\. Generates a response grounded in those historical interactions.

4\. Decides whether the response should be auto-handled or escalated to a human.

5\. Provides a reason and supporting evidence for the decision.



The design prioritizes safe automation: when the system lacks sufficient confidence or supporting evidence, it can escalate instead of confidently inventing an answer.



\---



\## Architecture



```text

Customer Message

&#x20;     |

&#x20;     v

Intent Classification

(Gemini)

&#x20;     |

&#x20;     v

Historical Retrieval

(TF-IDF over AppleSupport cases)

&#x20;     |

&#x20;     v

Grounded Response Generation

(Gemini)

&#x20;     |

&#x20;     v

Trust / Escalation Gate

&#x20;     |

&#x20;     +----------------------+

&#x20;     |                      |

&#x20;     v                      v

&#x20;AUTO-HANDLE             ESCALATE

&#x20;     |                      |

&#x20;     v                      v

&#x20;Draft reply             Human review

